r"""
文件作用：S1、S2、S3、S4与顺序微调的五随机种子重复策略训练。
输入：原五折清单、九通道影像数据，以及本seed各折的全时期预训练模型。
流程：外层seed=40--44循环五次；每次内层完整五折，各折再进行逐年训练/评价。
输出：每个seed的年度各折权重、验证预测和五折精度；再汇总五组五折均值。
机制：S1每年重置同折全时期模型，顺序微调继承上一年；S2/S4单年训练，S3不微调。
参数：各组各折/年度训练均用该组seed，不加折号/年份偏移；--seed 42只运行一组五折。
环境依赖：train/training.py、NumPy、Pandas；地图预测、评价模块和原数据均不改动。
命令：cd /d F:\LCZ\reviewer_code
      D:\Anaconda\envs\LCZ\python.exe -B -m strategies.run_strategy_comparison --strategy s1
      D:\Anaconda\envs\LCZ\python.exe -B -m strategies.run_strategy_comparison --strategy sequential_yft
说明：本次只定义训练设置，未重新生成论文数值或地图。
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.metrics import save_metric_outputs, calculate_metrics, save_confusion_figure
from train.training import load_region_features, validate_manifest
from train.training import fit_stage
from evaluation.metrics import save_scores
from train.training import prepare_run
from config import (
    get_experiment_config, get_training_strategy_settings,
    ensure_training_parameters_confirmed, get_temporal_mechanism_settings,
)
from train.training import (
    resolve_sf_ganet_reference, reference_fold_files, resume_signature,
    configure_experiment, repetition_seeds, save_seed_summary,
)


def strategy_config(strategy_name, seed=None):
    # 策略仅改变初始化、预训练、年度训练及损失，不改变区域清单。
    cfg = configure_experiment('sf_ganet', seed=seed)
    settings = get_training_strategy_settings(strategy_name)
    cfg['experiment']['name'] = f'training_strategy_{strategy_name}'
    cfg['experiment']['use_supcon'] = settings['use_supcon']
    profile = deepcopy(cfg['strategy_training_profiles'][strategy_name])
    cfg['training'].update(deepcopy(profile.get('training', {})))
    cfg['parameter_provenance'] = profile
    cfg['paths']['experiment_output_root'] = (
        cfg['paths']['output_root'] / f'training_strategy_{strategy_name}'
    )
    if seed is not None:
        cfg['training']['seed'] = seed
    return cfg, settings


def annual_config(cfg, settings):
    # 年度微调使用已确认预算；S2/S4使用各自完整单年预算，待确认项由上游拦截。
    annual = deepcopy(cfg)
    if settings['global_pretrain'] and settings['yearly_train']:
        annual['training']['learning_rate'] = cfg['training']['finetune_learning_rate']
        annual['training']['num_epochs'] = cfg['training']['finetune_epochs']
    return annual


def run_strategy(strategy_name, seed=None, pretrain_run=None, resume=None):
    seeds = repetition_seeds(seed)
    if resume and len(seeds) != 1:
        raise ValueError('--resume需要同时指定--seed。')
    seed_metrics, outputs = [], []
    for run_seed in seeds:
        if strategy_name == 'sequential_yft':
            cfg, settings = temporal_config(strategy_name, run_seed)
        else:
            cfg, settings = strategy_config(strategy_name, run_seed)
        # 可通过{seed}模板为各重复显式提供自己的预训练来源，避免五组混用seed=42。
        source = str(pretrain_run).format(seed=run_seed) if pretrain_run is not None else None
        print(f'{strategy_name}随机种子重复 {run_seed}：完整五折', flush=True)
        output = run_configured_strategy(cfg, settings, pretrain_run=source, resume=resume)
        fold_metrics = pd.read_csv(output / 'cv_summary' / 'fold_metrics.csv')
        averages = fold_metrics[['oa', 'aa', 'kappa', 'macro_f1']].mean().to_dict()
        averages.update(seed=run_seed, fold_count=cfg['evaluation']['n_splits'])
        seed_metrics.append(averages)
        outputs.append(output)
    summary_root = cfg['paths']['experiment_output_root'] / cfg['evaluation']['output_mode']
    save_seed_summary(summary_root / ('seed_summary' if seed is None else f'seed{seed}_summary'),
                      seed_metrics, [], [])
    return outputs


def reference_predictions(source, data, cfg, year, fold=None):
    # S3直接按年份拆分来源的已保存预测，不重复预训练或前向计算；loss无法由类别还原。
    if fold is None:
        raise ValueError('当前策略只评价对应验证折，不存在固定测试集。')
    base = data['fold'] == fold
    path = source / f'validation_fold_{fold + 1}' / 'predictions.csv'
    saved = pd.read_csv(path)
    if not np.array_equal(saved.target.to_numpy(), data['y'][base]):
        raise ValueError('来源预测顺序或标签与当前清单不一致，不能拆分复用。')
    selected = data['year'][base] == year
    truth = saved.target.to_numpy()[selected]
    prediction = saved.prediction.to_numpy()[selected]
    metrics, classes, matrix = calculate_metrics(prediction, truth, cfg['data']['num_classes'])
    return np.nan, metrics, classes, matrix, prediction, truth


def choose_annual_checkpoint(settings, global_checkpoint, previous_checkpoint):
    # AP-YFT每年重置；连续微调只传模型参数，不传优化器或学习率调度器状态。
    if settings['annual_initialization'] == 'previous_year':
        return previous_checkpoint if previous_checkpoint is not None else global_checkpoint
    return global_checkpoint


def run_annual_stage(
    data, train, validation, cfg, settings, output, seed,
    initial_checkpoint, percentiles,
):
    # S1--S4与顺序微调保留原监督训练接口。
    checkpoint, percentiles, epoch, result = fit_stage(
        data, train, validation, cfg, output, settings['image_net'], seed,
        initial_checkpoint=initial_checkpoint, percentiles=percentiles,
    )
    record = {
        'supervision': 'ground_truth', 'status': 'adapted',
        'candidate_count': int(train.sum()), 'selected_count': int(train.sum()),
        'effective_epochs': epoch,
    }
    return checkpoint, percentiles, epoch, result, record


def record_annual_stage(
    records, output, stage, fold, year, initial_checkpoint, checkpoint, record,
):
    # 每完成一年记录实际初始化和输出权重，供追踪年度训练关系。
    records.append({
        'stage': stage, 'fold': fold, 'year': year,
        'initial_checkpoint': str(initial_checkpoint) if initial_checkpoint else '',
        'output_checkpoint': str(checkpoint), **record,
    })
    pd.DataFrame(records).to_csv(output / 'annual_provenance.csv', index=False)


def run_configured_strategy(cfg, settings, pretrain_run=None, resume=None):
    # 固定协议与数据提取保持唯一；新机制只控制年度初始权重和监督来源。
    ensure_training_parameters_confirmed(cfg)
    cfg, settings = deepcopy(cfg), deepcopy(settings)
    settings.setdefault('annual_initialization', 'shared_global')
    settings.setdefault(
        'annual_supervision', 'ground_truth' if settings['yearly_train'] else 'none',
    )
    if settings['annual_initialization'] not in {'shared_global', 'previous_year'}:
        raise ValueError('不支持的年度初始化机制。')
    if settings['annual_supervision'] not in {'ground_truth', 'none'}:
        raise ValueError('不支持的年度监督来源。')
    if settings['yearly_train'] and settings['annual_supervision'] == 'none':
        raise ValueError('年度训练不能缺少监督方式。')
    if settings['annual_initialization'] == 'previous_year' and not settings['global_pretrain']:
        raise ValueError('当前顺序对照要求同折全时期预训练及其固定归一化参数。')
    years = cfg['data']['years']
    if years != sorted(set(years)):
        raise ValueError('年度训练要求按年份递增且不能重复。')
    cfg['active_training_mechanism'] = settings
    # 所有含全时期预训练的机制复用同一个已完成主实验，先验证来源再创建任何输出。
    reference = None
    if settings['global_pretrain']:
        reference = resolve_sf_ganet_reference(cfg, pretrain_run)
        cfg['reference_pretrain_run'] = str(reference)
    elif pretrain_run is not None:
        raise ValueError('S2/S4不含全时期预训练，不能指定预训练来源。')
    # 固定训练协议；材料只保留S1--S4及顺序微调。
    cfg.pop('comparison_anchor', None)
    if resume is not None:
        # 运行级恢复还要核对年度机制及预训练来源，不能只比较通用训练签名。
        resume_path = Path(resume).resolve()
        saved = json.loads((resume_path / 'run_config.json').read_text(encoding='utf-8'))
        if (saved.get('active_training_mechanism') != settings
                or saved.get('reference_pretrain_run') != cfg.get('reference_pretrain_run')):
            raise ValueError('指定目录的年度机制或预训练来源与当前策略不一致。')
        cfg['runtime'] = {'resume': str(resume_path), 'new_run': False}
    # 单项命令明确再次运行时新建编号目录；--all由批调度器提前跳过完整结果。
    previous_root = cfg['paths']['experiment_output_root'] / cfg['evaluation']['output_mode']
    for marker in sorted(previous_root.glob('*/run_complete.json'), reverse=True):
        saved = json.loads((marker.parent / 'run_config.json').read_text(encoding='utf-8'))
        if (resume_signature(saved) == resume_signature(cfg)
                and saved.get('active_training_mechanism') == settings
                and saved.get('reference_pretrain_run') == cfg.get('reference_pretrain_run')):
            print(f'已有完成策略：{marker.parent}；本次单项重跑使用新的seed编号目录。', flush=True)
            break
    records, output = prepare_run(cfg)
    if (output / 'run_complete.json').exists():
        print(f'策略实验已完成：{output}', flush=True)
        return output
    if reference is not None:
        (output / 'pretrain_reference.json').write_text(
            json.dumps({'source_run': str(reference), 'reuse': 'matching_cv_folds_only'}, indent=2),
            encoding='utf-8',
        )
    data = load_region_features(records, cfg)
    dev = data['role'] == 'development'
    annual = annual_config(cfg, settings)
    global_epochs = []
    annual_epochs = {year: [] for year in cfg['data']['years']}
    validation_records, fold_metrics, provenance, artifacts = [], [], [], []
    fold_confusion_matrices = []
    for fold in range(cfg['evaluation']['n_splits']):
        train = dev & (data['fold'] != fold)
        validation = dev & (data['fold'] == fold)
        fold_output = output / f'fold_{fold + 1}'
        global_checkpoint, global_percentiles = None, None
        if settings['global_pretrain']:
            global_checkpoint, percentile_path, epoch = reference_fold_files(reference, fold + 1)
            global_percentiles = np.load(percentile_path, allow_pickle=False)
            print(f'复用第{fold + 1}折预训练：{global_checkpoint}', flush=True)
            global_epochs.append(epoch)
        yearly_metrics, previous_checkpoint = [], None
        # 一折内累加各年混淆计数；不改变原有年度等权精度汇总口径。
        fold_matrix = np.zeros((cfg['data']['num_classes'], cfg['data']['num_classes']), dtype=np.int64)
        for year in cfg['data']['years']:
            year_mask = data['year'] == year
            initial_checkpoint = choose_annual_checkpoint(
                settings, global_checkpoint, previous_checkpoint,
            )
            if settings['yearly_train']:
                checkpoint, percentiles, epoch, result, stage_record = run_annual_stage(
                    data, train & year_mask, validation & year_mask, annual, settings,
                    fold_output / f'year_{year}',
                    cfg['training']['seed'],
                    initial_checkpoint, global_percentiles,
                )
                annual_epochs[year].append(epoch)
                previous_checkpoint = checkpoint
            else:
                # S3用同一折全时期模型评价各年，不做年度微调或单年分位数重算。
                result = reference_predictions(reference, data, cfg, year, fold)
                checkpoint, percentiles = global_checkpoint, global_percentiles
                stage_record = {
                    'supervision': 'none', 'status': 'pooled_only',
                    'effective_epochs': 0,
                }
            # 保存该年度实际使用的归一化分位数。
            artifact_root = fold_output / f'year_{year}'
            artifact_root.mkdir(parents=True, exist_ok=True)
            percentile_file = artifact_root / 'selection_percentiles.npy'
            np.save(percentile_file, percentiles)
            artifacts.append(dict(seed=cfg['training']['seed'], fold=fold + 1, year=year, checkpoint_path=str(checkpoint),
                                  percentiles_path=str(percentile_file)))
            record_annual_stage(
                provenance, output, 'selected_validation_fold', fold + 1, year,
                initial_checkpoint, checkpoint, stage_record,
            )
            record = save_scores(
                result, fold_output / f'validation_{year}', 'selected_validation_fold',
                {'seed': cfg['training']['seed'], 'fold': fold + 1, 'year': year},
            )
            validation_records.append(record)
            yearly_metrics.append(record)
            fold_matrix += result[3]
        averages = pd.DataFrame(yearly_metrics)[['oa', 'aa', 'kappa', 'macro_f1']].mean().to_dict()
        averages.update(seed=cfg['training']['seed'], fold=fold + 1,
                        evaluation_scope='selected_validation_fold_year_equal_mean')
        fold_metrics.append(averages)
        fold_confusion_matrices.append(fold_matrix)
        np.save(fold_output / 'confusion_matrix.npy', fold_matrix)
        save_confusion_figure(fold_output / 'confusion_matrix.npy')
    pd.DataFrame(validation_records).to_csv(output / 'year_fold_metrics.csv', index=False)
    save_metric_outputs(output / 'cv_summary', fold_metrics, [], fold_confusion_matrices)

    # 按年汇总五折均值和标准差，保留实际年度权重。
    summary = pd.DataFrame(validation_records)
    summary.groupby('year')[['oa', 'aa', 'kappa', 'macro_f1']].agg(['mean', 'std']).to_csv(
        output / 'year_cv_summary.csv',
    )
    pd.DataFrame(artifacts).to_csv(output / 'fold_artifacts.csv', index=False)
    # 保留年度及五折指标、实际年度权重；跨折制图选择属于原项目的运行管理。
    (output / 'selected_training_plan.json').write_text(
        json.dumps({'global_best_epochs_by_fold': global_epochs,
                    'annual_best_epochs_by_fold': annual_epochs,
                    'evaluation_scope': 'selected_validation_fold'},
                   ensure_ascii=False, indent=2), encoding='utf-8',
    )
    (output / 'run_complete.json').write_text(
        json.dumps({'completed': True, 'seed': cfg['training']['seed'],
                    'reference_pretrain_run': str(reference) if reference else None}),
        encoding='utf-8',
    )
    print(f'STRATEGY_FIVE_FOLD_COMPLETE={output}', flush=True)
    return output


def temporal_config(mechanism_name, seed=None):
    # 顺序微调使用同一主网络和预训练预算，年度初始模型改为前一年参数。
    cfg = configure_experiment('sf_ganet', seed=seed)
    settings = get_temporal_mechanism_settings(mechanism_name)
    cfg['experiment']['name'] = f'temporal_training_{mechanism_name}'
    cfg['experiment']['use_supcon'] = settings['use_supcon']
    cfg['paths']['experiment_output_root'] = (
        cfg['paths']['output_root'] / f'temporal_training_{mechanism_name}'
    )
    cfg['active_training_mechanism'] = deepcopy(settings)
    # 各机制维持原匹配设置：主模型预训练、年度微调最多5轮，不改变损失。
    cfg['parameter_provenance'] = {
        'status': 'matched_to_s1',
        'source': '论文主模型预训练与最多5轮年度适配',
    }
    if seed is not None:
        cfg['training']['seed'] = seed
    return cfg, settings


def main():
    parser = argparse.ArgumentParser(description='S1--S4及顺序微调：40--44各重复完整五折')
    parser.add_argument('--strategy', required=True, choices=['s1', 's2', 's3', 's4', 'sequential_yft'])
    parser.add_argument('--seed', type=int, choices=[40, 41, 42, 43, 44],
                        help='省略时循环40--44；--seed 42只运行一组五折')
    parser.add_argument('--pretrain_run', help='预训练来源，可用{seed}作为五组路径模板')
    parser.add_argument('--resume')
    parser.add_argument('--check_only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        for run_seed in repetition_seeds(args.seed):
            if args.strategy == 'sequential_yft':
                cfg, settings = temporal_config(args.strategy, run_seed)
            else:
                cfg, settings = strategy_config(args.strategy, run_seed)
            ensure_training_parameters_confirmed(cfg)
            validate_manifest(cfg)
            print(f"{args.strategy}: seed={run_seed}；五折数={cfg['evaluation']['n_splits']}；{settings}")
        return
    run_strategy(args.strategy, args.seed, args.pretrain_run, args.resume)


if __name__ == '__main__':
    main()
