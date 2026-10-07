r"""
文件作用：主模型五随机种子重复训练，40--44每个seed均完成原五折评价。
输入：九通道斑块和y/year/fold/role；沿用原五折区域清单，不固定某一验证折。
输出：每个seed的各折模型、分位数、预测和五折汇总；再汇总五个seed的五折均值。
参数：省略--seed时外层循环40--44，内层循环五折，共25次折训练；--seed 42只做一组五折。
随机规则：每一组中的各折及各年度阶段均使用该组seed，不另加折号或年份偏移。
环境依赖：PyTorch、NumPy、Pandas、tqdm和同目录数据/损失模块；评价公式保持原样。
命令：cd /d F:\LCZ\reviewer_code
      D:\Anaconda\envs\LCZ\python.exe -B -m train.training --new_run
      D:\Anaconda\envs\LCZ\python.exe -B -m train.training --seed 42
说明：仅修改审稿材料的训练设置，未运行训练或改写已有结果。
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
import random
from time import perf_counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluation.metrics import calculate_metrics
import argparse
from contextlib import contextmanager
from evaluation.metrics import save_metric_outputs, save_confusion_figure
from datetime import datetime
from evaluation.metrics import save_scores
from config import get_experiment_config, config, ensure_training_parameters_confirmed


from train.data_handler import (
    calculate_percentiles_from_patches,
    create_evaluation_loader,
    load_region_features,
    prepare_dataloaders,
    validate_manifest,
)
from train.losses import SupConLoss, focal_loss


# 五个随机种子各完成原五折评价；默认单次种子和地图使用42。
repetition_seed_values = [40, 41, 42, 43, 44]
repetition_output_mode = 'region_five_fold_seed_repeats'


def save_best_checkpoint(model, checkpoint_path, epoch, validation_metrics, run_config=None):
    """立即写出 state_dict 的深拷贝，隔离后续 epoch 的原地参数更新。"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": deepcopy(model.state_dict()),
        "epoch": int(epoch),
        "validation_metrics": deepcopy(validation_metrics),
        "run_config": deepcopy(run_config) if run_config is not None else None,
    }
    torch.save(checkpoint, checkpoint_path)


def load_model_checkpoint(model, checkpoint_path, device):
    """将 checkpoint 中的模型参数加载至给定模型，并返回完整元数据。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def is_metric_improved(current_metrics, best_metrics, selection_metric):
    """按预先指定的验证指标判断当前模型是否应覆盖最佳 checkpoint。"""
    if best_metrics is None:
        return True
    if selection_metric not in current_metrics:
        raise ValueError(f"当前验证指标不含 {selection_metric}")
    if selection_metric not in best_metrics:
        raise ValueError(f"最佳验证指标不含 {selection_metric}")
    current = current_metrics[selection_metric]
    best = best_metrics[selection_metric]
    return current > best or (
        current == best and current_metrics['oa'] > best_metrics['oa']
    )


def resume_signature(run_config):
    # 只比较影响实验的配置，忽略输出目录、日志标签及本次启动控制参数。
    config = run_config or {}
    training = {k: v for k, v in config.get('training', {}).items()
                if not k.startswith(('log_', 'legacy_', 'resume_'))}
    fields = {key: config.get(key) for key in ['data', 'evaluation', 'experiment']}
    # 原主模型断点可能附带其他模型配置；这些字段不参与 SF-GANet 的续跑匹配。
    fields['data'] = deepcopy(config.get('data', {}))
    fields['data'].pop('recent_baselines', None)
    training.pop('recent_batch_size', None)
    fields['training'] = training
    return json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str)


def capture_random_state():
    # 当前项目DataLoader默认num_workers=0且采样器使用全局随机数，可按epoch恢复。
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    # 必须在模型、优化器和DataLoader构建后恢复，以抵消初始化消费的随机数。
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_epoch_checkpoint(path, model, optimizer, scheduler, epoch, history,
                          run_config, extra=None):
    # 临时写入成功后原子替换，保留上一完整epoch直到新状态完整落盘。
    path = Path(path)
    state = {
        'format_version': 1, 'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch, 'history': history, 'random_state': capture_random_state(),
        'signature': resume_signature(run_config),
    }
    state.update(extra or {})
    temporary = path.with_name(path.name + f'.writing_{os.getpid()}')
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_epoch_checkpoint(path, model, optimizer, scheduler, run_config):
    # 不允许更换数据/训练参数后无提示地继承优化器状态。
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('signature') != resume_signature(run_config):
        raise ValueError('续跑配置不一致，请使用原参数，或明确新开运行。')
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler_state'])
    restore_random_state(state['random_state'])
    return state


def resolve_sf_ganet_reference(cfg, source_run=None):
    # 每个seed复用自己完整五折的全时期模型，年度适配再按对应折读取权重。
    source = (Path(source_run).resolve() if source_run is not None else
              Path(cfg['paths']['output_root']) / 'sf_ganet' /
              cfg['evaluation']['output_mode'] / f"sf_ganet_seed{cfg['training']['seed']}")
    required = ['run_complete.json', 'run_config.json', 'split_manifest.csv',
                'cv_summary/metrics_summary.csv']
    if any(not (source / name).is_file() for name in required):
        raise FileNotFoundError(f'缺少当前seed的完整五折预训练结果：{source}')
    saved = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    expected = deepcopy(cfg)
    expected['experiment'] = {'name': 'sf_ganet', 'model_name': 'sf_ganet', 'use_supcon': True}
    if resume_signature(saved) != resume_signature(expected):
        raise ValueError('预训练来源的seed、五折划分或训练设置不一致。')
    if not json.loads((source / 'run_complete.json').read_text(encoding='utf-8')).get('completed'):
        raise ValueError('预训练尚未完整完成。')
    if not pd.read_csv(source / 'split_manifest.csv').equals(validate_manifest(cfg)):
        raise ValueError('预训练数据清单不同。')
    for fold in range(1, cfg['evaluation']['n_splits'] + 1):
        reference_fold_files(source, fold)
    return source


def reference_fold_files(source, fold):
    # 一组五折始终采用该组seed，不能再把折号加到seed上。
    source = Path(source).resolve()
    record = pd.read_csv(source / f'validation_fold_{fold}' / 'metrics.csv').iloc[0]
    expected = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    if (int(record['seed']) != expected['training']['seed']
            or int(record['fold']) != fold
            or record['evaluation_scope'] != 'selected_validation_fold'):
        raise ValueError('预训练指标不属于当前seed的对应验证折。')
    checkpoint = Path(record['checkpoint_path']).resolve()
    if not checkpoint.is_relative_to(source):
        raise ValueError('权重不在指定预训练结果目录内。')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (int(saved['epoch']) != int(record['best_epoch'])
            or resume_signature(saved.get('run_config')) != resume_signature(expected)):
        raise ValueError('预训练权重与seed、折号或配置不匹配。')
    return checkpoint, checkpoint.parent / 'train_percentiles.npy', int(record['best_epoch'])


def set_random_seeds(seed):
    """统一设置 Python、NumPy 与 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_model_output(model_output):
    """兼容仅返回 logits 的基准模型与返回 features、logits 的 SupCon 模型。"""
    if isinstance(model_output, tuple):
        return model_output
    return None, model_output


def train_one_epoch(model, train_loader, optimizer, training_config, use_supcon):
    """完成一个训练 epoch，并返回平均总损失、分类损失和对比损失。"""
    model.train()
    # 沿用论文温度0.1及当前损失实现的默认基温0.1，不恢复旧脚本内部默认值。
    contrastive_loss = SupConLoss(training_config['supcon_temperature'])
    loss_records = {"loss": 0.0, "classification_loss": 0.0, "contrastive_loss": 0.0}
    if len(train_loader) == 0:
        raise ValueError("训练 DataLoader 为空，无法开始训练。")

    progress_bar = tqdm(train_loader, desc="训练", leave=False)
    for batch_index, (inputs, labels) in enumerate(progress_bar, start=1):
        inputs = inputs.to(training_config["device"])
        labels = labels.to(training_config["device"])
        optimizer.zero_grad()
        features, logits = split_model_output(model(inputs))
        classification_loss = focal_loss(logits, labels)
        current_contrastive_loss = torch.tensor(0.0, device=logits.device)
        if use_supcon:
            if features is None:
                raise ValueError("当前实验启用了 SupCon，但模型没有返回特征向量。")
            current_contrastive_loss = contrastive_loss(features, labels)
        total_loss = classification_loss + training_config["supcon_lambda"] * current_contrastive_loss
        total_loss.backward()
        optimizer.step()

        loss_records["loss"] += total_loss.item()
        loss_records["classification_loss"] += classification_loss.item()
        loss_records["contrastive_loss"] += current_contrastive_loss.item()
        progress_bar.set_postfix(loss=f"{loss_records['loss'] / batch_index:.4f}")

    return {name: value / len(train_loader) for name, value in loss_records.items()}


def evaluate_model(model, data_loader, data_config, training_config):
    """在给定评价数据上计算损失、主指标、逐类别指标和混淆矩阵，不更新权重。"""
    model.eval()
    total_loss = 0.0
    predictions, labels_list = [], []
    if len(data_loader) == 0:
        raise ValueError("评价 DataLoader 为空，无法计算指标。")

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(training_config["device"])
            labels = labels.to(training_config["device"])
            _, logits = split_model_output(model(inputs))
            total_loss += focal_loss(logits, labels).item()
            # 近期多分类器基线显式执行PKC；loss仍为主分支Focal，不伪造概率。
            prediction = model.predict(inputs, logits) if hasattr(model, 'predict') else logits.argmax(dim=1)
            predictions.extend(prediction.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    metrics, class_metrics, matrix = calculate_metrics(predictions, labels_list, data_config["num_classes"])
    return (
        total_loss / len(data_loader),
        metrics,
        class_metrics,
        matrix,
        np.asarray(predictions),
        np.asarray(labels_list),
    )


def write_training_message(message, log_path):
    # 普通print加文本追加，不引入额外日志框架；每条摘要同时保留在终端和文件。
    print(message, flush=True)
    if log_path is not None:
        with open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(message + '\n')


def dataset_counts(loader, num_classes):
    # 直接读取数据集标签，不再额外跑一遍模型。
    labels = getattr(loader.dataset, 'labels', None)
    if labels is None and hasattr(loader.dataset, 'tensors'):
        labels = loader.dataset.tensors[1]
    if labels is None:
        return '标签数量不可直接读取'
    counts = np.bincount(np.asarray(labels), minlength=num_classes)
    return ' | '.join(f'{i + 1}:{n}' for i, n in enumerate(counts))


def fit_model(
    model, train_loader, validation_loader, data_config, training_config,
    evaluation_config, output_dir, use_supcon, run_config,
):
    """保留原早停算法；每轮完整打印并及时落盘，不只记录创新高的轮次。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'best_checkpoint.pt'
    log_path = output_dir / 'training.log'
    history_path = output_dir / 'history.csv'
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=max(1, training_config['patience'] // 2),
        factor=0.2,
    )
    selection_metric = evaluation_config['selection_metric']
    experiment = (run_config or {}).get('experiment', {}).get('name', model.__class__.__name__)
    label = training_config.get('log_label', str(output_dir))
    write_training_message(
        f'实验={experiment} | 阶段={label}\n'
        f'设备={training_config["device"]} | seed={training_config["seed"]}\n'
        f'Train={len(train_loader.dataset)} | Validation={len(validation_loader.dataset)}\n'
        f'AdamW | batch={training_config["batch_size"]} | '
        f'lr={training_config["learning_rate"]:.2e} | weight_decay={training_config["weight_decay"]:.2e}\n'
        f'最多{training_config["num_epochs"]}轮 | patience={training_config["patience"]} | '
        f'选模={selection_metric} | 学习率监控=验证Focal损失\n'
        f'SupCon={use_supcon} | temperature={training_config["supcon_temperature"]} | '
        'base_temperature=0.1\n'
        f'训练类别计数：{dataset_counts(train_loader, data_config["num_classes"])}\n'
        f'验证类别计数：{dataset_counts(validation_loader, data_config["num_classes"])}\n'
        f'权重路径：{checkpoint_path}',
        log_path,
    )
    best_metrics, best_epoch = None, 0
    patience_counter, history = 0, []
    last_path = output_dir / 'last_checkpoint.pt'
    start_epoch = 1
    if last_path.exists():
        if training_config.get('num_workers', 0) != 0:
            raise ValueError('当前精确epoch恢复要求num_workers=0。')
        state = load_epoch_checkpoint(last_path, model, optimizer, scheduler, run_config)
        history = state['history']
        best_metrics, best_epoch = state['best_metrics'], state['best_epoch']
        patience_counter = state['patience_counter']
        start_epoch = state['epoch'] + 1
        # 与最后完整epoch同时保存的最佳快照恢复到磁盘，避免中断写入产生不一致。
        if state['best_checkpoint'] is not None:
            torch.save(state['best_checkpoint'], checkpoint_path)
            pd.DataFrame(state['best_class_records']).to_csv(
                output_dir / 'best_validation_class_metrics.csv', index=False,
            )
        write_training_message(f'精确epoch续跑：从第{start_epoch}轮开始，优化器/调度器/RNG已恢复。', log_path)
    elif checkpoint_path.exists() or history_path.exists():
        raise RuntimeError('旧阶段缺少完整last_checkpoint，不能自动从中断epoch恢复。')
    if not last_path.exists():
        # 保存epoch零状态，第一轮中断后也能恢复初始模型及随机状态。
        initial_best = (torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                        if checkpoint_path.exists() else None)
        initial_classes = (pd.read_csv(output_dir / 'best_validation_class_metrics.csv').to_dict('records')
                           if initial_best is not None else [])
        save_epoch_checkpoint(
            last_path, model, optimizer, scheduler, start_epoch - 1, history, run_config,
            {'best_metrics': best_metrics, 'best_epoch': best_epoch, 'patience_counter': patience_counter,
             'best_checkpoint': initial_best, 'best_class_records': initial_classes},
        )
    start_time = perf_counter()
    stop_reason = '达到最大训练轮数'
    if patience_counter >= training_config['patience']:
        start_epoch = training_config['num_epochs'] + 1
        stop_reason = '恢复时确认此前已触发早停'
    for epoch in range(start_epoch, training_config['num_epochs'] + 1):
        epoch_start = perf_counter()
        learning_rate = optimizer.param_groups[0]['lr']
        write_training_message(
            f'[{label}][Epoch {epoch}/{training_config["num_epochs"]}] 开始训练', log_path,
        )
        losses = train_one_epoch(model, train_loader, optimizer, training_config, use_supcon)
        validation_loss, metrics, class_metrics, _, _, _ = evaluate_model(
            model, validation_loader, data_config, training_config,
        )
        scheduler.step(validation_loss)
        next_learning_rate = optimizer.param_groups[0]['lr']
        improved = is_metric_improved(metrics, best_metrics, selection_metric)
        if improved:
            best_metrics, best_epoch = dict(metrics), epoch
            save_best_checkpoint(model, checkpoint_path, epoch, metrics, run_config)
            class_metrics.to_csv(output_dir / 'best_validation_class_metrics.csv', index=False)
            patience_counter = 0
        else:
            patience_counter += 1
        elapsed = perf_counter() - epoch_start
        history.append({
            'epoch': epoch, **losses, 'validation_loss': validation_loss, **metrics,
            'learning_rate': learning_rate, 'next_learning_rate': next_learning_rate,
            'epoch_seconds': elapsed, 'patience_counter': patience_counter,
            'best_epoch': best_epoch, 'checkpoint_updated': improved,
        })
        # last保存最后完整epoch的全部状态；best只用于选模，二者不能混用。
        best_snapshot = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        save_epoch_checkpoint(
            last_path, model, optimizer, scheduler, epoch, history, run_config,
            {'best_metrics': best_metrics, 'best_epoch': best_epoch,
             'patience_counter': patience_counter, 'best_checkpoint': best_snapshot,
             'best_class_records': pd.read_csv(
                 output_dir / 'best_validation_class_metrics.csv'
             ).to_dict('records')},
        )
        # 每轮写出历史，即使正常训练中途退出也能保留已完成epoch。
        pd.DataFrame(history).to_csv(history_path, index=False, encoding='utf-8-sig')
        write_training_message(
            f'训练：总损失={losses["loss"]:.4f} | Focal={losses["classification_loss"]:.4f} | '
            f'SupCon原值={losses["contrastive_loss"]:.4f} | '
            f'lambda={training_config["supcon_lambda"] if use_supcon else 0}\n'
            f'验证：Focal={validation_loss:.4f} | OA={metrics["oa"]:.4f} | '
            f'AA={metrics["aa"]:.4f} | Kappa={metrics["kappa"]:.4f} | Macro-F1={metrics["macro_f1"]:.4f}\n'
            f'学习率：本轮={learning_rate:.2e} | 下一轮={next_learning_rate:.2e}\n'
            f'最佳{selection_metric}={best_metrics[selection_metric]:.4f}（epoch {best_epoch}） | '
            f'早停计数={patience_counter}/{training_config["patience"]} | 耗时={elapsed:.1f}秒\n'
            f'权重：{"已更新" if improved else "未更新"}',
            log_path,
        )
        if next_learning_rate != learning_rate:
            write_training_message('验证损失触发学习率调整。', log_path)
        if patience_counter >= training_config['patience']:
            stop_reason = '验证选模指标连续未改善，触发早停'
            break
    if best_metrics is None:
        raise RuntimeError('训练没有产生最佳权重。')
    load_model_checkpoint(model, checkpoint_path, training_config['device'])
    pd.DataFrame(history).to_csv(history_path, index=False, encoding='utf-8-sig')
    write_training_message(
        f'阶段结束：{stop_reason} | 实际{len(history)}轮 | 最佳epoch={best_epoch} | '
        f'总耗时={perf_counter() - start_time:.1f}秒\n'
        f'最佳验证：OA={best_metrics["oa"]:.4f} | AA={best_metrics["aa"]:.4f} | '
        f'Kappa={best_metrics["kappa"]:.4f} | Macro-F1={best_metrics["macro_f1"]:.4f}\n'
        f'已恢复最佳权重：{checkpoint_path}', log_path,
    )
    # 每阶段结束只展示一次完整类别指标，避免每epoch刷出17行表格。
    best_classes = pd.read_csv(output_dir / 'best_validation_class_metrics.csv')
    write_training_message(best_classes.to_string(index=False), log_path)
    return model, best_metrics, history, checkpoint_path


def stage_model(cfg, pretrained, initial_checkpoint=None):
    # 初始权重由调用方决定：AP-YFT传全时期模型，顺序对照传前一年模型。
    from model.model import SF_GANet

    model = SF_GANet(
        cfg['data']['num_classes'], 9,
        pretrained if initial_checkpoint is None else False,
        cfg['experiment']['use_supcon'],
    )
    model = model.to(cfg['training']['device'])
    if initial_checkpoint is not None:
        load_model_checkpoint(model, initial_checkpoint, cfg['training']['device'])
    return model


def fit_stage(data, train, validation, cfg, output, pretrained, seed,
              initial_checkpoint=None, percentiles=None):
    # 验证阶段从不接受test成员；归一化只从本阶段训练侧计算或继承同折预训练。
    if not train.any() or not validation.any():
        raise ValueError('训练或验证子集为空。')
    if (train & validation).any() or (data['role'][train | validation] == 'test').any():
        raise ValueError('验证选模阶段出现测试成员或交叉样本。')
    cfg = deepcopy(cfg)
    output = Path(output)
    # 旧折暖启动产生新子目录后，之后默认运行可自动恢复该子目录的完整状态。
    if (output / 'continued_from_best' / 'last_checkpoint.pt').exists():
        output = output / 'continued_from_best'
    old_complete = False
    if (output / 'best_checkpoint.pt').exists() and not (output / 'last_checkpoint.pt').exists():
        # 有完整阶段结束证据的旧折可以复用；否则只有用户显式选择才允许暖启动。
        history_file = output / 'history.csv'
        old_history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
        finished = not old_history.empty and (
            int(old_history.iloc[-1].get('patience_counter', 0)) >= cfg['training']['patience']
            or int(old_history.iloc[-1]['epoch']) >= cfg['training']['num_epochs']
        )
        old_complete = finished
        if not finished:
            raise RuntimeError('旧折缺少完整断点，不能精确续跑；请使用--new_run。')
    cfg['training']['seed'] = seed
    cfg['training']['log_label'] = (
        f"{cfg['experiment']['name']}/{output.parent.name}/{output.name}"
    )
    cfg['training']['log_output_dir'] = str(output)
    set_random_seeds(seed)
    if percentiles is None:
        percentiles = calculate_percentiles_from_patches(data['x'][train], cfg['data'])
    train_loader, val_loader = prepare_dataloaders(
        data['x'][train], data['y'][train], data['x'][validation], data['y'][validation],
        percentiles, cfg['data'], cfg['training'],
    )
    # 有last状态或旧暖启动权重时无需重复下载ImageNet初始化。
    resume_weights = output / 'last_checkpoint.pt'
    model = stage_model(
        cfg, pretrained and not resume_weights.exists(), initial_checkpoint,
    )
    if old_complete:
        # 旧阶段已触发早停但未来得及保存验证汇总时，只补评价，不重新训练。
        checkpoint = output / 'best_checkpoint.pt'
        metadata = load_model_checkpoint(model, checkpoint, cfg['training']['device'])
        np.save(output / 'train_percentiles.npy', percentiles)
        result = evaluate_model(model, val_loader, cfg['data'], cfg['training'])
        print(f'复用已结束的旧阶段：{output}', flush=True)
        return checkpoint, percentiles, int(metadata['epoch']), result
    model, _, history, checkpoint = fit_model(
        model, train_loader, val_loader, cfg['data'], cfg['training'], cfg['evaluation'],
        output, cfg['experiment']['use_supcon'], cfg,
    )
    epoch = torch.load(checkpoint, map_location='cpu', weights_only=False)['epoch']
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
    np.save(output / 'train_percentiles.npy', percentiles)
    result = evaluate_model(model, val_loader, cfg['data'], cfg['training'])
    del model, train_loader, val_loader
    return checkpoint, percentiles, int(epoch), result


def score_checkpoint(data, selected, cfg, checkpoint, percentiles):
    # 单独评价已冻结模型，函数不更新模型参数或执行模型选择。
    model = stage_model(cfg, False, checkpoint)
    loader = create_evaluation_loader(data['x'][selected], data['y'][selected],
                                      percentiles, cfg['training'])
    result = evaluate_model(model, loader, cfg['data'], cfg['training'])
    del model, loader
    return result


def choose_run_directory(cfg):
    # 默认只续跑相同配置的未完成实验；无历史时首次创建，否则新实验必须显式指定。
    root = cfg['paths']['experiment_output_root'] / cfg['evaluation']['output_mode']
    # 新目录使用稳定名称；不重命名旧结果，重复实验用递增run编号避免覆盖。
    run_name = f"{cfg['experiment']['name']}_seed{cfg['training']['seed']}"
    fresh_path = root / run_name
    version = 2
    # 旧时间戳目录也算同seed的一次历史运行；保留旧目录，新重跑从run02开始。
    if not fresh_path.exists() and any(root.glob(f'{run_name}_*/run_config.json')):
        fresh_path = root / f'{run_name}_run02'
    while fresh_path.exists():
        fresh_path = root / f'{run_name}_run{version:02d}'
        version += 1
    runtime = cfg.get('runtime', {})
    if not runtime:
        return fresh_path, True
    if runtime.get('resume'):
        output = Path(runtime['resume']).resolve()
        saved = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))
        if resume_signature(saved) != resume_signature(cfg):
            raise ValueError('指定目录与当前实验配置不同。')
        return output, False
    if not runtime.get('new_run', False):
        histories = sorted(
            root.glob('*/run_config.json'), key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in histories:
            saved = json.loads(path.read_text(encoding='utf-8'))
            if (resume_signature(saved) == resume_signature(cfg)
                    and not (path.parent / 'run_complete.json').exists()):
                return path.parent, False
        # 同seed已完成后再次单独运行，或换seed时，使用新稳定/编号目录。
        # --all在进入本函数前判断完整结果并跳过，不因此重复训练。
    return fresh_path, True


def prepare_run(cfg):
    # 续跑不覆盖原始配置与分组清单，明确打印使用的原目录。
    ensure_training_parameters_confirmed(cfg)
    print(f"训练参数来源：{cfg.get('parameter_provenance', {})}", flush=True)
    records = validate_manifest(cfg)
    output, fresh = choose_run_directory(cfg)
    if fresh:
        output.mkdir(parents=True, exist_ok=False)
        records.to_csv(output / 'split_manifest.csv', index=False)
        (output / 'run_config.json').write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding='utf-8',
        )
    elif not pd.read_csv(output / 'split_manifest.csv').equals(records):
        raise ValueError('当前清单与原运行清单不同。')
    print(f'{"新实验" if fresh else "继续原实验"}：{output}', flush=True)
    return records, output


@contextmanager
def run_directory_lock(output):
    # 锁文件保留；操作系统释放锁后下次可用，不删除文件。
    stream = open(output / 'run.lock', 'a+b')
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('同一运行目录已有进程占用，请勿重复启动。')
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == 'nt':
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def configure_experiment(experiment_name, manifest=None, audit=None, seed=None):
    # 只调整训练配置副本；保留原五折清单，增加独立的五种子结果目录。
    cfg = get_experiment_config(experiment_name)
    cfg['evaluation']['output_mode'] = repetition_output_mode
    if manifest is not None:
        cfg['paths']['split_manifest'] = manifest
    if audit is not None:
        cfg['paths']['pixel_audit'] = audit
    if seed is not None:
        cfg['training']['seed'] = seed
    return cfg


def repetition_seeds(seed=None):
    # 无参数运行五次；显式--seed只运行指定种子，便于恢复或生成42模型。
    seeds = repetition_seed_values if seed is None else [seed]
    if any(value not in repetition_seed_values for value in seeds):
        raise ValueError('训练随机种子只允许40、41、42、43、44。')
    return list(seeds)




def run_model_experiment(experiment_name='sf_ganet', pretrained=True, manifest=None,
                         audit=None, seed=None, resume=None, new_run=False):
    seeds = repetition_seeds(seed)
    if resume and len(seeds) != 1:
        raise ValueError('--resume需要同时指定--seed。')
    seed_metrics, outputs = [], []
    for run_seed in seeds:
        cfg = configure_experiment(experiment_name, manifest, audit, run_seed)
        cfg['runtime'] = {'resume': resume, 'new_run': new_run}
        records, output = prepare_run(cfg)
        print(f'随机种子重复 {run_seed}：完整五折', flush=True)
        with run_directory_lock(output):
            run_model_stages(cfg, records, output, experiment_name, pretrained)
        fold_metrics = pd.read_csv(output / 'cv_summary' / 'fold_metrics.csv')
        averages = fold_metrics[['oa', 'aa', 'kappa', 'macro_f1']].mean().to_dict()
        averages.update(seed=run_seed, fold_count=cfg['evaluation']['n_splits'])
        seed_metrics.append(averages)
        outputs.append(output)
    # 先获得每个seed的五折均值，再汇总五个seed的均值和样本标准差。
    summary_root = cfg['paths']['experiment_output_root'] / cfg['evaluation']['output_mode']
    save_seed_summary(summary_root / ('seed_summary' if seed is None else f'seed{seed}_summary'),
                      seed_metrics, [], [])
    return outputs


def run_model_stages(cfg, records, output, experiment_name, pretrained):
    # 已完成阶段直接复用；续跑仍需重建内存数据，不重复已完成训练。
    if (output / 'run_complete.json').exists():
        print(f'实验已完成：{output}')
        return output
    # 旧不完整折先检查，避免读取20年影像后才发现缺少恢复所需状态。
    for fold in range(cfg['evaluation']['n_splits']):
        stage = output / f'fold_{fold + 1}'
        if (stage / 'restarted_from_scratch').exists():
            continue
        full = stage / 'last_checkpoint.pt'
        continued = stage / 'continued_from_best' / 'last_checkpoint.pt'
        if ((stage / 'best_checkpoint.pt').exists()
                and not full.exists() and not continued.exists()):
            history = pd.read_csv(stage / 'history.csv')
            ended = (int(history.iloc[-1].get('patience_counter', 0))
                     >= cfg['training']['patience']
                     or int(history.iloc[-1]['epoch']) >= cfg['training']['num_epochs'])
            if not ended:
                raise RuntimeError(
                    f'第{fold + 1}折旧文件缺少完整断点，不能精确续跑。'
                    '请使用--new_run开始新实验，原结果会保留。'
                )
    data = load_region_features(records, cfg)
    if pretrained is None:
        pretrained = True
    dev = data['role'] == 'development'
    metrics, classes, matrices, epochs = [], [], [], []
    for fold in range(cfg['evaluation']['n_splits']):
        saved = output / f'validation_fold_{fold + 1}'
        stage = output / f'fold_{fold + 1}'
        required = ['metrics.csv', 'class_metrics.csv', 'confusion_matrix.npy', 'predictions.csv']
        if all((saved / name).exists() for name in required):
            record = pd.read_csv(saved / 'metrics.csv').iloc[0].to_dict()
            if record.get('evaluation_scope') != 'selected_validation_fold':
                raise ValueError('旧折指标来源不一致。')
            epochs.append(int(record['best_epoch']))
            metrics.append(record)
            classes.append(pd.read_csv(saved / 'class_metrics.csv'))
            matrices.append(np.load(saved / 'confusion_matrix.npy'))
            # 旧运行已有指标但缺图时，仅补图，不重训已完成折。
            save_confusion_figure(saved / 'confusion_matrix.npy')
            print(f'跳过已完成验证折 {fold + 1}', flush=True)
            continue
        train = dev & (data['fold'] != fold)
        validation = dev & (data['fold'] == fold)
        if (stage / 'restarted_from_scratch').exists():
            stage = stage / 'restarted_from_scratch'
            print(f'第{fold + 1}折使用从头重跑分支：{stage}；原折文件保留。', flush=True)
        checkpoint, _, epoch, result = fit_stage(
            data, train, validation, cfg, stage,
            pretrained, cfg['training']['seed'],
        )
        epochs.append(epoch)
        record = save_scores(
            result, output / f'validation_fold_{fold + 1}', 'selected_validation_fold',
            {'seed': cfg['training']['seed'], 'fold': fold + 1, 'best_epoch': epoch, 'checkpoint_path': str(checkpoint),
             'continuation_mode': 'legacy_warm_start' if 'continued_from_best' in checkpoint.parts
             else 'restarted_fold' if 'restarted_from_scratch' in checkpoint.parts
             else 'full_state_or_fresh'},
        )
        metrics.append(record)
        classes.append(result[2])
        matrices.append(result[3])
    save_metric_outputs(output / 'cv_summary', metrics, classes, matrices)
    np.save(output / 'selected_epochs.npy', np.asarray(epochs))
    (output / 'run_complete.json').write_text(
        json.dumps({'completed': True, 'seed': cfg['training']['seed'],
                    'protocol': 'region_five_fold_seed_repeats',
                    'contains_legacy_warm_start': bool(list(
                        output.glob('fold_*/continued_from_best')
                    ))}), encoding='utf-8',
    )
    print(f'FIVE_FOLD_EXPERIMENT_COMPLETE={output}', flush=True)
    return output


def save_seed_summary(output_dir, seed_metrics, class_metrics_by_seed, confusion_matrices):
    # 各seed先完成五折，再独立汇总五组结果；共享评价接口保持原样。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(seed_metrics)
    if 'seed' not in table or table.seed.duplicated().any():
        raise ValueError('重复统计必须包含唯一的实际seed。')
    metric_names = ['oa', 'aa', 'kappa', 'macro_f1']
    summary = pd.DataFrame([
        {'metric': name, 'mean': table[name].mean(), 'std': table[name].std(ddof=1)}
        for name in metric_names
    ])
    table.to_csv(output_dir / 'seed_metrics.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(output_dir / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    class_frames = []
    for seed, class_metrics in zip(table.seed, class_metrics_by_seed):
        current = class_metrics.copy()
        current.insert(0, 'seed', seed)
        class_frames.append(current)
    if class_frames:
        pd.concat(class_frames, ignore_index=True).to_csv(
            output_dir / 'class_metrics_by_seed.csv', index=False, encoding='utf-8-sig',
        )
    if confusion_matrices:
        np.save(output_dir / 'confusion_matrices.npy', np.asarray(confusion_matrices))


def main():
    parser = argparse.ArgumentParser(description='种子40--44重复五次，每次完整五折')
    parser.add_argument('--manifest')
    parser.add_argument('--audit')
    parser.add_argument('--seed', type=int, choices=[40, 41, 42, 43, 44],
                        help='省略时循环40--44；指定42只运行地图使用的模型')
    parser.add_argument('--check_only', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume')
    mode.add_argument('--new_run', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        for run_seed in repetition_seeds(args.seed):
            cfg = configure_experiment('sf_ganet', args.manifest, args.audit, run_seed)
            validate_manifest(cfg)
            print(f"seed={run_seed}；五折数={cfg['evaluation']['n_splits']}")
        return
    run_model_experiment(
        manifest=args.manifest, audit=args.audit, seed=args.seed,
        resume=args.resume, new_run=args.new_run,
    )


if __name__ == '__main__':
    main()
