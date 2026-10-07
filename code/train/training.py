
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


repetition_seed_values = [40, 41, 42, 43, 44]
repetition_output_mode = 'region_five_fold_seed_repeats'


def save_best_checkpoint(model, checkpoint_path, epoch, validation_metrics, run_config=None):
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def is_metric_improved(current_metrics, best_metrics, selection_metric):
    if best_metrics is None:
        return True
    if selection_metric not in current_metrics:
        raise ValueError(f" {selection_metric}")
    if selection_metric not in best_metrics:
        raise ValueError(f" {selection_metric}")
    current = current_metrics[selection_metric]
    best = best_metrics[selection_metric]
    return current > best or (
        current == best and current_metrics['oa'] > best_metrics['oa']
    )


def resume_signature(run_config):
    config = run_config or {}
    training = {k: v for k, v in config.get('training', {}).items()
                if not k.startswith(('log_', 'legacy_', 'resume_'))}
    fields = {key: config.get(key) for key in ['data', 'evaluation', 'experiment']}
    fields['data'] = deepcopy(config.get('data', {}))
    fields['data'].pop('recent_baselines', None)
    training.pop('recent_batch_size', None)
    fields['training'] = training
    return json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str)


def capture_random_state():
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_epoch_checkpoint(path, model, optimizer, scheduler, epoch, history,
                          run_config, extra=None):
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
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('signature') != resume_signature(run_config):
        raise ValueError('，，。')
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler_state'])
    restore_random_state(state['random_state'])
    return state


def resolve_sf_ganet_reference(cfg, source_run=None):
    source = (Path(source_run).resolve() if source_run is not None else
              Path(cfg['paths']['output_root']) / 'sf_ganet' /
              cfg['evaluation']['output_mode'] / f"sf_ganet_seed{cfg['training']['seed']}")
    required = ['run_complete.json', 'run_config.json', 'split_manifest.csv',
                'cv_summary/metrics_summary.csv']
    if any(not (source / name).is_file() for name in required):
        raise FileNotFoundError(f'seed：{source}')
    saved = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    expected = deepcopy(cfg)
    expected['experiment'] = {'name': 'sf_ganet', 'model_name': 'sf_ganet', 'use_supcon': True}
    if resume_signature(saved) != resume_signature(expected):
        raise ValueError('seed、。')
    if not json.loads((source / 'run_complete.json').read_text(encoding='utf-8')).get('completed'):
        raise ValueError('。')
    if not pd.read_csv(source / 'split_manifest.csv').equals(validate_manifest(cfg)):
        raise ValueError('。')
    for fold in range(1, cfg['evaluation']['n_splits'] + 1):
        reference_fold_files(source, fold)
    return source


def reference_fold_files(source, fold):
    source = Path(source).resolve()
    record = pd.read_csv(source / f'validation_fold_{fold}' / 'metrics.csv').iloc[0]
    expected = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    if (int(record['seed']) != expected['training']['seed']
            or int(record['fold']) != fold
            or record['evaluation_scope'] != 'selected_validation_fold'):
        raise ValueError('seed。')
    checkpoint = Path(record['checkpoint_path']).resolve()
    if not checkpoint.is_relative_to(source):
        raise ValueError('。')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (int(saved['epoch']) != int(record['best_epoch'])
            or resume_signature(saved.get('run_config')) != resume_signature(expected)):
        raise ValueError('seed、。')
    return checkpoint, checkpoint.parent / 'train_percentiles.npy', int(record['best_epoch'])


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_model_output(model_output):
    if isinstance(model_output, tuple):
        return model_output
    return None, model_output


def train_one_epoch(model, train_loader, optimizer, training_config, use_supcon):
    model.train()
    contrastive_loss = SupConLoss(
        temperature=training_config['supcon_temperature'],
        base_temperature=training_config['supcon_temperature'],
    )
    loss_records = {"loss": 0.0, "classification_loss": 0.0, "contrastive_loss": 0.0}
    if len(train_loader) == 0:
        raise ValueError(" DataLoader ，。")

    progress_bar = tqdm(train_loader, desc="", leave=False)
    for batch_index, (inputs, labels) in enumerate(progress_bar, start=1):
        inputs = inputs.to(training_config["device"])
        labels = labels.to(training_config["device"])
        optimizer.zero_grad()
        features, logits = split_model_output(model(inputs))
        classification_loss = focal_loss(logits, labels)
        current_contrastive_loss = torch.tensor(0.0, device=logits.device)
        if use_supcon:
            if features is None:
                raise ValueError(" SupCon，。")
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
    model.eval()
    total_loss = 0.0
    predictions, labels_list = [], []
    if len(data_loader) == 0:
        raise ValueError(" DataLoader ，。")

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(training_config["device"])
            labels = labels.to(training_config["device"])
            _, logits = split_model_output(model(inputs))
            total_loss += focal_loss(logits, labels).item()
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
    print(message, flush=True)
    if log_path is not None:
        with open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(message + '\n')


def dataset_counts(loader, num_classes):
    labels = getattr(loader.dataset, 'labels', None)
    if labels is None and hasattr(loader.dataset, 'tensors'):
        labels = loader.dataset.tensors[1]
    if labels is None:
        return ''
    counts = np.bincount(np.asarray(labels), minlength=num_classes)
    return ' | '.join(f'{i + 1}:{n}' for i, n in enumerate(counts))


def fit_model(
    model, train_loader, validation_loader, data_config, training_config,
    evaluation_config, output_dir, use_supcon, run_config,
):
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
        f'={experiment} | ={label}\n'
        f'={training_config["device"]} | seed={training_config["seed"]}\n'
        f'Train={len(train_loader.dataset)} | Validation={len(validation_loader.dataset)}\n'
        f'AdamW | batch={training_config["batch_size"]} | '
        f'lr={training_config["learning_rate"]:.2e} | weight_decay={training_config["weight_decay"]:.2e}\n'
        f'{training_config["num_epochs"]} | patience={training_config["patience"]} | '
        f'={selection_metric} | =Focal\n'
        f'SupCon={use_supcon} | temperature={training_config["supcon_temperature"]} | '
        'base_temperature=0.1\n'
        f'：{dataset_counts(train_loader, data_config["num_classes"])}\n'
        f'：{dataset_counts(validation_loader, data_config["num_classes"])}\n'
        f'：{checkpoint_path}',
        log_path,
    )
    best_metrics, best_epoch = None, 0
    patience_counter, history = 0, []
    last_path = output_dir / 'last_checkpoint.pt'
    start_epoch = 1
    if last_path.exists():
        if training_config.get('num_workers', 0) != 0:
            raise ValueError('epochnum_workers=0。')
        state = load_epoch_checkpoint(last_path, model, optimizer, scheduler, run_config)
        history = state['history']
        best_metrics, best_epoch = state['best_metrics'], state['best_epoch']
        patience_counter = state['patience_counter']
        start_epoch = state['epoch'] + 1
        if state['best_checkpoint'] is not None:
            torch.save(state['best_checkpoint'], checkpoint_path)
            pd.DataFrame(state['best_class_records']).to_csv(
                output_dir / 'best_validation_class_metrics.csv', index=False,
            )
        write_training_message(f'epoch：{start_epoch}，//RNG。', log_path)
    elif checkpoint_path.exists() or history_path.exists():
        raise RuntimeError('last_checkpoint，epoch。')
    if not last_path.exists():
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
    stop_reason = ''
    if patience_counter >= training_config['patience']:
        start_epoch = training_config['num_epochs'] + 1
        stop_reason = ''
    for epoch in range(start_epoch, training_config['num_epochs'] + 1):
        epoch_start = perf_counter()
        learning_rate = optimizer.param_groups[0]['lr']
        write_training_message(
            f'[{label}][Epoch {epoch}/{training_config["num_epochs"]}] ', log_path,
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
        best_snapshot = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        save_epoch_checkpoint(
            last_path, model, optimizer, scheduler, epoch, history, run_config,
            {'best_metrics': best_metrics, 'best_epoch': best_epoch,
             'patience_counter': patience_counter, 'best_checkpoint': best_snapshot,
             'best_class_records': pd.read_csv(
                 output_dir / 'best_validation_class_metrics.csv'
             ).to_dict('records')},
        )
        pd.DataFrame(history).to_csv(history_path, index=False, encoding='utf-8-sig')
        write_training_message(
            f'：={losses["loss"]:.4f} | Focal={losses["classification_loss"]:.4f} | '
            f'SupCon={losses["contrastive_loss"]:.4f} | '
            f'lambda={training_config["supcon_lambda"] if use_supcon else 0}\n'
            f'：Focal={validation_loss:.4f} | OA={metrics["oa"]:.4f} | '
            f'AA={metrics["aa"]:.4f} | Kappa={metrics["kappa"]:.4f} | Macro-F1={metrics["macro_f1"]:.4f}\n'
            f'：={learning_rate:.2e} | ={next_learning_rate:.2e}\n'
            f'{selection_metric}={best_metrics[selection_metric]:.4f}（epoch {best_epoch}） | '
            f'={patience_counter}/{training_config["patience"]} | ={elapsed:.1f}\n'
            f'：{"" if improved else ""}',
            log_path,
        )
        if next_learning_rate != learning_rate:
            write_training_message('。', log_path)
        if patience_counter >= training_config['patience']:
            stop_reason = '，'
            break
    if best_metrics is None:
        raise RuntimeError('。')
    load_model_checkpoint(model, checkpoint_path, training_config['device'])
    pd.DataFrame(history).to_csv(history_path, index=False, encoding='utf-8-sig')
    write_training_message(
        f'：{stop_reason} | {len(history)} | epoch={best_epoch} | '
        f'={perf_counter() - start_time:.1f}\n'
        f'：OA={best_metrics["oa"]:.4f} | AA={best_metrics["aa"]:.4f} | '
        f'Kappa={best_metrics["kappa"]:.4f} | Macro-F1={best_metrics["macro_f1"]:.4f}\n'
        f'：{checkpoint_path}', log_path,
    )
    best_classes = pd.read_csv(output_dir / 'best_validation_class_metrics.csv')
    write_training_message(best_classes.to_string(index=False), log_path)
    return model, best_metrics, history, checkpoint_path


def stage_model(cfg, pretrained, initial_checkpoint=None):
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
    if not train.any() or not validation.any():
        raise ValueError('。')
    if (train & validation).any() or (data['role'][train | validation] == 'test').any():
        raise ValueError('。')
    cfg = deepcopy(cfg)
    output = Path(output)
    if (output / 'continued_from_best' / 'last_checkpoint.pt').exists():
        output = output / 'continued_from_best'
    old_complete = False
    if (output / 'best_checkpoint.pt').exists() and not (output / 'last_checkpoint.pt').exists():
        history_file = output / 'history.csv'
        old_history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
        finished = not old_history.empty and (
            int(old_history.iloc[-1].get('patience_counter', 0)) >= cfg['training']['patience']
            or int(old_history.iloc[-1]['epoch']) >= cfg['training']['num_epochs']
        )
        old_complete = finished
        if not finished:
            raise RuntimeError('，；--new_run。')
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
    resume_weights = output / 'last_checkpoint.pt'
    model = stage_model(
        cfg, pretrained and not resume_weights.exists(), initial_checkpoint,
    )
    if old_complete:
        checkpoint = output / 'best_checkpoint.pt'
        metadata = load_model_checkpoint(model, checkpoint, cfg['training']['device'])
        np.save(output / 'train_percentiles.npy', percentiles)
        result = evaluate_model(model, val_loader, cfg['data'], cfg['training'])
        print(f'：{output}', flush=True)
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
    model = stage_model(cfg, False, checkpoint)
    loader = create_evaluation_loader(data['x'][selected], data['y'][selected],
                                      percentiles, cfg['training'])
    result = evaluate_model(model, loader, cfg['data'], cfg['training'])
    del model, loader
    return result


def choose_run_directory(cfg):
    root = cfg['paths']['experiment_output_root'] / cfg['evaluation']['output_mode']
    run_name = f"{cfg['experiment']['name']}_seed{cfg['training']['seed']}"
    fresh_path = root / run_name
    version = 2
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
            raise ValueError('。')
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
    return fresh_path, True


def prepare_run(cfg):
    ensure_training_parameters_confirmed(cfg)
    print(f"：{cfg.get('parameter_provenance', {})}", flush=True)
    records = validate_manifest(cfg)
    output, fresh = choose_run_directory(cfg)
    if fresh:
        output.mkdir(parents=True, exist_ok=False)
        records.to_csv(output / 'split_manifest.csv', index=False)
        (output / 'run_config.json').write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding='utf-8',
        )
    elif not pd.read_csv(output / 'split_manifest.csv').equals(records):
        raise ValueError('。')
    print(f'{"" if fresh else ""}：{output}', flush=True)
    return records, output


@contextmanager
def run_directory_lock(output):
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
        raise RuntimeError('，。')
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
    seeds = repetition_seed_values if seed is None else [seed]
    if any(value not in repetition_seed_values for value in seeds):
        raise ValueError('40、41、42、43、44。')
    return list(seeds)




def run_model_experiment(experiment_name='sf_ganet', pretrained=True, manifest=None,
                         audit=None, seed=None, resume=None, new_run=False):
    seeds = repetition_seeds(seed)
    if resume and len(seeds) != 1:
        raise ValueError('--resume--seed。')
    seed_metrics, outputs = [], []
    for run_seed in seeds:
        cfg = configure_experiment(experiment_name, manifest, audit, run_seed)
        cfg['runtime'] = {'resume': resume, 'new_run': new_run}
        records, output = prepare_run(cfg)
        print(f' {run_seed}：', flush=True)
        with run_directory_lock(output):
            run_model_stages(cfg, records, output, experiment_name, pretrained)
        fold_metrics = pd.read_csv(output / 'cv_summary' / 'fold_metrics.csv')
        averages = fold_metrics[['oa', 'aa', 'kappa', 'macro_f1']].mean().to_dict()
        averages.update(seed=run_seed, fold_count=cfg['evaluation']['n_splits'])
        seed_metrics.append(averages)
        outputs.append(output)
    summary_root = cfg['paths']['experiment_output_root'] / cfg['evaluation']['output_mode']
    save_seed_summary(summary_root / ('seed_summary' if seed is None else f'seed{seed}_summary'),
                      seed_metrics, [], [])
    return outputs


def run_model_stages(cfg, records, output, experiment_name, pretrained):
    if (output / 'run_complete.json').exists():
        print(f'：{output}')
        return output
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
                    f'{fold + 1}，。'
                    '--new_run，。'
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
                raise ValueError('。')
            epochs.append(int(record['best_epoch']))
            metrics.append(record)
            classes.append(pd.read_csv(saved / 'class_metrics.csv'))
            matrices.append(np.load(saved / 'confusion_matrix.npy'))
            save_confusion_figure(saved / 'confusion_matrix.npy')
            print(f' {fold + 1}', flush=True)
            continue
        train = dev & (data['fold'] != fold)
        validation = dev & (data['fold'] == fold)
        if (stage / 'restarted_from_scratch').exists():
            stage = stage / 'restarted_from_scratch'
            print(f'{fold + 1}：{stage}；。', flush=True)
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(seed_metrics)
    if 'seed' not in table or table.seed.duplicated().any():
        raise ValueError('seed。')
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
    parser = argparse.ArgumentParser(description='40--44，')
    parser.add_argument('--manifest')
    parser.add_argument('--audit')
    parser.add_argument('--seed', type=int, choices=[40, 41, 42, 43, 44],
                        help='40--44；42')
    parser.add_argument('--check_only', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume')
    mode.add_argument('--new_run', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        for run_seed in repetition_seeds(args.seed):
            cfg = configure_experiment('sf_ganet', args.manifest, args.audit, run_seed)
            validate_manifest(cfg)
            print(f"seed={run_seed}；={cfg['evaluation']['n_splits']}")
        return
    run_model_experiment(
        manifest=args.manifest, audit=args.audit, seed=args.seed,
        resume=args.resume, new_run=args.new_run,
    )


if __name__ == '__main__':
    main()
