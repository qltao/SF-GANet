import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import (
    config as base_config,
    ensure_training_parameters_confirmed,
    get_experiment_config,
)
from evaluation.metrics import (
    calculate_metrics,
    save_metric_outputs,
    save_scores,
)
from train.data_handler import (
    calculate_percentiles_from_patches,
    create_evaluation_loader,
    load_region_features,
    prepare_dataloaders,
    validate_manifest,
)
from train.losses import SupConLoss, focal_loss


repetition_seed_values = tuple(base_config["training"]["repetition_seeds"])
repetition_output_mode = "nested_spatial_five_fold"


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


def stage_model(cfg, pretrained, initial_checkpoint=None):
    from model.model import SF_GANet

    model = SF_GANet(
        cfg["data"]["num_classes"],
        9,
        pretrained if initial_checkpoint is None else False,
        cfg["experiment"]["use_supcon"],
    ).to(cfg["training"]["device"])
    if initial_checkpoint is not None:
        load_model_checkpoint(
            model, initial_checkpoint, cfg["training"]["device"]
        )
    return model


def save_model_checkpoint(model, path, epoch, run_config, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": deepcopy(model.state_dict()),
            "epoch": int(epoch),
            "metrics": deepcopy(metrics),
            "run_config": deepcopy(run_config),
        },
        path,
    )


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_one_epoch(model, train_loader, optimizer, training_config, use_supcon):
    model.train()
    contrastive_loss = SupConLoss(
        temperature=training_config["supcon_temperature"],
        base_temperature=training_config["supcon_temperature"],
    )
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "contrastive_loss": 0.0,
    }

    for inputs, labels in tqdm(train_loader, leave=False):
        inputs = inputs.to(training_config["device"])
        labels = labels.to(training_config["device"])
        optimizer.zero_grad()

        features, logits = split_model_output(model(inputs))
        classification_loss = focal_loss(logits, labels)
        supcon_value = torch.tensor(0.0, device=logits.device)
        if use_supcon:
            if features is None:
                raise ValueError(
                    "SupCon is enabled but the model did not return features."
                )
            supcon_value = contrastive_loss(features, labels)

        loss = (
            classification_loss
            + training_config["supcon_lambda"] * supcon_value
        )
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["classification_loss"] += classification_loss.item()
        totals["contrastive_loss"] += supcon_value.item()

    count = max(len(train_loader), 1)
    return {key: value / count for key, value in totals.items()}


def evaluate_model(model, data_loader, data_config, training_config):
    model.eval()
    total_loss = 0.0
    predictions, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(training_config["device"])
            labels = labels.to(training_config["device"])
            _, logits = split_model_output(model(inputs))
            total_loss += focal_loss(logits, labels).item()
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    metrics, class_metrics, matrix = calculate_metrics(
        predictions, labels_list, data_config["num_classes"]
    )
    return (
        total_loss / max(len(data_loader), 1),
        metrics,
        class_metrics,
        matrix,
        np.asarray(predictions),
        np.asarray(labels_list),
    )


def metric_is_better(current, best, metric_name):
    if best is None:
        return True
    if current[metric_name] != best[metric_name]:
        return current[metric_name] > best[metric_name]
    return current["oa"] > best["oa"]


def fit_stage(
    data,
    train,
    validation,
    cfg,
    output,
    pretrained,
    seed,
    initial_checkpoint=None,
    percentiles=None,
):
    if not train.any() or not validation.any():
        raise ValueError("Training and validation subsets must be non-empty.")
    if (train & validation).any():
        raise ValueError("Training and validation subsets overlap.")

    cfg = deepcopy(cfg)
    cfg["training"]["seed"] = seed
    set_random_seeds(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if percentiles is None:
        percentiles = calculate_percentiles_from_patches(
            data["x"][train], cfg["data"]
        )

    train_loader, validation_loader = prepare_dataloaders(
        data["x"][train],
        data["y"][train],
        data["x"][validation],
        data["y"][validation],
        percentiles,
        cfg["data"],
        cfg["training"],
    )

    model = stage_model(cfg, pretrained, initial_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=max(1, cfg["training"]["patience"] // 2),
        factor=0.2,
    )

    selection_metric = cfg["evaluation"]["selection_metric"]
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    checkpoint = output / "best_checkpoint.pt"
    history = []

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            cfg["training"],
            cfg["experiment"]["use_supcon"],
        )
        validation_loss, metrics, _, _, _, _ = evaluate_model(
            model, validation_loader, cfg["data"], cfg["training"]
        )
        scheduler.step(validation_loss)

        improved = metric_is_better(
            metrics, best_metrics, selection_metric
        )
        if improved:
            best_metrics = dict(metrics)
            best_epoch = epoch
            save_model_checkpoint(
                model, checkpoint, epoch, cfg, metrics
            )
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                **losses,
                "validation_loss": validation_loss,
                **metrics,
            }
        )
        if patience_counter >= cfg["training"]["patience"]:
            break

    if best_epoch < 1:
        raise RuntimeError("No checkpoint was selected.")

    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    np.save(output / "train_percentiles.npy", percentiles)
    load_model_checkpoint(model, checkpoint, cfg["training"]["device"])
    result = evaluate_model(
        model, validation_loader, cfg["data"], cfg["training"]
    )
    return checkpoint, percentiles, best_epoch, result


def fit_fixed_epochs(
    data,
    train,
    cfg,
    output,
    pretrained,
    seed,
    epochs,
    initial_checkpoint=None,
    percentiles=None,
):
    if not train.any():
        raise ValueError("The refit training subset is empty.")
    if epochs < 1:
        raise ValueError("The fixed refit epoch count must be positive.")

    cfg = deepcopy(cfg)
    cfg["training"]["seed"] = seed
    set_random_seeds(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if percentiles is None:
        percentiles = calculate_percentiles_from_patches(
            data["x"][train], cfg["data"]
        )

    train_loader, _ = prepare_dataloaders(
        data["x"][train],
        data["y"][train],
        data["x"][train],
        data["y"][train],
        percentiles,
        cfg["data"],
        cfg["training"],
    )

    model = stage_model(cfg, pretrained, initial_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    history = []
    for epoch in range(1, int(epochs) + 1):
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            cfg["training"],
            cfg["experiment"]["use_supcon"],
        )
        history.append({"epoch": epoch, **losses})

    checkpoint = output / "final_checkpoint.pt"
    save_model_checkpoint(model, checkpoint, epochs, cfg)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    np.save(output / "train_percentiles.npy", percentiles)
    return checkpoint, percentiles


def score_checkpoint(data, selected, cfg, checkpoint, percentiles):
    model = stage_model(cfg, False, checkpoint)
    loader = create_evaluation_loader(
        data["x"][selected],
        data["y"][selected],
        percentiles,
        cfg["training"],
    )
    result = evaluate_model(
        model, loader, cfg["data"], cfg["training"]
    )
    return result


def inner_validation_fold(outer_fold, cfg):
    n_splits = cfg["evaluation"]["n_splits"]
    offset = cfg["evaluation"]["inner_validation_offset"]
    candidate = (outer_fold + offset) % n_splits
    if candidate == outer_fold:
        candidate = (candidate + 1) % n_splits
    return candidate


def configure_experiment(experiment_name, manifest=None, audit=None, seed=None):
    cfg = get_experiment_config(experiment_name)
    cfg["evaluation"]["output_mode"] = repetition_output_mode
    if manifest is not None:
        cfg["paths"]["split_manifest"] = Path(manifest)
    if audit is not None:
        cfg["paths"]["pixel_audit"] = Path(audit)
    if seed is not None:
        cfg["training"]["seed"] = seed
    return cfg


def repetition_seeds(seed=None):
    values = repetition_seed_values if seed is None else [seed]
    if any(value not in repetition_seed_values for value in values):
        raise ValueError("Seed must be one of 40, 41, 42, 43, or 44.")
    return list(values)


def prepare_run(cfg):
    ensure_training_parameters_confirmed(cfg)
    records = validate_manifest(cfg)
    root = (
        cfg["paths"]["experiment_output_root"]
        / cfg["evaluation"]["output_mode"]
    )
    output = root / (
        f"{cfg['experiment']['name']}_seed{cfg['training']['seed']}"
    )
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "split_manifest.csv", index=False)
    (output / "run_config.json").write_text(
        json.dumps(cfg, indent=2, default=str),
        encoding="utf-8",
    )
    return records, output


def resume_signature(run_config):
    selected = {
        "data": (run_config or {}).get("data"),
        "evaluation": (run_config or {}).get("evaluation"),
        "experiment": (run_config or {}).get("experiment"),
        "training": (run_config or {}).get("training"),
    }
    return json.dumps(selected, sort_keys=True, default=str)


def reference_fold_bundle(source, outer_fold):
    source = Path(source)
    fold_dir = source / f"outer_fold_{outer_fold}"
    metrics = pd.read_csv(
        source / f"evaluation_fold_{outer_fold}" / "metrics.csv"
    ).iloc[0]

    selection_checkpoint = (
        fold_dir / "selection" / "best_checkpoint.pt"
    )
    selection_percentiles = (
        fold_dir / "selection" / "train_percentiles.npy"
    )
    refit_checkpoint = fold_dir / "refit" / "final_checkpoint.pt"
    refit_percentiles = fold_dir / "refit" / "train_percentiles.npy"

    required = [
        selection_checkpoint,
        selection_percentiles,
        refit_checkpoint,
        refit_percentiles,
    ]
    if any(not path.exists() for path in required):
        raise FileNotFoundError(
            f"Incomplete reference artifacts for outer fold {outer_fold}."
        )

    return {
        "selection_checkpoint": selection_checkpoint,
        "selection_percentiles": selection_percentiles,
        "refit_checkpoint": refit_checkpoint,
        "refit_percentiles": refit_percentiles,
        "selected_epoch": int(metrics["selected_epoch"]),
        "inner_fold": int(metrics["inner_fold"]),
    }


def reference_fold_files(source, fold):
    bundle = reference_fold_bundle(source, fold)
    return (
        bundle["refit_checkpoint"],
        bundle["refit_percentiles"],
        bundle["selected_epoch"],
    )


def resolve_sf_ganet_reference(cfg, source_run=None):
    if source_run is not None:
        source = Path(source_run)
    else:
        source = (
            Path(cfg["paths"]["output_root"])
            / "sf_ganet"
            / cfg["evaluation"]["output_mode"]
            / f"sf_ganet_seed{cfg['training']['seed']}"
        )

    if not (source / "run_complete.json").is_file():
        raise FileNotFoundError(
            f"Reference pretraining run not found: {source}"
        )
    for fold in range(1, cfg["evaluation"]["n_splits"] + 1):
        reference_fold_bundle(source, fold)
    return source


def save_seed_summary(output_dir, seed_metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(seed_metrics)
    metric_names = ["oa", "aa", "kappa", "macro_f1"]
    summary = pd.DataFrame(
        [
            {
                "metric": name,
                "mean": table[name].mean(),
                "std": table[name].std(ddof=1),
            }
            for name in metric_names
        ]
    )
    table.to_csv(output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)


def run_model_stages(cfg, records, output, pretrained):
    if (output / "run_complete.json").exists():
        return output

    data = load_region_features(records, cfg)
    development = data["role"] == "development"
    fold_records = []
    class_records = []
    matrices = []
    selected_epochs = []

    for outer_index in range(cfg["evaluation"]["n_splits"]):
        outer_fold = outer_index + 1
        inner_index = inner_validation_fold(outer_index, cfg)
        outer_train = development & (data["fold"] != outer_index)
        outer_test = development & (data["fold"] == outer_index)
        selection_train = outer_train & (data["fold"] != inner_index)
        selection_validation = outer_train & (data["fold"] == inner_index)

        fold_dir = output / f"outer_fold_{outer_fold}"
        selection_checkpoint, _, selected_epoch, _ = fit_stage(
            data,
            selection_train,
            selection_validation,
            cfg,
            fold_dir / "selection",
            pretrained,
            cfg["training"]["seed"],
        )

        refit_checkpoint, refit_percentiles = fit_fixed_epochs(
            data,
            outer_train,
            cfg,
            fold_dir / "refit",
            pretrained,
            cfg["training"]["seed"],
            selected_epoch,
        )

        result = score_checkpoint(
            data,
            outer_test,
            cfg,
            refit_checkpoint,
            refit_percentiles,
        )
        record = save_scores(
            result,
            output / f"evaluation_fold_{outer_fold}",
            "outer_test_fold",
            {
                "seed": cfg["training"]["seed"],
                "fold": outer_fold,
                "inner_fold": inner_index + 1,
                "selected_epoch": selected_epoch,
                "checkpoint_path": str(refit_checkpoint),
            },
        )
        fold_records.append(record)
        class_records.append(result[2])
        matrices.append(result[3])
        selected_epochs.append(selected_epoch)

    save_metric_outputs(
        output / "cv_summary",
        fold_records,
        class_records,
        matrices,
    )
    np.save(
        output / "selected_epochs.npy",
        np.asarray(selected_epochs, dtype=int),
    )

    final_epochs = int(
        np.median(np.asarray(selected_epochs, dtype=int))
    )
    final_checkpoint, final_percentiles = fit_fixed_epochs(
        data,
        development,
        cfg,
        output / "final_model",
        pretrained,
        cfg["training"]["seed"],
        final_epochs,
    )
    (output / "final_model" / "model_info.json").write_text(
        json.dumps(
            {
                "epoch_rule": cfg["evaluation"]["final_epoch_rule"],
                "selected_epochs": selected_epochs,
                "final_epochs": final_epochs,
                "checkpoint_path": str(final_checkpoint),
                "percentiles_path": str(final_percentiles),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (output / "run_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "seed": cfg["training"]["seed"],
                "protocol": "nested_spatial_five_fold",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def run_model_experiment(
    experiment_name="sf_ganet",
    pretrained=True,
    manifest=None,
    audit=None,
    seed=None,
):
    outputs = []
    seed_metrics = []

    for run_seed in repetition_seeds(seed):
        cfg = configure_experiment(
            experiment_name, manifest, audit, run_seed
        )
        records, output = prepare_run(cfg)
        run_model_stages(cfg, records, output, pretrained)

        fold_metrics = pd.read_csv(
            output / "cv_summary" / "fold_metrics.csv"
        )
        averages = (
            fold_metrics[["oa", "aa", "kappa", "macro_f1"]]
            .mean()
            .to_dict()
        )
        averages.update(
            seed=run_seed,
            fold_count=cfg["evaluation"]["n_splits"],
        )
        seed_metrics.append(averages)
        outputs.append(output)

    summary_root = (
        cfg["paths"]["experiment_output_root"]
        / cfg["evaluation"]["output_mode"]
    )
    save_seed_summary(
        summary_root
        / ("seed_summary" if seed is None else f"seed{seed}_summary"),
        seed_metrics,
    )
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--audit")
    parser.add_argument(
        "--seed", type=int, choices=repetition_seed_values
    )
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        for run_seed in repetition_seeds(args.seed):
            cfg = configure_experiment(
                "sf_ganet", args.manifest, args.audit, run_seed
            )
            validate_manifest(cfg)
            print(
                f"seed={run_seed}, folds={cfg['evaluation']['n_splits']}"
            )
        return

    run_model_experiment(
        manifest=args.manifest,
        audit=args.audit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
