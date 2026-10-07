import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    config as base_config,
    ensure_training_parameters_confirmed,
    get_temporal_mechanism_settings,
    get_training_strategy_settings,
)
from evaluation.metrics import save_metric_outputs, save_scores
from train.data_handler import load_region_features, validate_manifest
from train.training import (
    configure_experiment,
    fit_fixed_epochs,
    fit_stage,
    inner_validation_fold,
    reference_fold_bundle,
    repetition_seeds,
    resolve_sf_ganet_reference,
    save_seed_summary,
    score_checkpoint,
)


def strategy_config(strategy_name, seed=None):
    cfg = configure_experiment("sf_ganet", seed=seed)
    settings = get_training_strategy_settings(strategy_name)
    cfg["experiment"]["name"] = f"training_strategy_{strategy_name}"
    cfg["experiment"]["use_supcon"] = settings["use_supcon"]

    profile = deepcopy(cfg["strategy_training_profiles"][strategy_name])
    cfg["training"].update(deepcopy(profile.get("training", {})))
    cfg["paths"]["experiment_output_root"] = (
        cfg["paths"]["output_root"]
        / f"training_strategy_{strategy_name}"
    )
    return cfg, settings


def temporal_config(mechanism_name, seed=None):
    cfg = configure_experiment("sf_ganet", seed=seed)
    settings = get_temporal_mechanism_settings(mechanism_name)
    cfg["experiment"]["name"] = f"temporal_training_{mechanism_name}"
    cfg["experiment"]["use_supcon"] = settings["use_supcon"]
    cfg["paths"]["experiment_output_root"] = (
        cfg["paths"]["output_root"]
        / f"temporal_training_{mechanism_name}"
    )
    return cfg, settings


def annual_config(cfg, settings):
    annual = deepcopy(cfg)
    if settings["global_pretrain"] and settings["yearly_train"]:
        annual["training"]["learning_rate"] = cfg["training"][
            "finetune_learning_rate"
        ]
        annual["training"]["num_epochs"] = cfg["training"][
            "finetune_epochs"
        ]
    return annual


def choose_initial_checkpoint(
    settings,
    global_checkpoint,
    previous_checkpoint,
):
    if settings.get("annual_initialization") == "previous_year":
        return (
            previous_checkpoint
            if previous_checkpoint is not None
            else global_checkpoint
        )
    return global_checkpoint


def prepare_output(cfg):
    root = (
        cfg["paths"]["experiment_output_root"]
        / cfg["evaluation"]["output_mode"]
    )
    output = root / (
        f"{cfg['experiment']['name']}_seed{cfg['training']['seed']}"
    )
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_global_final_reference(reference):
    info_path = Path(reference) / "final_model" / "model_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Final pretraining model information not found: {info_path}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return (
        Path(info["checkpoint_path"]),
        Path(info["percentiles_path"]),
    )


def run_configured_strategy(cfg, settings, pretrain_run=None):
    ensure_training_parameters_confirmed(cfg)
    records = validate_manifest(cfg)
    output = prepare_output(cfg)

    if (output / "run_complete.json").exists():
        return output

    reference = None
    if settings["global_pretrain"]:
        reference = resolve_sf_ganet_reference(cfg, pretrain_run)
    elif pretrain_run is not None:
        raise ValueError(
            "This strategy does not use pooled pretraining."
        )

    data = load_region_features(records, cfg)
    development = data["role"] == "development"
    annual_cfg = annual_config(cfg, settings)

    fold_records = []
    matrices = []
    selected_epochs_by_year = {
        year: [] for year in cfg["data"]["years"]
    }

    for outer_index in range(cfg["evaluation"]["n_splits"]):
        outer_fold = outer_index + 1
        inner_index = inner_validation_fold(outer_index, cfg)
        outer_train = development & (data["fold"] != outer_index)
        outer_test = development & (data["fold"] == outer_index)
        selection_train = outer_train & (data["fold"] != inner_index)
        selection_validation = outer_train & (data["fold"] == inner_index)

        if reference is not None:
            bundle = reference_fold_bundle(reference, outer_fold)
            global_selection_checkpoint = bundle[
                "selection_checkpoint"
            ]
            global_selection_percentiles = np.load(
                bundle["selection_percentiles"],
                allow_pickle=False,
            )
            global_refit_checkpoint = bundle["refit_checkpoint"]
            global_refit_percentiles = np.load(
                bundle["refit_percentiles"],
                allow_pickle=False,
            )
        else:
            global_selection_checkpoint = None
            global_selection_percentiles = None
            global_refit_checkpoint = None
            global_refit_percentiles = None

        previous_selection_checkpoint = None
        previous_refit_checkpoint = None
        yearly_records = []
        fold_matrix = np.zeros(
            (
                cfg["data"]["num_classes"],
                cfg["data"]["num_classes"],
            ),
            dtype=np.int64,
        )

        for year in cfg["data"]["years"]:
            year_mask = data["year"] == year

            if settings["yearly_train"]:
                selection_initial = choose_initial_checkpoint(
                    settings,
                    global_selection_checkpoint,
                    previous_selection_checkpoint,
                )
                refit_initial = choose_initial_checkpoint(
                    settings,
                    global_refit_checkpoint,
                    previous_refit_checkpoint,
                )

                selection_checkpoint, _, selected_epoch, _ = fit_stage(
                    data,
                    selection_train & year_mask,
                    selection_validation & year_mask,
                    annual_cfg,
                    output
                    / f"outer_fold_{outer_fold}"
                    / f"year_{year}"
                    / "selection",
                    settings["image_net"],
                    cfg["training"]["seed"],
                    initial_checkpoint=selection_initial,
                    percentiles=global_selection_percentiles,
                )

                refit_checkpoint, refit_percentiles = fit_fixed_epochs(
                    data,
                    outer_train & year_mask,
                    annual_cfg,
                    output
                    / f"outer_fold_{outer_fold}"
                    / f"year_{year}"
                    / "refit",
                    settings["image_net"],
                    cfg["training"]["seed"],
                    selected_epoch,
                    initial_checkpoint=refit_initial,
                    percentiles=global_refit_percentiles,
                )

                previous_selection_checkpoint = selection_checkpoint
                previous_refit_checkpoint = refit_checkpoint
                selected_epochs_by_year[year].append(selected_epoch)
            else:
                refit_checkpoint = global_refit_checkpoint
                refit_percentiles = global_refit_percentiles
                selected_epoch = 0

            result = score_checkpoint(
                data,
                outer_test & year_mask,
                annual_cfg,
                refit_checkpoint,
                refit_percentiles,
            )
            record = save_scores(
                result,
                output
                / f"outer_fold_{outer_fold}"
                / f"evaluation_{year}",
                "outer_test_fold",
                {
                    "seed": cfg["training"]["seed"],
                    "fold": outer_fold,
                    "inner_fold": inner_index + 1,
                    "year": year,
                    "selected_epoch": selected_epoch,
                },
            )
            yearly_records.append(record)
            fold_matrix += result[3]

        averages = (
            pd.DataFrame(yearly_records)[
                ["oa", "aa", "kappa", "macro_f1"]
            ]
            .mean()
            .to_dict()
        )
        averages.update(
            seed=cfg["training"]["seed"],
            fold=outer_fold,
            inner_fold=inner_index + 1,
            evaluation_scope="outer_test_fold_year_equal_mean",
        )
        fold_records.append(averages)
        matrices.append(fold_matrix)

    save_metric_outputs(
        output / "cv_summary",
        fold_records,
        [],
        matrices,
    )

    build_final_models(
        data,
        development,
        cfg,
        annual_cfg,
        settings,
        reference,
        selected_epochs_by_year,
        output,
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


def build_final_models(
    data,
    development,
    cfg,
    annual_cfg,
    settings,
    reference,
    selected_epochs_by_year,
    output,
):
    final_root = output / "final_models"
    final_root.mkdir(parents=True, exist_ok=True)

    if reference is not None:
        global_checkpoint, global_percentiles_path = (
            load_global_final_reference(reference)
        )
        global_percentiles = np.load(
            global_percentiles_path, allow_pickle=False
        )
    else:
        global_checkpoint = None
        global_percentiles = None

    previous_checkpoint = None
    artifacts = []

    for year in cfg["data"]["years"]:
        year_mask = data["year"] == year

        if settings["yearly_train"]:
            observed = selected_epochs_by_year[year]
            if not observed:
                raise RuntimeError(
                    f"No selected annual epochs were recorded for {year}."
                )
            final_epochs = int(np.median(np.asarray(observed, dtype=int)))
            initial_checkpoint = choose_initial_checkpoint(
                settings,
                global_checkpoint,
                previous_checkpoint,
            )
            checkpoint, percentiles = fit_fixed_epochs(
                data,
                development & year_mask,
                annual_cfg,
                final_root / f"year_{year}",
                settings["image_net"],
                cfg["training"]["seed"],
                final_epochs,
                initial_checkpoint=initial_checkpoint,
                percentiles=global_percentiles,
            )
            previous_checkpoint = checkpoint
        else:
            final_epochs = 0
            checkpoint = global_checkpoint
            percentiles = global_percentiles
            year_dir = final_root / f"year_{year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            np.save(year_dir / "train_percentiles.npy", percentiles)
            checkpoint_copy = year_dir / "final_checkpoint.pt"
            checkpoint_copy.write_bytes(Path(checkpoint).read_bytes())
            checkpoint = checkpoint_copy

        percentiles_path = final_root / f"year_{year}" / "train_percentiles.npy"
        artifacts.append(
            {
                "year": year,
                "checkpoint_path": str(checkpoint),
                "percentiles_path": str(percentiles_path),
                "final_epochs": final_epochs,
            }
        )

    (final_root / "index.json").write_text(
        json.dumps(
            {
                "seed": cfg["training"]["seed"],
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_strategy(strategy_name, seed=None, pretrain_run=None):
    seed_metrics = []
    outputs = []

    for run_seed in repetition_seeds(seed):
        if strategy_name == "sequential_yft":
            cfg, settings = temporal_config(
                strategy_name, run_seed
            )
        else:
            cfg, settings = strategy_config(
                strategy_name, run_seed
            )

        output = run_configured_strategy(
            cfg, settings, pretrain_run=pretrain_run
        )
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
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "s1",
            "s2",
            "s3",
            "s4",
            "sequential_yft",
        ],
    )
    parser.add_argument(
        "--seed", type=int, choices=base_config["training"]["repetition_seeds"]
    )
    parser.add_argument("--pretrain_run")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        for run_seed in repetition_seeds(args.seed):
            if args.strategy == "sequential_yft":
                cfg, _ = temporal_config(
                    args.strategy, run_seed
                )
            else:
                cfg, _ = strategy_config(
                    args.strategy, run_seed
                )
            validate_manifest(cfg)
            print(
                f"{args.strategy}: seed={run_seed}, "
                f"folds={cfg['evaluation']['n_splits']}"
            )
        return

    run_strategy(
        args.strategy,
        args.seed,
        args.pretrain_run,
    )


if __name__ == "__main__":
    main()
