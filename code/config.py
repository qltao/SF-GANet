from copy import deepcopy
from pathlib import Path
import os

import torch


code_root = Path(__file__).resolve().parent
repo_root = code_root.parent
data_root = Path(os.environ.get("LCZ_DATA_ROOT", repo_root / "data"))
output_root = Path(os.environ.get("LCZ_OUTPUT_ROOT", repo_root / "outputs"))
training_data_root = Path(
    os.environ.get("LCZ_TRAINING_DATA_ROOT", data_root / "labels")
)

lcz_class_names = [
    "LCZ 1", "LCZ 2", "LCZ 3", "LCZ 4", "LCZ 5", "LCZ 6",
    "LCZ 7", "LCZ 8", "LCZ 9", "LCZ 10", "LCZ A", "LCZ B",
    "LCZ C", "LCZ D", "LCZ E", "LCZ F", "LCZ G",
]


config = {
    "paths": {
        "code_root": code_root,
        "repo_root": repo_root,
        "data_root": data_root,
        "training_data_root": training_data_root,
        "image_root": Path(
            os.environ.get("LCZ_IMAGE_ROOT", data_root / "images")
        ),
        "label_root": training_data_root,
        "boundary_path": Path(
            os.environ.get(
                "LCZ_BOUNDARY_PATH",
                data_root / "boundaries" / "study_area.shp",
            )
        ),
        "six_ring_boundary_path": Path(
            os.environ.get(
                "LCZ_SIX_RING_BOUNDARY",
                data_root / "boundaries" / "six_ring_boundary.shp",
            )
        ),
        "output_root": output_root,
        "feature_cache_root": output_root / "preprocessing" / "feature_cache",
        "split_manifest": Path(
            os.environ.get(
                "LCZ_SPLIT_MANIFEST",
                data_root / "splits" / "candidate_split_members.csv",
            )
        ),
        "pixel_audit": Path(
            os.environ.get(
                "LCZ_PIXEL_AUDIT",
                data_root / "splits" / "pixel_audit",
            )
        ),
    },
    "data": {
        "years": list(range(2001, 2021)),
        "image_filename_format": "Beijing_{year}_Summer_30m.tif",
        "label_filename_format": "final_labels_{year}.shp",
        "label_column": "SymbolID",
        "group_id_column": "Parcel_ID",
        "window_size": 8,
        "overlap": 0.5,
        "nodata_value": 0,
        "nodata_threshold": 0.95,
        "minimum_polygon_pixels": 16,
        "percentile_range": (0.01, 99.99),
        "num_classes": 17,
        "class_names": lcz_class_names,
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 100,
        "learning_rate": 1e-5,
        "weight_decay": 5e-3,
        "patience": 10,
        "num_workers": 0,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "supcon_lambda": 0.10,
        "supcon_temperature": 0.1,
        "finetune_epochs": 5,
        "finetune_learning_rate": 1e-6,
    },
    "strategy_training_profiles": {
        "s1": {
            "training": {
                "learning_rate": 1e-5,
                "num_epochs": 100,
                "finetune_learning_rate": 1e-6,
                "finetune_epochs": 5,
            }
        },
        "s2": {
            "training": {"learning_rate": 1e-5, "num_epochs": 100}
        },
        "s3": {
            "training": {"learning_rate": 1e-5, "num_epochs": 100}
        },
        "s4": {
            "training": {"learning_rate": 1e-5, "num_epochs": 100}
        },
    },
    "evaluation": {
        "split_mode": "region_five_fold",
        "n_splits": 5,
        "inner_validation_offset": 1,
        "evaluation_scope": "nested_spatial_five_fold",
        "selection_metric": "macro_f1",
        "selection_tie_break": "oa",
        "final_epoch_rule": "median",
    },
    "inference": {
        "tile_size": 4096,
        "output_resolution_m": 100,
        "map_seed": 42,
    },
    "training_strategies": {
        "s1": {
            "global_pretrain": True,
            "yearly_train": True,
            "image_net": True,
            "use_supcon": True,
        },
        "s2": {
            "global_pretrain": False,
            "yearly_train": True,
            "image_net": False,
            "use_supcon": True,
        },
        "s3": {
            "global_pretrain": True,
            "yearly_train": False,
            "image_net": True,
            "use_supcon": True,
        },
        "s4": {
            "global_pretrain": False,
            "yearly_train": True,
            "image_net": True,
            "use_supcon": True,
        },
    },
    "temporal_training_mechanisms": {
        "sequential_yft": {
            "global_pretrain": True,
            "yearly_train": True,
            "image_net": True,
            "use_supcon": True,
            "annual_initialization": "previous_year",
            "annual_supervision": "ground_truth",
        },
    },
    "experiment_profiles": {
        "sf_ganet": {"model_name": "sf_ganet", "use_supcon": True},
    },
}


def get_experiment_config(experiment_name):
    if experiment_name not in config["experiment_profiles"]:
        available = ", ".join(config["experiment_profiles"])
        raise ValueError(
            f"Unknown experiment: {experiment_name}. Available: {available}"
        )
    experiment_config = deepcopy(config)
    experiment_config["experiment"] = deepcopy(
        config["experiment_profiles"][experiment_name]
    )
    experiment_config["experiment"]["name"] = experiment_name
    experiment_config["paths"]["experiment_output_root"] = (
        output_root / experiment_name
    )
    return experiment_config


def ensure_training_parameters_confirmed(experiment_config):
    return None


def get_training_strategy_settings(strategy_name):
    if strategy_name not in config["training_strategies"]:
        available = ", ".join(config["training_strategies"])
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {available}"
        )
    return deepcopy(config["training_strategies"][strategy_name])


def get_temporal_mechanism_settings(mechanism_name):
    mechanisms = config["temporal_training_mechanisms"]
    if mechanism_name not in mechanisms:
        available = ", ".join(mechanisms)
        raise ValueError(
            f"Unknown temporal mechanism: {mechanism_name}. Available: {available}"
        )
    return deepcopy(mechanisms[mechanism_name])
