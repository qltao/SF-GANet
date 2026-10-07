r"""
文件作用：主模型审稿材料配置，保留 SF-GANet、S1--S4 与顺序微调的实际参数。
流程位置：训练、年度策略、地图预测及后处理共用配置；不单独运行。
主要输入：年度七波段影像、SymbolID/Parcel_ID 标注和已生成的区域五折清单。
输出：配置副本，供 train、strategies、postprocessing 调用；本模块不写结果。
重要参数：seed 默认 42、17 类、8×8 九通道斑块、五折、预训练及年度训练预算。
环境依赖：Python、PyTorch；输入路径仍指向原 F:\LCZ 项目。
调用入口：train/training.py、strategies/run_strategy_comparison.py。
"""

from copy import deepcopy
from pathlib import Path
import os

import torch


# 项目根目录与结果目录。所有新结果都与历史 code_pre_fine 存档分离。
project_root = Path(r'F:\LCZ\code')
workspace_root = project_root.parent
output_root = workspace_root / "outputs"
archive_root = workspace_root / "code_pre_fine"

# 数据路径默认跟随项目迁移，也允许通过环境变量覆盖，避免再次写死个人电脑盘符。
data_root = Path(os.environ.get("lcz_data_root", workspace_root / "data" / "data_pre"))
derived_training_data_root = workspace_root / "outputs" / "preprocessing" / "final_training_data"
training_data_root = Path(os.environ.get("lcz_training_data_root", derived_training_data_root))

# LCZ 标签统一采用模型内部的 0--16 编码，导出地图时转换为 1--17 编码。
lcz_class_names = [
    "LCZ 1",
    "LCZ 2",
    "LCZ 3",
    "LCZ 4",
    "LCZ 5",
    "LCZ 6",
    "LCZ 7",
    "LCZ 8",
    "LCZ 9",
    "LCZ 10",
    "LCZ A",
    "LCZ B",
    "LCZ C",
    "LCZ D",
    "LCZ E",
    "LCZ F",
    "LCZ G",
]


config = {
    'paths': {
        'project_root': project_root,
        'workspace_root': workspace_root,
        'archive_root': archive_root,
        'archive_training_data_root': archive_root / "final_training_data",
        'data_root': data_root,
        'training_data_root': training_data_root,
        'image_root': data_root / "pre-processing",
        'label_root': training_data_root,
        'boundary_path': data_root / "clipped" / "reprojected_boundary.shp",
        'six_ring_boundary_path': data_root
        / "clipped"
        / "六环边界数据"
        / "reprojected_liuhuan_boundary.shp",
        'raw_label_root': data_root / "clipped",
        'output_root': output_root,
        'feature_cache_root': output_root / "preprocessing" / "feature_cache",
        'fold_source_manifest': output_root / "evaluation" / "region_holdout" / "20260911_101910"
        / "candidate_split_members.csv",
        'split_manifest': output_root / "evaluation" / "region_five_fold" / "candidate_split_members.csv",
        'pixel_audit': output_root / "evaluation" / "region_five_fold" / "pixel_audit",
    },
    'data': {
        'years': list(range(2001, 2021)),
        'image_filename_format': "Beijing_{year}_Summer_30m.tif",
        'label_filename_format': "final_labels_{year}.shp",
        'raw_label_filename_format': "reprojected_vector_{year}.shp",
        'label_column': "SymbolID",
        'group_id_column': "Parcel_ID",
        'window_size': 8,
        'overlap': 0.5,
        'nodata_value': 0,
        'nodata_threshold': 0.95,
        'minimum_polygon_pixels': 16,
        'percentile_range': (0.01, 99.99),
        'num_classes': 17,
        'class_names': lcz_class_names,
    },
    'training': {
        'batch_size': 128,
        'num_epochs': 100,
        'learning_rate': 1e-5,
        'weight_decay': 5e-3,
        'patience': 10,
        'num_workers': 0,
        'seed': 42,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'supcon_lambda': 0.10,
        'supcon_temperature': 0.1,
        'finetune_epochs': 5,
        'finetune_learning_rate': 1e-6,
    },
    'experiment_training_profiles': {
        'sf_ganet': {
            "status": "verified_legacy", "source": "code_pre_fine/SF_GANet",
            "training": {"learning_rate": 1e-5, "num_epochs": 100},
        },
    },
    'strategy_training_profiles': {
        's1': {
            "status": "paper_main_reference", "source": "论文主模型设置及既定复用关系",
            "training": {"learning_rate": 1e-5, "num_epochs": 100,
                         "finetune_learning_rate": 1e-6, "finetune_epochs": 5},
        },
        's2': {
            "status": "verified_legacy", "source": "code_pre_fine/exp_liucheng/3_run_strategy2_from_scratch.py",
            "training": {"learning_rate": 1e-5, "num_epochs": 100},
        },
        's3': {
            "status": "paper_main_reference", "source": "论文主模型设置及既定复用关系",
            "training": {"learning_rate": 1e-5, "num_epochs": 100},
        },
        's4': {
            "status": "verified_legacy", "source": "code_pre_fine/exp_liucheng/5_run_strategy4_imagenet.py",
            "training": {"learning_rate": 1e-5, "num_epochs": 100},
        },
    },
    'evaluation': {
        'split_mode': "region_five_fold",
        'n_splits': 5,
        # 保留原断点配置字段以匹配已有预训练权重；此材料包不生成候选划分。
        'fold_candidate_seeds': list(range(20)),
        'spatial_block_size_m': 6000,
        'evaluation_scope': "five_fold_selected_validation",
        'manifest_sha256': "41a125bc3183bf54da128c0705c0538299353b876f8fbd46be84e3b8e1486eb2",
        'selection_metric': "macro_f1",
        'selection_tie_break': "oa",
    },
    'inference': {
        'tile_size': 4096,
        'output_resolution_m': 100,
    },
    'training_strategies': {
        's1': {
            "global_pretrain": True,
            "yearly_train": True,
            "image_net": True,
            "use_supcon": True,
        },
        's2': {
            "global_pretrain": False,
            "yearly_train": True,
            "image_net": False,
            "use_supcon": True,
        },
        's3': {
            "global_pretrain": True,
            "yearly_train": False,
            "image_net": True,
            "use_supcon": True,
        },
        's4': {
            "global_pretrain": False,
            "yearly_train": True,
            "image_net": True,
            "use_supcon": True,
        },
    },
    'temporal_training_mechanisms': {
        'sequential_yft': {
            "global_pretrain": True,
            "yearly_train": True,
            "image_net": True,
            "use_supcon": True,
            "annual_initialization": "previous_year",
            "annual_supervision": "ground_truth",
        },
    },
    'experiment_profiles': {
        'sf_ganet': {"model_name": "sf_ganet", "use_supcon": True},
    },
}


def get_experiment_config(experiment_name):
    """返回指定实验的独立配置副本，调用方修改时不会污染全局配置。"""
    if experiment_name not in config["experiment_profiles"]:
        available_names = ", ".join(config["experiment_profiles"])
        raise ValueError(f"未定义实验 {experiment_name}，可用实验为：{available_names}")

    experiment_config = deepcopy(config)
    experiment_config["experiment"] = deepcopy(config["experiment_profiles"][experiment_name])
    experiment_config["experiment"]["name"] = experiment_name
    # 来源说明不进入模型构造字段；实际生效值仍完整保存在training与run_config中。
    profile = deepcopy(config['experiment_training_profiles'].get(
        experiment_name, {'status': 'current_no_legacy', 'source': '当前新增实验，暂无旧参数对应'},
    ))
    overrides = {}
    if profile.get('inherits'):
        parent = config['experiment_training_profiles'][profile['inherits']]
        overrides.update(parent.get('training', {}))
        profile['source'] = parent.get('source', '')
    overrides.update(profile.get('training', {}))
    experiment_config['training'].update(deepcopy(overrides))
    profile['applied_training'] = deepcopy(overrides)
    experiment_config['parameter_provenance'] = profile
    experiment_config["paths"]["experiment_output_root"] = output_root / experiment_name
    return experiment_config


def ensure_training_parameters_confirmed(experiment_config):
    # 仅限制新训练/选模入口，允许只读配置和既有模型推理；不改动旧运行文件。
    provenance = experiment_config.get('parameter_provenance', {})
    if provenance.get('status') == 'pending_confirmation':
        name = experiment_config['experiment']['name']
        raise ValueError(f"{name}的历史参数待确认：{provenance.get('reason', '')} 未启动训练。")


def get_training_strategy_settings(strategy_name):
    """返回S1--S4共享定义，避免五折训练验证和推理各自维护一份。"""
    if strategy_name not in config["training_strategies"]:
        available_names = ", ".join(config["training_strategies"])
        raise ValueError(
            f"未定义训练策略 {strategy_name}，可用策略为：{available_names}"
        )
    return deepcopy(config["training_strategies"][strategy_name])


def get_temporal_mechanism_settings(mechanism_name):
    # 新对照使用独立配置入口，不让旧生产制图误按 S1 定义加载新机制。
    mechanisms = config["temporal_training_mechanisms"]
    if mechanism_name not in mechanisms:
        raise ValueError(f"未定义时间训练机制：{mechanism_name}")
    return deepcopy(mechanisms[mechanism_name])
