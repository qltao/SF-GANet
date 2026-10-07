r"""
文件作用：主模型的九通道斑块提取、训练侧归一化、数据增强和区域五折数据加载。
输入：七波段 GeoTIFF、SymbolID/Parcel_ID 标注，以及 role/year/fold 等固定清单字段。
输出：features/y/year/fold/role 数组、训练/验证 DataLoader 与训练分位数。
重要参数：8×8 窗口、NDVI/NDBI、类别编码 0--16；验证使用训练侧分位数。
环境依赖：GeoPandas、Rasterio、Shapely、PyTorch、Torchvision、NumPy、Pandas、tqdm。
调用入口：train/training.py；年度策略调用训练模块，地图预测调用指数计算函数；不单独运行。
"""

from hashlib import sha256
import inspect
import json
import os
from uuid import uuid4
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import torch
from rasterio.mask import mask
from shapely.geometry import mapping
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as transform_functional
from tqdm import tqdm


def calculate_indices_from_array(data_array):
    """按历史流程从七个原始波段计算 NDVI 与 NDBI，构成九通道输入。"""
    if data_array.shape[0] != 7:
        raise ValueError(f"期望七个原始波段，实际得到 {data_array.shape[0]} 个波段。")
    red = data_array[2].astype(np.float32)
    nir = data_array[3].astype(np.float32)
    swir1 = data_array[4].astype(np.float32)
    epsilon = 1e-8
    ndvi = (nir - red) / (nir + red + epsilon)
    ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
    return np.stack([ndvi, ndbi]).astype(np.float32)


def load_data(image_path, label_path):
    """读取单年影像和标注面，并在坐标系不一致时投影标注面。"""
    source = rasterio.open(image_path)
    labels = gpd.read_file(label_path)
    if labels.crs != source.crs:
        labels = labels.to_crs(source.crs)
    return source, labels


def calculate_band_percentiles(image_data, percentile_range, nodata_value):
    """在非 nodata 像元上计算每个原始波段的截断归一化分位数。"""
    if image_data.ndim != 3:
        raise ValueError("输入影像必须为 bands、height、width 三维数组。")
    percentiles = []
    low_percentile, high_percentile = percentile_range
    for band_index in range(image_data.shape[0]):
        valid_pixels = image_data[band_index][image_data[band_index] != nodata_value]
        if valid_pixels.size == 0:
            percentiles.append((0.0, 1.0))
        else:
            percentiles.append(tuple(np.percentile(valid_pixels, [low_percentile, high_percentile])))
    return np.asarray(percentiles, dtype=np.float32)


def calculate_percentiles_from_patches(features, data_config):
    """从训练斑块的七个原始波段计算归一化分位数，不读取验证或测试斑块。"""
    if features.ndim != 4 or features.shape[1] < 7:
        raise ValueError("训练斑块应为 sample、channel、height、width 四维数组且至少含七个原始波段。")
    spectral_data = features[:, :7].transpose(1, 0, 2, 3).reshape(7, 1, -1)
    return calculate_band_percentiles(
        spectral_data,
        data_config["percentile_range"],
        data_config["nodata_value"],
    )



def extract_patches(source, labels, data_config, group_id_column=None):
    """在已划分的地块内提取斑块，保证同一地块的重叠窗口不会跨训练与测试集合。"""
    group_id_column = group_id_column or data_config["group_id_column"]
    label_column = data_config["label_column"]
    required_columns = [label_column, group_id_column]
    missing_columns = [column for column in required_columns if column not in labels.columns]
    if missing_columns:
        raise ValueError(f"标注数据缺少必要字段：{missing_columns}")

    window_size = data_config["window_size"]
    stride = max(1, int(round(window_size * (1 - data_config["overlap"]))))
    patches, patch_labels, groups = [], [], []
    discarded_count = 0

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc=f"提取 {window_size}×{window_size} 斑块"):
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        try:
            masked_data, _ = mask(source, [mapping(geometry)], crop=True, nodata=data_config["nodata_value"])
        except Exception:
            continue
        if masked_data.shape[0] != 7:
            continue

        valid_positions = np.argwhere(masked_data[0] != data_config["nodata_value"])
        if valid_positions.size == 0:
            continue
        top_row, left_column = valid_positions.min(axis=0)
        bottom_row, right_column = valid_positions.max(axis=0)
        masked_data = masked_data[:, top_row:bottom_row + 1, left_column:right_column + 1]
        _, height, width = masked_data.shape
        if height * width < data_config["minimum_polygon_pixels"]:
            discarded_count += 1
            continue

        feature_data = np.vstack([masked_data, calculate_indices_from_array(masked_data)])
        label = int(row[label_column]) - 1
        if label < 0 or label >= data_config["num_classes"]:
            raise ValueError(f"发现超出范围的 {label_column}={row[label_column]}，应为 1--{data_config['num_classes']}。")

        if height > window_size or width > window_size:
            for row_start in range(0, height - window_size + 1, stride):
                for column_start in range(0, width - window_size + 1, stride):
                    patch = feature_data[:, row_start:row_start + window_size, column_start:column_start + window_size]
                    valid_ratio = np.mean(patch[0] != data_config["nodata_value"])
                    if valid_ratio >= data_config["nodata_threshold"]:
                        patches.append(patch)
                        patch_labels.append(label)
                        groups.append(row[group_id_column])
        else:
            height_padding = window_size - height
            width_padding = window_size - width
            padded_patch = np.pad(
                feature_data,
                (
                    (0, 0),
                    (height_padding // 2, height_padding - height_padding // 2),
                    (width_padding // 2, width_padding - width_padding // 2),
                ),
                mode="reflect",
            )
            patches.append(padded_patch)
            patch_labels.append(label)
            groups.append(row[group_id_column])

    if discarded_count:
        print(f"按最小地块阈值丢弃 {discarded_count} 个碎片多边形。")
    if not patches:
        return np.empty((0, 9, window_size, window_size), dtype=np.float32), np.array([], dtype=np.int64), np.array([])
    return np.asarray(patches, dtype=np.float32), np.asarray(patch_labels, dtype=np.int64), np.asarray(groups)


class SupervisedLCZDataset(Dataset):
    """在取样时执行训练集分位数归一化和可复现的空间增强。"""

    def __init__(self, patches, labels, percentiles, augment):
        self.patches = patches
        self.labels = labels
        self.percentiles = percentiles
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = self.patches[index].copy()
        normalized_spectral = self.normalize_spectral(patch[:7], self.percentiles)
        normalized_indices = np.clip((patch[7:] + 1) / 2, 0, 1)
        patch_tensor = torch.from_numpy(np.vstack([normalized_spectral, normalized_indices]).astype(np.float32))
        if self.augment:
            patch_tensor = self.apply_augmentations(patch_tensor)
        return patch_tensor, torch.tensor(self.labels[index], dtype=torch.long)

    @staticmethod
    def normalize_spectral(spectral_data, percentiles):
        normalized_data = np.zeros_like(spectral_data, dtype=np.float32)
        for band_index, (lower_bound, upper_bound) in enumerate(percentiles):
            if upper_bound > lower_bound:
                normalized_data[band_index] = np.clip(
                    (spectral_data[band_index] - lower_bound) / (upper_bound - lower_bound),
                    0,
                    1,
                )
        return normalized_data

    @staticmethod
    def apply_augmentations(patch_tensor):
        if torch.rand(1).item() > 0.5:
            patch_tensor = transform_functional.hflip(patch_tensor)
        if torch.rand(1).item() > 0.5:
            patch_tensor = transform_functional.vflip(patch_tensor)
        rotation_count = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(patch_tensor, rotation_count, [1, 2])


def prepare_dataloaders(
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    percentiles,
    data_config,
    training_config,
):
    """建立类别重加权训练集和不增强验证集的数据加载器。"""
    train_dataset = SupervisedLCZDataset(train_features, train_labels, percentiles, augment=True)
    validation_dataset = SupervisedLCZDataset(validation_features, validation_labels, percentiles, augment=False)
    class_counts = np.bincount(train_labels, minlength=data_config["num_classes"])
    class_weights = 1.0 / (torch.tensor(class_counts, dtype=torch.float32) + 1e-6)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        sampler=sampler,
        num_workers=training_config["num_workers"],
        pin_memory=True,
        drop_last=len(train_dataset) >= training_config["batch_size"],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=True,
    )
    return train_loader, validation_loader


def create_evaluation_loader(features, labels, percentiles, training_config):
    """为独立测试折或最终复核数据创建不增强、不重采样的 DataLoader。"""
    dataset = SupervisedLCZDataset(features, labels, percentiles, augment=False)
    return DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=True,
    )



def validate_manifest(cfg):
    # 哈希绑定已核查版本；角色、年份、同源区域和内部折号必须全部有效。
    manifest = Path(cfg['paths']['split_manifest']).resolve()
    audit = Path(cfg['paths']['pixel_audit']).resolve()
    digest = sha256(manifest.read_bytes()).hexdigest()
    if digest != cfg['evaluation']['manifest_sha256']:
        raise ValueError('清单哈希与已审核版本不一致，必须重新进行像元核查。')
    if audit.parent != manifest.parent:
        raise ValueError('像元审计与清单目录不对应。')
    if not pd.read_csv(audit / 'conflicts.csv').empty:
        raise ValueError('存在跨集合共享像元，禁止训练。')
    if cfg['evaluation']['split_mode'] == 'region_five_fold':
        metadata = json.loads((audit / 'audit_metadata.json').read_text(encoding='utf-8'))
        if not metadata.get('completed') or metadata['manifest_sha256'] != digest:
            raise ValueError('共享像元审计不属于当前五折清单。')
    records = pd.read_csv(manifest)
    required = ['year', 'source_row', 'Parcel_ID', 'SymbolID', 'region_id',
                'role', 'inner_fold', 'patch_count']
    if not set(required).issubset(records.columns) or records[required].isna().any().any():
        raise ValueError('清单字段缺失或存在空值。')
    expected_roles = ({'development'} if cfg['evaluation']['split_mode'] == 'region_five_fold'
                      else {'test', 'development'})
    if set(records.role) != expected_roles:
        raise ValueError(f'当前协议要求角色为{expected_roles}。')
    if set(records.year) != set(cfg['data']['years']):
        raise ValueError('配置年份与固定清单不一致。')
    checks = pd.read_csv(audit / 'pixel_audit.csv')
    expected = records.groupby('year').patch_count.sum().sort_index()
    actual = checks.set_index('year').patches.sort_index()
    if not expected.equals(actual):
        raise ValueError('年度斑块数与像元核查不一致。')
    dev = records[records.role == 'development']
    if set(dev.inner_fold) != set(range(cfg['evaluation']['n_splits'])):
        raise ValueError('开发集内部折号不完整。')
    if (records.loc[records.role == 'test', 'inner_fold'] != -1).any():
        raise ValueError('测试记录不能拥有内部折号。')
    if (records.patch_count <= 0).any() or records.duplicated(['year', 'source_row']).any():
        raise ValueError('记录重复或斑块计数无效。')
    for column in ['Parcel_ID', 'region_id']:
        if records.groupby(column).role.nunique().max() != 1:
            raise ValueError(f'{column}跨开发/测试集合。')
        if dev.groupby(column).inner_fold.nunique().max() != 1:
            raise ValueError(f'{column}跨内部折。')
    return records

def extract_region_year(records, cfg, year):
    # 真实训练提取函数统一调用；在分组之后提取，核对逐组计数。
    parts, targets, roles, folds, years = [], [], [], [], []
    labels = gpd.read_file(cfg['paths']['label_root'] / f'final_labels_{year}.shp')
    image_path = cfg['paths']['image_root'] / f'Beijing_{year}_Summer_10m.tif'
    with rasterio.open(image_path) as source:
        if labels.crs != source.crs or source.count != 7:
            raise ValueError(f'{year}影像波段或标注坐标系不符合已审核输入。')
        for role, fold in records[['role', 'inner_fold']].drop_duplicates().sort_values(
            ['role', 'inner_fold'], ascending=[False, True],
        ).itertuples(index=False, name=None):
            selected = records[(records.year == year) & (records.role == role)
                               & (records.inner_fold == fold)]
            if selected.empty:
                raise ValueError(f'{year}/{role}/{fold}没有标注。')
            subset = labels.iloc[selected.source_row.to_numpy()].copy()
            if not (np.array_equal(subset.Parcel_ID, selected.Parcel_ID)
                    and np.array_equal(subset.SymbolID, selected.SymbolID)):
                raise ValueError('标注行内容与固定清单不一致。')
            x, y, _ = extract_patches(source, subset, cfg['data'])
            if len(y) != selected.patch_count.sum():
                raise ValueError(f'{year}/{role}/{fold}实际斑块数变化，需重新审计。')
            parts.append(x)
            targets.append(y)
            roles.extend([role] * len(y))
            folds.extend([fold] * len(y))
            years.extend([year] * len(y))
    print(f'FEATURES_READY_YEAR={year}', flush=True)
    return {'x': np.concatenate(parts), 'y': np.concatenate(targets),
            'role': np.asarray(roles), 'fold': np.asarray(folds), 'year': np.asarray(years)}

def region_cache_key(records, cfg, year):
    # 大影像使用绝对路径、字节数、纳秒修改时间；小标注文件同时校验内容哈希。
    image = cfg['paths']['image_root'] / f'Beijing_{year}_Summer_10m.tif'
    label = cfg['paths']['label_root'] / f'final_labels_{year}.shp'
    files = [image]
    files.extend(Path(str(image) + suffix) for suffix in ['.aux.xml', '.msk']
                 if Path(str(image) + suffix).exists())
    files.extend(label.with_suffix(suffix) for suffix in ['.shp', '.shx', '.dbf', '.prj', '.cpg']
                 if label.with_suffix(suffix).exists())
    sources = []
    for path in files:
        stat = path.stat()
        item = {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if path != image:
            item['sha256'] = sha256(path.read_bytes()).hexdigest()
        sources.append(item)
    keys = ['window_size', 'overlap', 'nodata_value', 'nodata_threshold',
            'minimum_polygon_pixels', 'num_classes', 'label_column', 'group_id_column']
    source_code = ''.join(inspect.getsource(function) for function in
                          [extract_patches, calculate_indices_from_array, extract_region_year])
    specification = {
        'version': 1, 'year': year, 'sources': sources,
        'parameters': {key: cfg['data'][key] for key in keys},
        'records_sha256': sha256(records[records.year == year].to_csv(index=False).encode()).hexdigest(),
        'extractor_sha256': sha256(source_code.encode()).hexdigest(),
        'numpy_version': np.__version__, 'rasterio_version': rasterio.__version__,
    }
    digest = sha256(json.dumps(specification, sort_keys=True, default=str).encode()).hexdigest()
    return digest


def check_cached_year(data, records, cfg, year):
    # 检查形状、标签和行顺序；不把文件存在等同于缓存有效。
    expected = records[records.year == year]
    count = int(expected.patch_count.sum())
    size = cfg['data']['window_size']
    if data['x'].shape != (count, 9, size, size) or data['x'].dtype != np.float32:
        raise ValueError('缓存斑块形状或类型不匹配。')
    if any(data[key].shape != (count,) for key in ['y', 'role', 'fold', 'year']):
        raise ValueError('缓存标签或分组长度不匹配。')
    position = 0
    for role, fold in [('test', -1)] + [('development', i) for i in range(5)]:
        selected = expected[(expected.role == role) & (expected.inner_fold == fold)]
        for row in selected.itertuples():
            end = position + int(row.patch_count)
            if not (np.all(data['y'][position:end] == int(row.SymbolID) - 1)
                    and np.all(data['role'][position:end] == role)
                    and np.all(data['fold'][position:end] == fold)
                    and np.all(data['year'][position:end] == year)):
                raise ValueError('缓存内斑块顺序或角色与固定清单不一致。')
            position = end
    if position != count:
        raise ValueError('缓存计数不一致。')


def load_region_features(records, cfg):
    # 缓存位于项目共享目录，不在单次实验目录；不缓存归一化参数或随机增强结果。
    cache_root = Path(cfg['paths'].get(
        'feature_cache_root', cfg['paths']['output_root'] / 'preprocessing' / 'feature_cache',
    ))
    cache_root.mkdir(parents=True, exist_ok=True)
    parts = {key: [] for key in ['x', 'y', 'role', 'fold', 'year']}
    for year in cfg['data']['years']:
        digest = region_cache_key(records, cfg, year)
        cache_file = cache_root / f'{year}_{digest}.npz'
        if cache_file.exists():
            try:
                with np.load(cache_file, allow_pickle=False) as saved:
                    if str(saved['cache_key'].item()) != digest:
                        raise ValueError('缓存指纹不一致。')
                    data = {key: saved[key] for key in parts}
                check_cached_year(data, records, cfg, year)
            except Exception as error:
                raise RuntimeError(
                    f'缓存损坏或不一致，已保留原文件，请先核查：{cache_file}'
                ) from error
            print(f'[缓存命中] {year}：载入{len(data["y"])}个斑块，不重新提取。', flush=True)
        else:
            print(f'[缓存缺失或输入变化] {year}：提取并保存原始斑块。', flush=True)
            data = extract_region_year(records, cfg, year)
            check_cached_year(data, records, cfg, year)
            if region_cache_key(records, cfg, year) != digest:
                raise RuntimeError('提取期间输入文件发生变化，未发布缓存。')
            # 写完并关闭临时npz后才发布；中断的临时文件不会被当作有效缓存。
            temporary = cache_root / f'.{year}_{uuid4().hex}.writing.npz'
            np.savez(temporary, cache_key=np.asarray(digest), **data)
            os.replace(temporary, cache_file)
            print(f'[缓存已保存] {cache_file}', flush=True)
        for key in parts:
            parts[key].append(data[key])
    return {key: np.concatenate(values) for key, values in parts.items()}
