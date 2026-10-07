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
    if data_array.shape[0] != 7:
        raise ValueError(f"， {data_array.shape[0]} 。")
    red = data_array[2].astype(np.float32)
    nir = data_array[3].astype(np.float32)
    swir1 = data_array[4].astype(np.float32)
    epsilon = 1e-8
    ndvi = (nir - red) / (nir + red + epsilon)
    ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
    return np.stack([ndvi, ndbi]).astype(np.float32)


def load_data(image_path, label_path):
    source = rasterio.open(image_path)
    labels = gpd.read_file(label_path)
    if labels.crs != source.crs:
        labels = labels.to_crs(source.crs)
    return source, labels


def calculate_band_percentiles(image_data, percentile_range, nodata_value):
    if image_data.ndim != 3:
        raise ValueError(" bands、height、width 。")
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
    if features.ndim != 4 or features.shape[1] < 7:
        raise ValueError(" sample、channel、height、width 。")
    spectral_data = features[:, :7].transpose(1, 0, 2, 3).reshape(7, 1, -1)
    return calculate_band_percentiles(
        spectral_data,
        data_config["percentile_range"],
        data_config["nodata_value"],
    )



def extract_patches(source, labels, data_config, group_id_column=None):
    group_id_column = group_id_column or data_config["group_id_column"]
    label_column = data_config["label_column"]
    required_columns = [label_column, group_id_column]
    missing_columns = [column for column in required_columns if column not in labels.columns]
    if missing_columns:
        raise ValueError(f"：{missing_columns}")

    window_size = data_config["window_size"]
    stride = max(1, int(round(window_size * (1 - data_config["overlap"]))))
    patches, patch_labels, groups = [], [], []
    discarded_count = 0

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc=f" {window_size}×{window_size} "):
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
            raise ValueError(f" {label_column}={row[label_column]}， 1--{data_config['num_classes']}。")

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
        print(f" {discarded_count} 。")
    if not patches:
        return np.empty((0, 9, window_size, window_size), dtype=np.float32), np.array([], dtype=np.int64), np.array([])
    return np.asarray(patches, dtype=np.float32), np.asarray(patch_labels, dtype=np.int64), np.asarray(groups)


class SupervisedLCZDataset(Dataset):

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
    dataset = SupervisedLCZDataset(features, labels, percentiles, augment=False)
    return DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=True,
    )




def validate_manifest(cfg):
    manifest = Path(cfg["paths"]["split_manifest"]).resolve()
    audit = Path(cfg["paths"]["pixel_audit"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest}")
    if not audit.is_dir():
        raise FileNotFoundError(f"Pixel audit directory not found: {audit}")

    conflicts = audit / "conflicts.csv"
    if conflicts.is_file() and not pd.read_csv(conflicts).empty:
        raise ValueError("Shared pixels were detected across folds.")

    records = pd.read_csv(manifest)
    required = [
        "year", "source_row", "Parcel_ID", "SymbolID",
        "region_id", "role", "inner_fold", "patch_count",
    ]
    if not set(required).issubset(records.columns):
        raise ValueError("The split manifest is missing required columns.")
    if records[required].isna().any().any():
        raise ValueError("The split manifest contains missing values.")
    if set(records["year"]) != set(cfg["data"]["years"]):
        raise ValueError("The split manifest does not cover all configured years.")
    if (records["patch_count"] <= 0).any():
        raise ValueError("Patch counts must be positive.")
    if records.duplicated(["year", "source_row"]).any():
        raise ValueError("Duplicate yearly source records were found.")

    development = records[records["role"] == "development"]
    if development.empty:
        raise ValueError("No development records were found.")
    expected_folds = set(range(cfg["evaluation"]["n_splits"]))
    if set(development["inner_fold"]) != expected_folds:
        raise ValueError("The spatial folds do not match the configured five-fold protocol.")

    for column in ["Parcel_ID", "region_id"]:
        if development.groupby(column)["inner_fold"].nunique().max() != 1:
            raise ValueError(f"{column} crosses spatial folds.")

    audit_table = audit / "pixel_audit.csv"
    if audit_table.is_file():
        checks = pd.read_csv(audit_table)
        expected = records.groupby("year")["patch_count"].sum().sort_index()
        actual = checks.set_index("year")["patches"].sort_index()
        if not expected.equals(actual):
            raise ValueError("Patch counts do not match the pixel audit.")

    return records


def extract_region_year(records, cfg, year):
    parts, targets, roles, folds, years = [], [], [], [], []
    labels = gpd.read_file(
        cfg["paths"]["label_root"]
        / cfg["data"]["label_filename_format"].format(year=year)
    )
    image_path = (
        cfg["paths"]["image_root"]
        / cfg["data"]["image_filename_format"].format(year=year)
    )
    with rasterio.open(image_path) as source:
        if labels.crs != source.crs or source.count != 7:
            raise ValueError(f"Invalid image or label geometry for {year}.")
        combinations = (
            records[["role", "inner_fold"]]
            .drop_duplicates()
            .sort_values(["role", "inner_fold"], ascending=[False, True])
        )
        for role, fold in combinations.itertuples(index=False, name=None):
            selected = records[
                (records["year"] == year)
                & (records["role"] == role)
                & (records["inner_fold"] == fold)
            ]
            if selected.empty:
                continue
            subset = labels.iloc[selected["source_row"].to_numpy()].copy()
            if not (
                np.array_equal(subset["Parcel_ID"], selected["Parcel_ID"])
                and np.array_equal(subset["SymbolID"], selected["SymbolID"])
            ):
                raise ValueError("Label rows do not match the fixed split manifest.")
            x, y, _ = extract_patches(source, subset, cfg["data"])
            if len(y) != int(selected["patch_count"].sum()):
                raise ValueError(
                    f"Patch count changed for year={year}, role={role}, fold={fold}."
                )
            parts.append(x)
            targets.append(y)
            roles.extend([role] * len(y))
            folds.extend([fold] * len(y))
            years.extend([year] * len(y))

    if not parts:
        raise ValueError(f"No patches were extracted for {year}.")
    return {
        "x": np.concatenate(parts),
        "y": np.concatenate(targets),
        "role": np.asarray(roles),
        "fold": np.asarray(folds),
        "year": np.asarray(years),
    }


def check_cached_year(data, records, cfg, year):
    expected = records[records["year"] == year]
    count = int(expected["patch_count"].sum())
    size = cfg["data"]["window_size"]
    if data["x"].shape != (count, 9, size, size):
        raise ValueError("Cached feature shape does not match the split manifest.")
    if data["x"].dtype != np.float32:
        raise ValueError("Cached features must use float32.")
    for key in ["y", "role", "fold", "year"]:
        if data[key].shape != (count,):
            raise ValueError(f"Cached field {key} has an invalid shape.")


def load_region_features(records, cfg):
    cache_root = Path(cfg["paths"]["feature_cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    parts = {key: [] for key in ["x", "y", "role", "fold", "year"]}

    for year in cfg["data"]["years"]:
        cache_file = cache_root / f"{year}.npz"
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as saved:
                data = {key: saved[key] for key in parts}
            check_cached_year(data, records, cfg, year)
        else:
            data = extract_region_year(records, cfg, year)
            check_cached_year(data, records, cfg, year)
            temporary = cache_root / f".{year}.writing.npz"
            np.savez(temporary, **data)
            temporary.replace(cache_file)

        for key in parts:
            parts[key].append(data[key])

    return {key: np.concatenate(values) for key, values in parts.items()}
