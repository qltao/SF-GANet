
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter

from config import get_experiment_config


class_radii = {
    1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 5, 9: 3, 10: 5,
    11: 2, 12: 2, 13: 2, 14: 2, 15: 3, 16: 2, 17: 1,
}
class_sigmas = {
    1: 1, 2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5, 7: 1.5, 8: 2.5, 9: 1.5, 10: 2.5,
    11: 1, 12: 1, 13: 1, 14: 1, 15: 1.5, 16: 1, 17: 0.5,
}
temporal_half_window = 2


def clip_and_resample(input_path, output_path, boundary_path, output_resolution_m):
    with rasterio.open(input_path) as source:
        boundary = gpd.read_file(boundary_path)
        if boundary.crs != source.crs:
            boundary = boundary.to_crs(source.crs)
        geometries = [feature["geometry"] for _, feature in boundary.iterrows()]
        clipped_data, clipped_transform = mask(source, geometries, crop=True, nodata=0)
        clipped_profile = source.profile.copy()
        clipped_profile.update(
            height=clipped_data.shape[1],
            width=clipped_data.shape[2],
            transform=clipped_transform,
            nodata=0,
        )
        clipped_bounds = array_bounds(clipped_data.shape[1], clipped_data.shape[2], clipped_transform)
        transform, width, height = calculate_default_transform(
            source.crs,
            source.crs,
            clipped_data.shape[2],
            clipped_data.shape[1],
            *clipped_bounds,
            resolution=(output_resolution_m, output_resolution_m),
        )
        output_profile = clipped_profile.copy()
        output_profile.update(transform=transform, width=width, height=height, dtype=rasterio.uint8, nodata=0)
        output_array = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=clipped_data[0],
            destination=output_array,
            src_transform=clipped_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=source.crs,
            resampling=Resampling.nearest,
        )
    with rasterio.open(output_path, "w", **output_profile) as destination:
        destination.write(output_array, 1)


def apply_classwise_gaussian_filter(image):
    valid_mask = image > 0
    labels = np.unique(image[valid_mask])
    if labels.size == 0:
        return image
    filtered_layers = []
    for label in labels:
        class_mask = (image == label).astype(np.float32)
        sigma = class_sigmas.get(int(label), 1.0)
        radius = class_radii.get(int(label), 2)
        filtered_layers.append(gaussian_filter(class_mask, sigma=sigma, truncate=radius / sigma, mode="nearest"))
    filtered_image = labels[np.argmax(np.stack(filtered_layers), axis=0)]
    filtered_image[~valid_mask] = 0
    return filtered_image.astype(np.uint8)


def temporal_majority(images):
    stacked_images = np.stack(images)
    output = np.zeros(stacked_images.shape[1:], dtype=np.uint8)
    best_counts = np.zeros(stacked_images.shape[1:], dtype=np.int16)
    for label in range(1, 18):
        label_counts = np.sum(stacked_images == label, axis=0)
        update_mask = label_counts > best_counts
        output[update_mask] = label
        best_counts[update_mask] = label_counts[update_mask]
    return output


def find_year_map(input_dir, year):
    candidate_paths = sorted(Path(input_dir).glob(f"*{year}*.tif"))
    if len(candidate_paths) != 1:
        raise FileNotFoundError(
            f" {input_dir}  {year}  GeoTIFF"
            f" {len(candidate_paths)} "
        )
    return candidate_paths[0]


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="30 m",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="prepared",
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "full"],
        default="full",
        help="prepare full ",
    )
    arguments = parser.parse_args()
    experiment_config = get_experiment_config("sf_ganet")
    output_root = Path(arguments.output_root).resolve()
    raw_root = Path(arguments.input_dir).resolve()
    input_parts = {path_part.lower() for path_part in raw_root.parts}
    output_parts = {path_part.lower() for path_part in output_root.parts}
    if "training_strategy_s4" in input_parts | output_parts:
        parser.error("S4")
    if not raw_root.is_dir():
        parser.error(f'{raw_root}')
    prepared_root = output_root / "prepared"
    spatial_root = output_root / "spatial_filtered"
    temporal_root = output_root / "temporal_filtered"
    required_directories = [prepared_root]
    if arguments.mode == "full":
        required_directories.extend([spatial_root, temporal_root])
    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)

    years = experiment_config["data"]["years"]
    prepared_maps = {}
    for year in years:
        raw_path = find_year_map(raw_root, year)
        prepared_path = prepared_root / f"LCZ_Map_{year}_prepared_100m.tif"
        clip_and_resample(
            raw_path,
            prepared_path,
            experiment_config["paths"]["boundary_path"],
            experiment_config["inference"]["output_resolution_m"],
        )
        with rasterio.open(prepared_path) as source:
            prepared_maps[year] = (source.read(1), source.profile.copy())

    if arguments.mode == "prepare":
        print(f"100 m {prepared_root}")
        return

    for year in years:
        spatial_path = spatial_root / f"LCZ_Map_{year}_spatial_100m.tif"
        with rasterio.open(spatial_path, "w", **prepared_maps[year][1]) as destination:
            destination.write(
                apply_classwise_gaussian_filter(prepared_maps[year][0]),
                1,
            )

    spatial_maps = {}
    for year in years:
        spatial_path = spatial_root / f"LCZ_Map_{year}_spatial_100m.tif"
        with rasterio.open(spatial_path) as source:
            spatial_maps[year] = (source.read(1), source.profile.copy())
    for year in years:
        window_years = [
            candidate_year
            for candidate_year in years
            if abs(candidate_year - year) <= temporal_half_window
        ]
        filtered_image = temporal_majority(
            [spatial_maps[candidate_year][0] for candidate_year in window_years]
        )
        output_path = temporal_root / f"LCZ_Map_{year}_temporal_100m.tif"
        with rasterio.open(output_path, "w", **spatial_maps[year][1]) as destination:
            destination.write(filtered_image, 1)
    print(f"{output_root}")


if __name__ == "__main__":
    main()


