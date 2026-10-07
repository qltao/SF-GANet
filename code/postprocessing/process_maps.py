import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)
from scipy.ndimage import gaussian_filter

from config import get_experiment_config


class_radii = {
    1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3,
    8: 5, 9: 3, 10: 5, 11: 2, 12: 2, 13: 2, 14: 2,
    15: 3, 16: 2, 17: 1,
}
class_sigmas = {
    1: 1, 2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5,
    7: 1.5, 8: 2.5, 9: 1.5, 10: 2.5, 11: 1, 12: 1,
    13: 1, 14: 1, 15: 1.5, 16: 1, 17: 0.5,
}
temporal_half_window = 2


def clip_and_resample(
    input_path,
    output_path,
    boundary_path,
    output_resolution_m,
):
    with rasterio.open(input_path) as source:
        boundary = gpd.read_file(boundary_path)
        if boundary.crs != source.crs:
            boundary = boundary.to_crs(source.crs)

        geometries = [
            feature["geometry"]
            for _, feature in boundary.iterrows()
        ]
        clipped_data, clipped_transform = mask(
            source,
            geometries,
            crop=True,
            nodata=0,
        )
        clipped_bounds = array_bounds(
            clipped_data.shape[1],
            clipped_data.shape[2],
            clipped_transform,
        )
        transform, width, height = calculate_default_transform(
            source.crs,
            source.crs,
            clipped_data.shape[2],
            clipped_data.shape[1],
            *clipped_bounds,
            resolution=(
                output_resolution_m,
                output_resolution_m,
            ),
        )

        profile = source.profile.copy()
        profile.update(
            transform=transform,
            width=width,
            height=height,
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
        )
        output_array = np.zeros(
            (height, width),
            dtype=np.uint8,
        )
        reproject(
            source=clipped_data[0],
            destination=output_array,
            src_transform=clipped_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=source.crs,
            resampling=Resampling.nearest,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
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
        filtered_layers.append(
            gaussian_filter(
                class_mask,
                sigma=sigma,
                truncate=radius / sigma,
                mode="nearest",
            )
        )

    filtered_image = labels[
        np.argmax(np.stack(filtered_layers), axis=0)
    ]
    filtered_image[~valid_mask] = 0
    return filtered_image.astype(np.uint8)


def temporal_majority(images):
    stacked = np.stack(images)
    output = np.zeros(
        stacked.shape[1:],
        dtype=np.uint8,
    )
    best_counts = np.zeros(
        stacked.shape[1:],
        dtype=np.int16,
    )
    for label in range(1, 18):
        counts = np.sum(stacked == label, axis=0)
        update = counts > best_counts
        output[update] = label
        best_counts[update] = counts[update]
    return output


def find_year_map(input_dir, year):
    candidates = sorted(
        Path(input_dir).glob(f"*{year}*.tif")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one map for {year} in "
            f"{input_dir}, found {len(candidates)}."
        )
    return candidates[0]


def prepare_raw_100m(
    input_dir,
    output_root,
    experiment_config,
):
    raw_100m_root = output_root / "raw_100m"
    raw_100m_root.mkdir(parents=True, exist_ok=True)

    raw_maps = {}
    for year in experiment_config["data"]["years"]:
        input_path = find_year_map(input_dir, year)
        output_path = (
            raw_100m_root
            / f"LCZ_Map_{year}_raw_100m.tif"
        )
        clip_and_resample(
            input_path,
            output_path,
            experiment_config["paths"]["boundary_path"],
            experiment_config["inference"][
                "output_resolution_m"
            ],
        )
        with rasterio.open(output_path) as source:
            raw_maps[year] = (
                source.read(1),
                source.profile.copy(),
            )
    return raw_100m_root, raw_maps


def apply_full_postprocessing(
    raw_maps,
    output_root,
    years,
):
    gaussian_root = output_root / "gaussian"
    majority_root = output_root / "majority5"
    combined_root = output_root / "gaussian_majority5"

    for directory in [
        gaussian_root,
        majority_root,
        combined_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    gaussian_maps = {}
    for year in years:
        image, profile = raw_maps[year]
        filtered = apply_classwise_gaussian_filter(image)
        gaussian_maps[year] = (filtered, profile)
        with rasterio.open(
            gaussian_root / f"LCZ_Map_{year}_gaussian_100m.tif",
            "w",
            **profile,
        ) as destination:
            destination.write(filtered, 1)

    for year in years:
        window_years = [
            candidate
            for candidate in years
            if abs(candidate - year) <= temporal_half_window
        ]

        majority = temporal_majority(
            [raw_maps[candidate][0] for candidate in window_years]
        )
        majority[raw_maps[year][0] == 0] = 0
        with rasterio.open(
            majority_root / f"LCZ_Map_{year}_majority5_100m.tif",
            "w",
            **raw_maps[year][1],
        ) as destination:
            destination.write(majority, 1)

        combined = temporal_majority(
            [
                gaussian_maps[candidate][0]
                for candidate in window_years
            ]
        )
        combined[raw_maps[year][0] == 0] = 0
        with rasterio.open(
            combined_root
            / f"LCZ_Map_{year}_gaussian_majority5_100m.tif",
            "w",
            **raw_maps[year][1],
        ) as destination:
            destination.write(combined, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--mode",
        choices=["prepare", "full"],
        default="full",
    )
    args = parser.parse_args()

    experiment_config = get_experiment_config("sf_ganet")
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    raw_100m_root, raw_maps = prepare_raw_100m(
        input_dir,
        output_root,
        experiment_config,
    )

    if args.mode == "full":
        apply_full_postprocessing(
            raw_maps,
            output_root,
            experiment_config["data"]["years"],
        )

    print(raw_100m_root)


if __name__ == "__main__":
    main()
