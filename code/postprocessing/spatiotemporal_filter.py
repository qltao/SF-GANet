import argparse
from pathlib import Path

import numpy as np
import rasterio

from postprocessing.process_maps import (
    apply_classwise_gaussian_filter,
    find_year_map,
    temporal_majority,
)


start_year = 2001
end_year = 2020
temporal_window_years = 5
processing_stages = (
    "gaussian",
    "majority5",
    "gaussian_majority5",
)


def output_path(output_root, strategy, stage, year):
    return (
        output_root
        / strategy
        / stage
        / f"LCZ_Map_{year}_{stage}_100m.tif"
    )


def validate_inputs(strategy_paths):
    years = list(range(start_year, end_year + 1))
    map_paths = {}

    for strategy, input_dir in strategy_paths.items():
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory not found: {input_dir}"
            )

        map_paths[strategy] = {}
        expected_grid = None
        for year in years:
            path = find_year_map(input_dir, year)
            with rasterio.open(path) as source:
                if (
                    source.count != 1
                    or source.crs is None
                    or not source.crs.is_projected
                ):
                    raise ValueError(
                        f"Invalid LCZ raster: {path}"
                    )

                unit_factor = source.crs.linear_units_factor[1]
                resolution_m = (
                    np.asarray(source.res) * unit_factor
                )
                if not np.allclose(
                    resolution_m,
                    (100, 100),
                    rtol=0,
                    atol=1e-6,
                ):
                    raise ValueError(
                        f"Expected 100 m input: {path}"
                    )

                grid = (
                    source.crs,
                    source.transform,
                    source.width,
                    source.height,
                )
                if expected_grid is None:
                    expected_grid = grid
                elif grid != expected_grid:
                    raise ValueError(
                        f"Grid mismatch within {strategy}: {path}"
                    )

            map_paths[strategy][year] = path

    return years, map_paths


def load_maps(map_paths):
    images = {}
    profile = None
    for year, path in map_paths.items():
        with rasterio.open(path) as source:
            image = source.read(1, masked=True)
            values = image.compressed()
            if (
                not np.isfinite(values).all()
                or ((values < 0) | (values > 17)).any()
                or (values != np.floor(values)).any()
            ):
                raise ValueError(
                    f"Invalid LCZ labels in {path}"
                )
            images[year] = image.filled(0).astype(np.uint8)
            if profile is None:
                profile = source.profile.copy()
    return images, profile


def temporal_year(images, years, year):
    half_window = temporal_window_years // 2
    window_years = [
        candidate
        for candidate in years
        if abs(candidate - year) <= half_window
    ]
    filtered = temporal_majority(
        [images[candidate] for candidate in window_years]
    )
    filtered[images[year] == 0] = 0
    return filtered


def write_map(image, profile, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(image, 1)


def process_strategy(
    strategy,
    years,
    map_paths,
    output_root,
):
    images, profile = load_maps(map_paths)

    gaussian_maps = {}
    for year in years:
        gaussian_maps[year] = apply_classwise_gaussian_filter(
            images[year]
        )
        if "gaussian" in processing_stages:
            write_map(
                gaussian_maps[year],
                profile,
                output_path(
                    output_root,
                    strategy,
                    "gaussian",
                    year,
                ),
            )

    if "majority5" in processing_stages:
        for year in years:
            write_map(
                temporal_year(images, years, year),
                profile,
                output_path(
                    output_root,
                    strategy,
                    "majority5",
                    year,
                ),
            )

    if "gaussian_majority5" in processing_stages:
        for year in years:
            filtered = temporal_year(
                gaussian_maps,
                years,
                year,
            )
            filtered[images[year] == 0] = 0
            write_map(
                filtered,
                profile,
                output_path(
                    output_root,
                    strategy,
                    "gaussian_majority5",
                    year,
                ),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1_dir", required=True)
    parser.add_argument("--s2_dir", required=True)
    parser.add_argument("--s3_dir", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    strategy_paths = {
        "s1": Path(args.s1_dir),
        "s2": Path(args.s2_dir),
        "s3": Path(args.s3_dir),
    }
    output_root = Path(args.output_root)

    years, map_paths = validate_inputs(strategy_paths)
    for strategy in ("s1", "s2", "s3"):
        process_strategy(
            strategy,
            years,
            map_paths[strategy],
            output_root,
        )


if __name__ == "__main__":
    main()
