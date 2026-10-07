
import os
from pathlib import Path
import sys

if os.name == "nt":
    runtime_dir = Path(sys.executable).parent
    runtime_bin = runtime_dir / "Library" / "bin"
    if runtime_bin.is_dir():
        os.environ["PATH"] = os.pathsep.join([
            str(runtime_dir), str(runtime_bin), str(runtime_dir / "Scripts"),
            os.environ.get("PATH", ""),
        ])

import numpy as np
import rasterio

from postprocessing.process_maps import (
    apply_classwise_gaussian_filter, temporal_majority, find_year_map,
)


strategy_paths = {
    "s1": Path(
        r"F:\LCZ\outputs\training_strategy_s1\region_five_fold"
        r"\training_strategy_s1_seed42\strategy_best\bd7ee18470de7fe2"
        r"\maps\temporal_majority3\20261004_132923_920944"
    ),
    "s2": Path(
        r"F:\LCZ\outputs\training_strategy_s2\region_five_fold"
        r"\training_strategy_s2_seed42\strategy_best\dea83248d69ccdc0\maps\prepared"
    ),
    "s3": Path(
        r"F:\LCZ\outputs\training_strategy_s3\region_five_fold"
        r"\training_strategy_s3_seed42\strategy_best\812b033d1efd89fe\maps\prepared"
    ),
}
output_root = Path(r"F:\LCZ\outputs\analysis\spatiotemporal_filter")
start_year = 2001
end_year = 2020
temporal_window_years = 5
processing_stages = ("gaussian", "majority5", "gaussian_majority5")


def output_path(strategy, stage, year):
    return output_root / strategy / stage / f"LCZ_Map_{year}_{stage}_100m.tif"


def validate_inputs():
    if not strategy_paths or end_year < start_year:
        raise ValueError(" end_year  start_year")
    if (
        not isinstance(temporal_window_years, int)
        or temporal_window_years < 1
        or temporal_window_years % 2 != 1
    ):
        raise ValueError("temporal_window_years  5")
    allowed_stages = {"gaussian", "majority5", "gaussian_majority5"}
    if (
        not processing_stages
        or len(set(processing_stages)) != len(processing_stages)
        or not set(processing_stages) <= allowed_stages
    ):
        raise ValueError("processing_stages  gaussian/majority5/gaussian_majority5")
    years = list(range(start_year, end_year + 1))
    map_paths = {}
    for strategy, input_dir in strategy_paths.items():
        if not strategy or Path(strategy).name != strategy or strategy in {".", ".."}:
            raise ValueError(f"{strategy}")
        if not input_dir.is_dir():
            raise FileNotFoundError(f"{input_dir}")
        if output_root.resolve() == input_dir.resolve():
            raise ValueError("output_root ")
        map_paths[strategy] = {}
        grid = None
        for year in years:
            path = find_year_map(input_dir, year)
            with rasterio.open(path) as source:
                if source.count != 1 or source.crs is None or not source.crs.is_projected:
                    raise ValueError(f" CRS  LCZ {path}")
                unit_factor = source.crs.linear_units_factor[1]
                resolution_m = np.asarray(source.res) * unit_factor
                if not np.allclose(resolution_m, (100, 100), rtol=0, atol=1e-6):
                    raise ValueError(f" 100 m {path} {resolution_m} m")
                current_grid = (source.crs, source.transform, source.width, source.height)
                if grid is not None and current_grid != grid:
                    raise ValueError(f" CRS{path}")
                grid = current_grid
            map_paths[strategy][year] = path
            for stage in processing_stages:
                target = output_path(strategy, stage, year)
                if target.exists():
                    raise FileExistsError(
                        f"{target}\n"
                        " output_root"
                    )
        print(f"{strategy.upper()}{len(years)} 100 m ", flush=True)
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
                raise ValueError(f" 0--17 {path}")
            images[year] = image.filled(0).astype(np.uint8)
            if profile is None:
                profile = source.profile.copy()
    return images, profile


def filter_temporal_year(images, years, year):
    half_window = temporal_window_years // 2
    window_years = [candidate for candidate in years if abs(candidate - year) <= half_window]
    filtered = temporal_majority([images[candidate] for candidate in window_years])
    filtered[images[year] == 0] = 0
    return filtered, window_years


def write_map(image, profile, path, strategy, stage, source_path, window_years=None):
    if path.exists():
        raise FileExistsError(f"{path}")
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff", count=1, dtype="uint8", nodata=0,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(image, 1)
        tags = {
            "strategy": strategy,
            "processing_stage": stage,
            "source_path": str(source_path),
            "clip_or_resample": "false",
            "target_nodata_preserved": "true",
        }
        if stage in {"gaussian", "gaussian_majority5"}:
            tags.update({"class_radii": str(class_radii), "class_sigmas": str(class_sigmas)})
        if window_years is not None:
            tags.update({
                "temporal_window_years": str(temporal_window_years),
                "actual_window_years": ",".join(map(str, window_years)),
                "temporal_tie_rule": "smaller_class",
            })
        destination.update_tags(**tags)


def process_strategy(strategy, years, map_paths):
    images, profile = load_maps(map_paths)
    for stage in processing_stages:
        (output_root / strategy / stage).mkdir(parents=True, exist_ok=True)
    spatial_maps = {}
    if set(processing_stages) & {"gaussian", "gaussian_majority5"}:
        for index, year in enumerate(years, 1):
            spatial_maps[year] = apply_classwise_gaussian_filter(images[year])
            if "gaussian" in processing_stages:
                write_map(
                    spatial_maps[year], profile, output_path(strategy, "gaussian", year),
                    strategy, "gaussian", map_paths[year],
                )
            print(f"{strategy.upper()} {index}/{len(years)}{year}", flush=True)
    for stage in processing_stages:
        if stage == "gaussian":
            continue
        temporal_inputs = images if stage == "majority5" else spatial_maps
        for index, year in enumerate(years, 1):
            filtered, window_years = filter_temporal_year(temporal_inputs, years, year)
            write_map(
                filtered, profile, output_path(strategy, stage, year),
                strategy, stage, map_paths[year], window_years,
            )
            print(f"{strategy.upper()} {stage}{index}/{len(years)}{year}", flush=True)
    print(f"{strategy.upper()} {output_root / strategy}", flush=True)


def main():
    if len(sys.argv) > 1:
        raise SystemExit("")
    print(" 100 m ", flush=True)
    print(f"{output_root}", flush=True)
    for strategy, path in strategy_paths.items():
        print(f"{strategy.upper()} {path}", flush=True)
    years, map_paths = validate_inputs()
    for strategy in strategy_paths:
        process_strategy(strategy, years, map_paths[strategy])
    print(" CSV", flush=True)


if __name__ == "__main__":
    main()

