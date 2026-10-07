r"""
文件作用：将主模型各策略的原始 10 m LCZ 地图裁剪、重采样至 100 m，并可继续执行空间与时间滤波。
流程位置：在主模型或 S1--S3 整图推理后运行；策略比较使用 prepare，最终产品使用 full。
主要输入：年度原始地图目录，以及 config.py 的北京市研究区边界。
主要输出：prepared、spatial_filtered、temporal_filtered 三个子目录下的连续年度 GeoTIFF。
重要参数：--input_dir和--output_root必须显式指定；--mode及滤波参数不变。
运行环境：Python 3.10、Rasterio、GeoPandas、SciPy、NumPy。
前后依赖：输入由主模型或策略推理生成；analysis 模块读取 prepared 或 temporal_filtered 地图。
运行命令：在F:\LCZ\reviewer_code执行D:\Anaconda\envs\LCZ\python.exe -B -m postprocessing.process_maps --input_dir "<本次运行目录>\maps\raw"
--output_root "<本次运行目录>\maps" --mode full；完整命令见材料包根目录README.md。
"""

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


# 类别专属空间滤波参数，单位为输出 100 m 像元；正式使用前应结合验证结果核对。
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
    """按研究区边界裁剪原始地图，并使用最近邻法重采样至目标分辨率。"""
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
    """对每个出现的 LCZ 类别独立平滑，再按最大响应恢复离散类别图。"""
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
    """在一个时间窗内按类别票数输出多数类别，忽略 nodata 的零值。"""
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
    """在一个策略目录中查找唯一包含指定年份的原始 GeoTIFF。"""
    candidate_paths = sorted(Path(input_dir).glob(f"*{year}*.tif"))
    if len(candidate_paths) != 1:
        raise FileNotFoundError(
            f"目录 {input_dir} 中年份 {year} 应有且仅有一幅 GeoTIFF，"
            f"实际找到 {len(candidate_paths)} 幅。"
        )
    return candidate_paths[0]


def main():
    """按准备、空间滤波、时间滤波顺序生成指定范围的地图产品。"""
    parser = argparse.ArgumentParser(
        description="准备策略地图，或生成完整时空后处理产品。"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="必须指定本次运行的原始10 m年度地图目录。",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="必须指定prepared等子目录的父目录，不自动选择旧路径。",
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "full"],
        default="full",
        help="prepare 只裁剪重采样；full 继续执行空间和时间滤波。",
    )
    arguments = parser.parse_args()
    experiment_config = get_experiment_config("sf_ganet")
    output_root = Path(arguments.output_root).resolve()
    raw_root = Path(arguments.input_dir).resolve()
    # S4仅用于训练精度，禁止把其历史或手工地图继续接入后处理。
    input_parts = {path_part.lower() for path_part in raw_root.parts}
    output_parts = {path_part.lower() for path_part in output_root.parts}
    if "training_strategy_s4" in input_parts | output_parts:
        parser.error("S4只报告训练精度，不生成或后处理年度地图。")
    if not raw_root.is_dir():
        parser.error(f'输入地图目录不存在：{raw_root}')
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
        print(f"100 m 策略比较地图已保存至：{prepared_root}")
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
    print(f"后处理地图已保存至：{output_root}")


if __name__ == "__main__":
    main()
