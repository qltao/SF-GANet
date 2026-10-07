r"""
文件作用：直接对已有 100 m 年度 LCZ 分类地图执行空间与时间滤波，不裁剪、不重采样。
流程位置：100 m 地图已生成后运行；下游可读取输出地图计算精度、TCR/TNI 或提取样点。
主要输入：前部 strategy_paths 中的年度 GeoTIFF 目录；每年一幅单波段地图，类别 1--17，0 为无效值。
默认来源：S1 为指定的保守时间纠正版；S2/S3 为各自 prepared 原图，不改变这些基线。
主要输出：output_root/策略名/处理条件 下的年度 GeoTIFF；三个条件各自从相同基线出发。
条件含义：gaussian 仅高斯；majority5 仅普通五年多数；gaussian_majority5 先高斯再普通五年多数。
重要参数：固定输入/输出路径、start_year/end_year、processing_stages 和 temporal_window_years 在前部。
算法实现：高斯、多数投票及年度文件查找复用同目录 process_maps.py，参数保持原样。
高斯参数：沿用原 class_radii/class_sigmas，单位为 100 m 像元，分别平滑类别掩膜后取最大响应。
时间规则：默认居中五年窗，首尾窗口截短；忽略零值，平票取较小类别；保留目标年的无效范围。
输出规则：保留输入 CRS、范围、行列数及仿射变换；输出 uint8、nodata=0，不覆盖已有输出地图。
环境依赖：项目 LCZ 环境中的 Python、NumPy、Rasterio、SciPy；不读取研究区矢量边界。
运行方式：在材料包根目录以模块方式调用，无需命令行参数。
完整命令：cd /d F:\LCZ\reviewer_code 后执行 D:\Anaconda\envs\LCZ\python.exe -B -m postprocessing.spatiotemporal_filter
上下游关系：只生成滤波地图，不改训练权重、样点表、指标 CSV 或绘图脚本的输入。
"""

import os
from pathlib import Path
import sys

# 支持 Windows 环境直接加载栅格运行库，无需导入项目其他模块。
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


# 固定输入：不添加 raw、prepared 或其他后缀，直接读取下列目录中的 100 m 地图。
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
    """先核对年份、100 m 网格及输出冲突，避免错误输入启动整批处理。"""
    if not strategy_paths or end_year < start_year:
        raise ValueError("请配置至少一组输入，并保证 end_year 不小于 start_year。")
    if (
        not isinstance(temporal_window_years, int)
        or temporal_window_years < 1
        or temporal_window_years % 2 != 1
    ):
        raise ValueError("temporal_window_years 必须为正奇数，默认普通五年多数使用 5。")
    allowed_stages = {"gaussian", "majority5", "gaussian_majority5"}
    if (
        not processing_stages
        or len(set(processing_stages)) != len(processing_stages)
        or not set(processing_stages) <= allowed_stages
    ):
        raise ValueError("processing_stages 须为不重复的 gaussian/majority5/gaussian_majority5。")
    years = list(range(start_year, end_year + 1))
    map_paths = {}
    for strategy, input_dir in strategy_paths.items():
        if not strategy or Path(strategy).name != strategy or strategy in {".", ".."}:
            raise ValueError(f"策略名必须是普通目录名：{strategy}")
        if not input_dir.is_dir():
            raise FileNotFoundError(f"输入目录不存在：{input_dir}")
        if output_root.resolve() == input_dir.resolve():
            raise ValueError("output_root 不能与输入地图目录相同。")
        map_paths[strategy] = {}
        grid = None
        for year in years:
            path = find_year_map(input_dir, year)
            with rasterio.open(path) as source:
                if source.count != 1 or source.crs is None or not source.crs.is_projected:
                    raise ValueError(f"输入须为有投影 CRS 的单波段 LCZ 地图：{path}")
                unit_factor = source.crs.linear_units_factor[1]
                resolution_m = np.asarray(source.res) * unit_factor
                if not np.allclose(resolution_m, (100, 100), rtol=0, atol=1e-6):
                    raise ValueError(f"输入不是 100 m 地图：{path}；像元大小为 {resolution_m} m")
                current_grid = (source.crs, source.transform, source.width, source.height)
                if grid is not None and current_grid != grid:
                    raise ValueError(f"同一策略各年度地图的 CRS、范围或网格不同：{path}")
                grid = current_grid
            map_paths[strategy][year] = path
            for stage in processing_stages:
                target = output_path(strategy, stage, year)
                if target.exists():
                    raise FileExistsError(
                        f"输出已存在，未覆盖：{target}\n"
                        "如需另存一套结果，请修改前部 output_root。"
                    )
        print(f"输入核对通过：{strategy.upper()}，{len(years)} 年，100 m 同网格。", flush=True)
    return years, map_paths


def load_maps(map_paths):
    """读取年度类别图，将原掩膜及零值保留为无效像元。"""
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
                raise ValueError(f"地图包含 0--17 之外或非整数的类别：{path}")
            images[year] = image.filled(0).astype(np.uint8)
            if profile is None:
                profile = source.profile.copy()
    return images, profile


def filter_temporal_year(images, years, year):
    """执行普通居中多数滤波，首尾截短，目标年原无效像元保持为零。"""
    half_window = temporal_window_years // 2
    window_years = [candidate for candidate in years if abs(candidate - year) <= half_window]
    filtered = temporal_majority([images[candidate] for candidate in window_years])
    # 不将邻年类别填入目标年的原无效范围，避免扩张地图有效区。
    filtered[images[year] == 0] = 0
    return filtered, window_years


def write_map(image, profile, path, strategy, stage, source_path, window_years=None):
    """只改变类别值和存储方式，不改变输入地图的空间网格。"""
    if path.exists():
        raise FileExistsError(f"输出已存在，未覆盖：{path}")
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
    """同一基线分别生成单项滤波与组合滤波，避免条件串行累加。"""
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
            print(f"{strategy.upper()} 高斯滤波：{index}/{len(years)}，{year}", flush=True)
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
            print(f"{strategy.upper()} {stage}：{index}/{len(years)}，{year}", flush=True)
    print(f"{strategy.upper()} 已完成：{output_root / strategy}", flush=True)


def main():
    if len(sys.argv) > 1:
        raise SystemExit("本脚本无需命令行参数，请修改前部固定路径及滤波配置后直接运行。")
    print("直接读取 100 m 地图，不裁剪、不重采样。", flush=True)
    print(f"输出根目录：{output_root}", flush=True)
    for strategy, path in strategy_paths.items():
        print(f"{strategy.upper()} 输入：{path}", flush=True)
    years, map_paths = validate_inputs()
    for strategy in strategy_paths:
        process_strategy(strategy, years, map_paths[strategy])
    print("全部完成；未修改输入地图、样点表或指标 CSV。", flush=True)


if __name__ == "__main__":
    main()
