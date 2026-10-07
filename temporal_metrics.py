r"""
作用：复算 TCR/TNI 和 S1 对 S2/S3 的双侧年度配对检验；位于地图生成之后。
输入：前部固定 input_dir 下的 tcr_paired.csv、tni_paired.csv，包含年度和像元计数。
参数：2001--2020 年、类别 1--17、置信水平 0.95；TCR 为百分数，TNI 为 0--1 比率。
输出：output_dir 下的 tcr_paired.csv、tni_paired.csv、temporal_tests.csv；覆盖同名文件，不执行滤波。
依赖：NumPy、Pandas、SciPy；--maps 从原地图复算，另需 Rasterio。
上下游：读取已有产品或计数表，结果用于年度稳定性比较；calculate_tests 供导出检验表调用。
直接运行：选择 D:\Anaconda\envs\LCZ\python.exe，路径已配置，不需传入参数。
运行：conda run --no-capture-output -n LCZ python -B -X utf8 F:\LCZ\outputs\analysis\map_change_reference\s1_vs_s2_s3\external_lcz_reference\paired_results\temporal_metrics.py
地图复算：在上述命令末尾加 --maps。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

# 固定配对表目录，直接运行不依赖当前工作目录。
input_dir = Path(
    r'F:\LCZ\outputs\analysis\map_change_reference\s1_vs_s2_s3'
    r'\external_lcz_reference\paired_results'
)
output_dir = input_dir
start_year, end_year = 2001, 2020
confidence_level = 0.95
strategy_paths = {
    's1': Path(r'F:\LCZ\outputs\training_strategy_s1\region_five_fold'
               r'\training_strategy_s1_seed42\strategy_best\bd7ee18470de7fe2'
               r'\maps\prepared'),
    's2': Path(r'F:\LCZ\outputs\training_strategy_s2\region_five_fold'
               r'\training_strategy_s2_seed42\strategy_best\dea83248d69ccdc0\maps\prepared'),
    's3': Path(r'F:\LCZ\outputs\training_strategy_s3\region_five_fold'
               r'\training_strategy_s3_seed42\strategy_best\812b033d1efd89fe\maps\prepared'),
}


def calculate_tcr(first, second, valid):
    # 分母为各策略在相邻两年共同有效的像元数。
    count = int(valid.sum())
    changed = int((valid & (first != second)).sum())
    return (100.0 * changed / count if count else np.nan), changed


def calculate_tni(previous, current, following, valid):
    # 分母为三年内至少变化一次的像元数，分子为 A→B→A。
    changed = valid & ((previous != current) | (current != following))
    count = int(changed.sum())
    oscillation = int((changed & (previous == following)).sum())
    value = oscillation / count if count else (0.0 if valid.any() else np.nan)
    return value, count, oscillation


def calculate_map_metrics():
    import rasterio

    history, tcr_rows, tni_rows = [], [], []
    expected_grid = None
    for year in range(start_year, end_year + 1):
        maps = {}
        for strategy, directory in strategy_paths.items():
            paths = sorted(directory.glob(f'*{year}*.tif'))
            if len(paths) != 1:
                raise ValueError(f'{strategy} {year} 应恰有一幅地图。')
            with rasterio.open(paths[0]) as source:
                grid = (source.shape, source.crs, source.transform)
                if source.count != 1 or source.crs is None:
                    raise ValueError('需要有 CRS 的单波段地图。')
                if not np.issubdtype(np.dtype(source.dtypes[0]), np.integer):
                    raise ValueError('地图类别必须为整数。')
                if expected_grid is None:
                    expected_grid = grid
                if grid != expected_grid:
                    raise ValueError('地图网格不一致。')
                maps[strategy] = source.read(1, masked=True).filled(0)
        common = np.logical_and.reduce([(a >= 1) & (a <= 17) for a in maps.values()])
        history.append((year, maps, common))
        if len(history) >= 2:
            previous_year, previous, previous_valid = history[-2]
            valid = previous_valid & common
            row = dict(from_year=previous_year, to_year=year, common_valid_pixels=int(valid.sum()))
            for strategy in strategy_paths:
                row[strategy], row[f'{strategy}_changed_pixels'] = calculate_tcr(
                    previous[strategy], maps[strategy], valid
                )
            tcr_rows.append(row)
        if len(history) == 3:
            first_year, first, first_valid = history[0]
            valid = first_valid & previous_valid & common
            row = dict(year=previous_year, common_valid_pixels=int(valid.sum()))
            for strategy in strategy_paths:
                value, changed, oscillation = calculate_tni(
                    first[strategy], previous[strategy], maps[strategy], valid
                )
                row.update({strategy: value, f'{strategy}_changed_pixels': changed,
                            f'{strategy}_oscillation_pixels': oscillation})
            tni_rows.append(row)
            history.pop(0)
        print(f'已计算 {year}', flush=True)
    return pd.DataFrame(tcr_rows), pd.DataFrame(tni_rows)


def read_paired_metrics():
    tcr = pd.read_csv(input_dir / 'tcr_paired.csv')
    tni = pd.read_csv(input_dir / 'tni_paired.csv')
    for strategy in strategy_paths:
        tcr[strategy] = 100 * tcr[f'{strategy}_changed_pixels'] / tcr.common_valid_pixels.replace(0, np.nan)
        changed = tni[f'{strategy}_changed_pixels']
        tni[strategy] = tni[f'{strategy}_oscillation_pixels'] / changed.replace(0, np.nan)
        tni.loc[changed.eq(0) & tni.common_valid_pixels.gt(0), strategy] = 0.0
    return tcr, tni


def calculate_tests(tcr, tni):
    rows = []
    for metric, table, year_column in [('tcr_percent', tcr, 'from_year'), ('tni', tni, 'year')]:
        if table[year_column].duplicated().any():
            raise ValueError('配对年份重复。')
        for other in ('s2', 's3'):
            pairs = table[[year_column, 's1', other]].replace([np.inf, -np.inf], np.nan)
            pairs = pairs.dropna().sort_values(year_column)
            first, second = pairs.s1.to_numpy(), pairs[other].to_numpy()
            difference, count = first - second, len(pairs)
            if count < 2 or not np.all(np.diff(pairs[year_column]) == 1):
                raise ValueError('需要至少两个连续年度配对。')
            mean, std = difference.mean(), difference.std(ddof=1)
            se, critical = std / np.sqrt(count), t.ppf((1 + confidence_level) / 2, count - 1)
            statistic = mean / se if se else (np.copysign(np.inf, mean) if mean else np.nan)
            dz = mean / std if std else (np.copysign(np.inf, mean) if mean else 0.0)
            # Bartlett 权重与 n/(n-1) 小样本修正，和现有年度 HAC 口径一致。
            lags = min(int(np.floor(4 * (count / 100) ** (2 / 9))), count - 1)
            centered = difference - mean
            covariance = np.dot(centered, centered)
            for lag in range(1, lags + 1):
                covariance += 2 * (1 - lag / (lags + 1)) * np.dot(centered[lag:], centered[:-lag])
            hac_se = np.sqrt(max(covariance, 0) / (count * (count - 1)))
            hac_statistic = mean / hac_se if hac_se else np.nan
            rows.append(dict(
                metric=metric, first_strategy='s1', second_strategy=other, pair_count=count,
                first_mean=first.mean(), second_mean=second.mean(), mean_difference=mean,
                difference_std=std, degrees_of_freedom=count - 1, standard_error=se,
                t_statistic=statistic, p_value=2 * t.sf(abs(statistic), count - 1),
                paired_cohens_d=dz, absolute_cohens_d=abs(dz),
                first_lower_count=int((difference < 0).sum()), first_higher_count=int((difference > 0).sum()),
                ci_lower=mean - critical * se, ci_upper=mean + critical * se,
                hac_lags=lags, hac_degrees_of_freedom=count - 1, hac_standard_error=hac_se,
                hac_t_statistic=hac_statistic, hac_p_value=2 * t.sf(abs(hac_statistic), count - 1),
                hac_confidence_level=confidence_level,
                hac_ci_lower=mean - critical * hac_se if hac_se else np.nan,
                hac_ci_upper=mean + critical * hac_se if hac_se else np.nan,
            ))
    results = pd.DataFrame(rows)
    p_values = results.hac_p_value.to_numpy()
    order = np.flatnonzero(np.isfinite(p_values))
    order = order[np.argsort(p_values[order])]
    adjusted = np.full(len(results), np.nan)
    adjusted[order] = np.minimum(np.maximum.accumulate(
        p_values[order] * (len(results) - np.arange(len(order)))
    ), 1.0)
    results['hac_holm_p_value'], results['holm_family_size'] = adjusted, len(results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', action='store_true')
    args = parser.parse_args()
    tcr, tni = calculate_map_metrics() if args.maps else read_paired_metrics()
    tests = calculate_tests(tcr, tni)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 三张正式结果表沿用原文件名，重复运行直接覆盖。
    for filename, table in [
        ('tcr_paired.csv', tcr),
        ('tni_paired.csv', tni),
        ('temporal_tests.csv', tests),
    ]:
        output_csv = output_dir / filename
        table.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f'已保存：{output_csv}')
