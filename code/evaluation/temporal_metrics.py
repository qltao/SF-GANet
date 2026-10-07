import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


start_year = 2001
end_year = 2020
confidence_level = 0.95


def calculate_tcr(first, second, valid):
    count = int(valid.sum())
    changed = int((valid & (first != second)).sum())
    value = 100.0 * changed / count if count else np.nan
    return value, changed


def calculate_tni(previous, current, following, valid):
    changed = valid & (
        (previous != current) | (current != following)
    )
    count = int(changed.sum())
    oscillation = int(
        (changed & (previous == following)).sum()
    )
    value = (
        oscillation / count
        if count
        else (0.0 if valid.any() else np.nan)
    )
    return value, count, oscillation


def unique_year_map(directory, year):
    paths = sorted(Path(directory).glob(f"*{year}*.tif"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one map for {year} in "
            f"{directory}, found {len(paths)}."
        )
    return paths[0]


def calculate_map_metrics(strategy_paths):
    import rasterio

    history = []
    tcr_rows = []
    tni_rows = []
    expected_grid = None

    for year in range(start_year, end_year + 1):
        maps = {}
        for strategy, directory in strategy_paths.items():
            path = unique_year_map(directory, year)
            with rasterio.open(path) as source:
                grid = (
                    source.shape,
                    source.crs,
                    source.transform,
                )
                if expected_grid is None:
                    expected_grid = grid
                elif grid != expected_grid:
                    raise ValueError(
                        f"Grid mismatch: {path}"
                    )
                maps[strategy] = (
                    source.read(1, masked=True).filled(0)
                )

        common = np.logical_and.reduce(
            [
                (array >= 1) & (array <= 17)
                for array in maps.values()
            ]
        )
        history.append((year, maps, common))

        if len(history) >= 2:
            previous_year, previous, previous_valid = history[-2]
            valid = previous_valid & common
            row = {
                "from_year": previous_year,
                "to_year": year,
                "common_valid_pixels": int(valid.sum()),
            }
            for strategy in strategy_paths:
                (
                    row[strategy],
                    row[f"{strategy}_changed_pixels"],
                ) = calculate_tcr(
                    previous[strategy],
                    maps[strategy],
                    valid,
                )
            tcr_rows.append(row)

        if len(history) == 3:
            _, first, first_valid = history[0]
            previous_year, previous, previous_valid = history[1]
            valid = first_valid & previous_valid & common
            row = {
                "year": previous_year,
                "common_valid_pixels": int(valid.sum()),
            }
            for strategy in strategy_paths:
                value, changed, oscillation = calculate_tni(
                    first[strategy],
                    previous[strategy],
                    maps[strategy],
                    valid,
                )
                row.update(
                    {
                        strategy: value,
                        f"{strategy}_changed_pixels": changed,
                        f"{strategy}_oscillation_pixels": oscillation,
                    }
                )
            tni_rows.append(row)
            history.pop(0)

    return pd.DataFrame(tcr_rows), pd.DataFrame(tni_rows)


def read_paired_metrics(input_dir):
    input_dir = Path(input_dir)
    tcr = pd.read_csv(input_dir / "tcr_paired.csv")
    tni = pd.read_csv(input_dir / "tni_paired.csv")

    for strategy in ("s1", "s2", "s3"):
        tcr[strategy] = (
            100
            * tcr[f"{strategy}_changed_pixels"]
            / tcr["common_valid_pixels"].replace(0, np.nan)
        )
        changed = tni[f"{strategy}_changed_pixels"]
        tni[strategy] = (
            tni[f"{strategy}_oscillation_pixels"]
            / changed.replace(0, np.nan)
        )
        tni.loc[
            changed.eq(0)
            & tni["common_valid_pixels"].gt(0),
            strategy,
        ] = 0.0

    return tcr, tni


def calculate_tests(tcr, tni):
    rows = []

    for metric, table, year_column in [
        ("tcr_percent", tcr, "from_year"),
        ("tni", tni, "year"),
    ]:
        for other in ("s2", "s3"):
            pairs = table[
                [year_column, "s1", other]
            ].replace([np.inf, -np.inf], np.nan)
            pairs = pairs.dropna().sort_values(year_column)

            first = pairs["s1"].to_numpy()
            second = pairs[other].to_numpy()
            difference = first - second
            count = len(pairs)
            if count < 2:
                raise ValueError(
                    "At least two paired years are required."
                )

            mean = difference.mean()
            std = difference.std(ddof=1)
            se = std / np.sqrt(count)
            critical = t.ppf(
                (1 + confidence_level) / 2,
                count - 1,
            )
            statistic = mean / se if se else np.nan
            dz = mean / std if std else 0.0

            lags = min(
                int(np.floor(4 * (count / 100) ** (2 / 9))),
                count - 1,
            )
            centered = difference - mean
            covariance = np.dot(centered, centered)
            for lag in range(1, lags + 1):
                covariance += (
                    2
                    * (1 - lag / (lags + 1))
                    * np.dot(
                        centered[lag:],
                        centered[:-lag],
                    )
                )

            hac_se = np.sqrt(
                max(covariance, 0)
                / (count * (count - 1))
            )
            hac_statistic = (
                mean / hac_se if hac_se else np.nan
            )

            rows.append(
                {
                    "metric": metric,
                    "first_strategy": "s1",
                    "second_strategy": other,
                    "pair_count": count,
                    "first_mean": first.mean(),
                    "second_mean": second.mean(),
                    "mean_difference": mean,
                    "difference_std": std,
                    "paired_cohens_d": dz,
                    "hac_lags": lags,
                    "hac_standard_error": hac_se,
                    "hac_t_statistic": hac_statistic,
                    "hac_p_value": 2
                    * t.sf(
                        abs(hac_statistic),
                        count - 1,
                    ),
                    "hac_ci_lower": mean
                    - critical * hac_se,
                    "hac_ci_upper": mean
                    + critical * hac_se,
                }
            )

    results = pd.DataFrame(rows)
    p_values = results["hac_p_value"].to_numpy()
    order = np.argsort(p_values)
    adjusted = np.empty(len(results), dtype=float)
    adjusted[order] = np.minimum(
        np.maximum.accumulate(
            p_values[order]
            * (
                len(results)
                - np.arange(len(results))
            )
        ),
        1.0,
    )
    results["hac_holm_p_value"] = adjusted
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--s1_dir")
    parser.add_argument("--s2_dir")
    parser.add_argument("--s3_dir")
    args = parser.parse_args()

    if args.maps:
        required = [args.s1_dir, args.s2_dir, args.s3_dir]
        if any(value is None for value in required):
            parser.error(
                "--maps requires --s1_dir, --s2_dir, and --s3_dir."
            )
        strategy_paths = {
            "s1": Path(args.s1_dir),
            "s2": Path(args.s2_dir),
            "s3": Path(args.s3_dir),
        }
        tcr, tni = calculate_map_metrics(strategy_paths)
    else:
        tcr, tni = read_paired_metrics(args.input_dir)

    tests = calculate_tests(tcr, tni)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tcr.to_csv(output_dir / "tcr_paired.csv", index=False)
    tni.to_csv(output_dir / "tni_paired.csv", index=False)
    tests.to_csv(
        output_dir / "temporal_tests.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
