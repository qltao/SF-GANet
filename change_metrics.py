r"""
作用：由 change_paired.csv 复算分类精度、变化率、误报/召回和正确变化保留率。
输入：前部固定 input_csv 指定的 800 点宽表，含四年年份及参考/预测类别。
字段：s1/s2/s3 的 before、gaussian、majority5、gaussian_majority5；sequential_yft 为单组顺序微调结果。
参数：类别 1--17；OA 使用两端观测，Macro-F1 固定 17 类，无支持类 F1 为 0。
口径：顺序微调取原表第4折已有赋值版本，与S1/S2/S3同折；保留率以同策略 before 的正确变化为分母。
输出：output_csv 指定的 change_metrics.csv，比例为 0--1；覆盖同名文件。
上下游：读取已配对样点，用于后处理及变化比较；不读地图、不抽样。依赖 NumPy、Pandas。
直接运行：选择 D:\Anaconda\envs\LCZ\python.exe，路径已配置，不需传入参数。
运行：conda run --no-capture-output -n LCZ python -B -X utf8 F:\LCZ\outputs\analysis\map_change_reference\s1_vs_s2_s3\external_lcz_reference\paired_results\change_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 固定输入表路径，直接运行不依赖当前工作目录。
input_csv = Path(
    r'F:\LCZ\outputs\analysis\map_change_reference\s1_vs_s2_s3'
    r'\external_lcz_reference\paired_results\change_paired.csv'
)
output_csv = input_csv.with_name('change_metrics.csv')
expected_sample_count = 800
periods = ('previous', 'from', 'to', 'next')
result_names = [
    f'{strategy}_{stage}' for strategy in ('s1', 's2', 's3')
    for stage in ('before', 'gaussian', 'majority5', 'gaussian_majority5')
] + ['sequential_yft']


def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def macro_f1(reference, prediction):
    confusion = np.zeros((17, 17), dtype=int)
    np.add.at(confusion, (reference.ravel() - 1, prediction.ravel() - 1), 1)
    denominator = confusion.sum(0) + confusion.sum(1)
    return np.divide(2 * confusion.diagonal(), denominator,
                     out=np.zeros(17), where=denominator > 0).mean()


def calculate_metrics(samples):
    if (len(samples) != expected_sample_count
            or samples.sample_id.isna().any() or samples.sample_id.duplicated().any()):
        raise ValueError('配对表必须有 800 个不重复样点。')
    samples = samples.set_index('sample_id').sort_index()
    year_columns = [f'{p}_year' for p in periods]
    reference_columns = [f'reference_{p}_lcz' for p in periods]
    prediction_columns = [f'{name}_{p}_lcz' for name in result_names for p in periods]
    values = samples[reference_columns + prediction_columns].to_numpy(dtype=float)
    valid = np.isfinite(values) & (values >= 1) & (values <= 17) & (values == np.floor(values))
    if not valid.all() or not np.all(np.diff(samples[year_columns], axis=1) == 1):
        raise ValueError('类别无效或年份不连续，不能静默丢点。')
    reference = samples[['reference_from_lcz', 'reference_to_lcz']].to_numpy(dtype=int)
    rows = []
    for result_name in result_names:
        strategy = result_name.split('_')[0]
        baseline_name = f'{strategy}_before' if strategy in ('s1', 's2', 's3') else None
        prediction = samples[[f'{result_name}_from_lcz', f'{result_name}_to_lcz']].to_numpy(dtype=int)
        sequence = samples[[f'{result_name}_{p}_lcz' for p in periods]].to_numpy(dtype=int)
        changed = reference[:, 0] != reference[:, 1]
        detected = prediction[:, 0] != prediction[:, 1]
        both_correct = np.all(prediction == reference, axis=1)
        correct = changed & both_correct
        changed_count, stable_count = int(changed.sum()), int((~changed).sum())
        tp, fp = int((changed & detected).sum()), int((~changed & detected).sum())
        fn, tn = changed_count - tp, stable_count - fp
        row = dict(
            result=result_name, sample_count=len(samples),
            reference_stable_count=stable_count, reference_changed_count=changed_count,
            reference_change_rate=ratio(changed_count, len(samples)),
            predicted_changed_count=int(detected.sum()), predicted_change_rate=float(detected.mean()),
            tp=tp, fp=fp, fn=fn, tn=tn, correct_transition_count=int(correct.sum()),
            false_change_rate=ratio(fp, stable_count), reference_change_recall=ratio(tp, changed_count),
            correct_transition_rate=ratio(int(correct.sum()), changed_count),
            classification_observation_count=int(reference.size),
            classification_oa=float((prediction == reference).mean()),
            macro_f1_17classes=float(macro_f1(reference, prediction)),
            both_endpoint_accuracy=float(both_correct.mean()),
            change_precision=ratio(tp, tp + fp), change_f1=ratio(2 * tp, 2 * tp + fp + fn),
        )
        if baseline_name is not None:
            baseline_values = samples[
                [f'{baseline_name}_from_lcz', f'{baseline_name}_to_lcz']
            ].to_numpy(dtype=int)
            baseline_correct = changed & np.all(baseline_values == reference, axis=1)
            baseline_count = int(baseline_correct.sum())
            retained = int((baseline_correct & correct).sum())
            # 正负一年内唯一且类别正确的 A→B 用于核查变化时点。
            matching = ((sequence[:, :-1] == reference[:, 0, None])
                        & (sequence[:, 1:] == reference[:, 1, None]) & changed[:, None])
            matches = matching.sum(axis=1)
            unique = changed & (matches == 1)
            errors = np.argmax(matching[unique], axis=1) - 1
            retained_nearby = int((baseline_correct & unique).sum())
            row.update(
                baseline_correct_change_count=baseline_count,
                retained_correct_change_same_year_count=retained,
                correct_change_retention_same_year=ratio(retained, baseline_count),
                timing_matched_count=int(unique.sum()), timing_early_count=int((errors < 0).sum()),
                timing_on_time_count=int((errors == 0).sum()), timing_delayed_count=int((errors > 0).sum()),
                timing_missed_count=int((changed & (matches == 0)).sum()),
                timing_ambiguous_count=int((changed & (matches > 1)).sum()),
                change_year_bias=float(errors.mean()) if errors.size else np.nan,
                change_year_mae=float(np.abs(errors).mean()) if errors.size else np.nan,
                retained_correct_change_within_one_year_count=retained_nearby,
                correct_change_retention_within_one_year=ratio(retained_nearby, baseline_count),
                baseline_correct_change_missed_within_one_year_count=int(
                    (baseline_correct & (matches == 0)).sum()
                ),
            )
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    results = calculate_metrics(pd.read_csv(input_csv))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'已保存：{output_csv}')
