"""
文件作用：统一计算 LCZ 分类指标，并保存折级、汇总级和逐类别精度结果。
流程位置：由所有验证、独立测试和空间五折交叉验证入口共同调用。
主要输入：模型预测的 0--16 类别编码、真实标签、类别数量和输出目录。
主要输出：OA、AA、Kappa、Macro-F1；逐类别指标、混淆矩阵NPY/CSV及旧版样式PNG。
重要参数：num_classes 必须与标签编码一致；zero_division 固定为 0，避免缺失类别时虚高。
运行环境：Python 3.10、numpy、pandas、scikit-learn；自动绘图还需Matplotlib、Seaborn。
前后依赖：训练脚本生成预测后调用本模块；论文表格和图件从本模块输出的 CSV 汇总。
运行方式：不直接运行，由 python -m sf_ganet.pretrain 或 experiments 下的入口调用。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def calculate_metrics(predictions, labels, num_classes):
    """计算主指标、逐类别指标和固定类别顺序的混淆矩阵。"""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    class_labels = list(range(num_classes))

    if predictions.shape != labels.shape:
        raise ValueError(f"预测与标签形状不一致：{predictions.shape} 与 {labels.shape}")
    if predictions.size == 0:
        raise ValueError("无法对空预测数组计算精度。")

    report = classification_report(
        labels,
        predictions,
        labels=class_labels,
        target_names=[str(class_id) for class_id in class_labels],
        output_dict=True,
        zero_division=0,
    )
    class_records = []
    for class_id in class_labels:
        class_report = report[str(class_id)]
        class_records.append({
            "class_id": class_id,
            "precision": class_report["precision"],
            "recall": class_report["recall"],
            "f1": class_report["f1-score"],
            "support": int(class_report["support"]),
        })

    metrics = {
        "oa": accuracy_score(labels, predictions),
        "aa": recall_score(labels, predictions, labels=class_labels, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(labels, predictions, labels=class_labels),
        "macro_f1": f1_score(labels, predictions, labels=class_labels, average="macro", zero_division=0),
    }
    class_metrics = pd.DataFrame(class_records)
    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    return metrics, class_metrics, matrix


def summarize_fold_metrics(fold_metrics):
    """按折叠记录生成每项主指标的均值与标准差。"""
    fold_dataframe = pd.DataFrame(fold_metrics)
    metric_columns = ["oa", "aa", "kappa", "macro_f1"]
    missing_columns = [column for column in metric_columns if column not in fold_dataframe.columns]
    if missing_columns:
        raise ValueError(f"折叠指标缺少字段：{missing_columns}")

    summary_records = []
    for metric_name in metric_columns:
        summary_records.append({
            "metric": metric_name,
            "mean": fold_dataframe[metric_name].mean(),
            "std": fold_dataframe[metric_name].std(ddof=1),
        })
    return fold_dataframe, pd.DataFrame(summary_records)


def save_metric_outputs(output_dir, fold_metrics, class_metrics_by_fold, confusion_matrices):
    """将空间折叠评价结果保存为可直接制表的 CSV 和 NPY 文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dataframe, summary_dataframe = summarize_fold_metrics(fold_metrics)
    fold_dataframe.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_dataframe.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    class_frames = []
    for fold_index, class_metrics in enumerate(class_metrics_by_fold, start=1):
        current_metrics = class_metrics.copy()
        current_metrics.insert(0, "fold", fold_index)
        class_frames.append(current_metrics)
    if class_frames:
        class_dataframe = pd.concat(class_frames, ignore_index=True)
        class_dataframe.to_csv(output_dir / "class_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
        class_summary = class_dataframe.groupby(
            "class_id",
            as_index=False,
        )[["precision", "recall", "f1", "support"]].agg(["mean", "std"])
        class_summary.to_csv(output_dir / "class_metrics_summary.csv", encoding="utf-8-sig")

    if confusion_matrices:
        np.save(output_dir / "confusion_matrices.npy", np.asarray(confusion_matrices))
        np.save(output_dir / "mean_confusion_matrix.npy", np.mean(confusion_matrices, axis=0))
        save_confusion_figure(output_dir / 'mean_confusion_matrix.npy')


def save_confusion_figure(matrix_path):
    # 复用旧实验绘图代码；已有图直接跳过，失败不丢弃已保存的指标或强制重训。
    matrix_path = Path(matrix_path)
    output_path = matrix_path.parent / 'confusion_matrix.png'
    if output_path.exists():
        print(f'混淆矩阵图已存在，保留：{output_path}', flush=True)
        return
    try:
        from visualization.plot_confusion_matrix import plot_matrix
        plot_matrix(matrix_path, output_path)
    except Exception as error:
        print(f'混淆矩阵数值已保存，但绘图失败：{error}。'
              f'可单独运行绘图脚本读取：{matrix_path}', flush=True)

def save_scores(result, output, scope, extra=None):
    # 验证与独立测试文件显式区分；预测使用内部0--16标签编码。
    # 续跑只补齐尚未写完的评价文件，不覆盖已保存的完整结果。
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loss, metrics, classes, matrix, predictions, truth = result
    record = dict(metrics, loss=loss, evaluation_scope=scope)
    record.update(extra or {})
    if (output / 'metrics.csv').exists():
        previous = pd.read_csv(output / 'metrics.csv').iloc[0]
        if previous['evaluation_scope'] != scope:
            raise ValueError('已有评价文件的用途不一致，拒绝覆盖。')
        for key in ['oa', 'aa', 'kappa', 'macro_f1']:
            if not np.isclose(previous[key], record[key], equal_nan=True):
                raise ValueError('恢复评价与已有分数不一致，拒绝混用结果。')
    else:
        pd.DataFrame([record]).to_csv(output / 'metrics.csv', index=False)
    if not (output / 'class_metrics.csv').exists():
        classes.to_csv(output / 'class_metrics.csv', index=False)
    if not (output / 'confusion_matrix.npy').exists():
        np.save(output / 'confusion_matrix.npy', matrix)
    if not (output / 'predictions.csv').exists():
        pd.DataFrame({'target': truth, 'prediction': predictions}).to_csv(
            output / 'predictions.csv', index=False,
        )
    save_confusion_figure(output / 'confusion_matrix.npy')
    return record
