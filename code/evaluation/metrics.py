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
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    class_labels = list(range(num_classes))

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
    fold_dataframe = pd.DataFrame(fold_metrics)
    metric_columns = ["oa", "aa", "kappa", "macro_f1"]
    missing_columns = [column for column in metric_columns if column not in fold_dataframe.columns]

    summary_records = []
    for metric_name in metric_columns:
        summary_records.append({
            "metric": metric_name,
            "mean": fold_dataframe[metric_name].mean(),
            "std": fold_dataframe[metric_name].std(ddof=1),
        })
    return fold_dataframe, pd.DataFrame(summary_records)


def save_metric_outputs(output_dir, fold_metrics, class_metrics_by_fold, confusion_matrices):
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
    matrix_path = Path(matrix_path)
    output_path = matrix_path.parent / 'confusion_matrix.png'
    
def save_scores(result, output, scope, extra=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loss, metrics, classes, matrix, predictions, truth = result
    record = dict(metrics, loss=loss, evaluation_scope=scope)
    record.update(extra or {})
    if (output / 'metrics.csv').exists():
        previous = pd.read_csv(output / 'metrics.csv').iloc[0]
        
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

