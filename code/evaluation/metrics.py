from pathlib import Path

import matplotlib.pyplot as plt
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
        current = report[str(class_id)]
        class_records.append(
            {
                "class_id": class_id,
                "precision": current["precision"],
                "recall": current["recall"],
                "f1": current["f1-score"],
                "support": int(current["support"]),
            }
        )

    metrics = {
        "oa": accuracy_score(labels, predictions),
        "aa": recall_score(
            labels,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
        "kappa": cohen_kappa_score(
            labels,
            predictions,
            labels=class_labels,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=class_labels,
    )
    return metrics, pd.DataFrame(class_records), matrix


def summarize_fold_metrics(fold_metrics):
    table = pd.DataFrame(fold_metrics)
    metric_columns = ["oa", "aa", "kappa", "macro_f1"]

    missing = [
        name
        for name in metric_columns
        if name not in table.columns
    ]
    if missing:
        raise ValueError(
            f"Missing metric columns: {missing}"
        )

    summary = pd.DataFrame(
        [
            {
                "metric": name,
                "mean": table[name].mean(),
                "std": table[name].std(ddof=1),
            }
            for name in metric_columns
        ]
    )
    return table, summary


def save_confusion_figure(matrix_path):
    matrix_path = Path(matrix_path)
    matrix = np.load(matrix_path)
    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(matrix)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Reference class")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(
        matrix_path.parent / "confusion_matrix.png",
        dpi=300,
    )
    plt.close(figure)


def save_metric_outputs(
    output_dir,
    fold_metrics,
    class_metrics_by_fold,
    confusion_matrices,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_table, summary = summarize_fold_metrics(fold_metrics)
    fold_table.to_csv(
        output_dir / "fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
    )

    class_frames = []
    for fold_index, class_metrics in enumerate(
        class_metrics_by_fold,
        start=1,
    ):
        current = class_metrics.copy()
        current.insert(0, "fold", fold_index)
        class_frames.append(current)

    if class_frames:
        class_table = pd.concat(
            class_frames,
            ignore_index=True,
        )
        class_table.to_csv(
            output_dir / "class_metrics_by_fold.csv",
            index=False,
        )

    if confusion_matrices:
        matrices = np.asarray(confusion_matrices)
        np.save(
            output_dir / "confusion_matrices.npy",
            matrices,
        )
        np.save(
            output_dir / "mean_confusion_matrix.npy",
            matrices.mean(axis=0),
        )
        save_confusion_figure(
            output_dir / "mean_confusion_matrix.npy"
        )


def save_scores(result, output, scope, extra=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    loss, metrics, classes, matrix, predictions, truth = result
    record = dict(
        metrics,
        loss=loss,
        evaluation_scope=scope,
    )
    record.update(extra or {})

    pd.DataFrame([record]).to_csv(
        output / "metrics.csv",
        index=False,
    )
    classes.to_csv(
        output / "class_metrics.csv",
        index=False,
    )
    np.save(
        output / "confusion_matrix.npy",
        matrix,
    )
    pd.DataFrame(
        {
            "target": truth,
            "prediction": predictions,
        }
    ).to_csv(
        output / "predictions.csv",
        index=False,
    )
    save_confusion_figure(
        output / "confusion_matrix.npy"
    )
    return record
