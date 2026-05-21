from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correctness = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if not np.any(in_bin):
            continue
        prop = float(np.mean(in_bin))
        acc = float(np.mean(correctness[in_bin]))
        conf = float(np.mean(confidence[in_bin]))
        ece += prop * abs(acc - conf)
    return ece


def binary_classification_metrics(
    y_true: list[int], y_prob: list[float], threshold: float = 0.5
) -> dict[str, float | int | None]:
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob)
    y_pred_array = (y_prob_array >= threshold).astype(int)
    has_two_classes = len(np.unique(y_true_array)) == 2

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true_array, y_pred_array)) if has_two_classes else None
        ),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ece": expected_calibration_error(y_true_array, y_prob_array),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if has_two_classes:
        metrics["auroc"] = float(roc_auc_score(y_true_array, y_prob_array))
        metrics["auprc"] = float(average_precision_score(y_true_array, y_prob_array))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    return metrics


def candidate_thresholds(y_prob: list[float]) -> np.ndarray:
    unique_probs = np.unique(np.asarray(y_prob, dtype=float))
    thresholds = np.concatenate(([0.0], unique_probs, [1.0]))
    return np.clip(thresholds, 0.0, 1.0)


def select_thresholds(
    y_true: list[int],
    y_prob: list[float],
    min_sensitivity: float = 0.90,
) -> dict[str, dict[str, float | int | None]]:
    selections: dict[str, tuple[float, dict[str, float | int | None]]] = {}
    for threshold in candidate_thresholds(y_prob):
        metrics = binary_classification_metrics(y_true, y_prob, threshold=float(threshold))
        current = selections.get("balanced_accuracy")
        if current is None or float(metrics["balanced_accuracy"] or -1.0) > float(
            current[1]["balanced_accuracy"] or -1.0
        ):
            selections["balanced_accuracy"] = (float(threshold), metrics)

        current = selections.get("f1")
        if current is None or float(metrics["f1"] or -1.0) > float(current[1]["f1"] or -1.0):
            selections["f1"] = (float(threshold), metrics)

        youden = float(metrics["recall_sensitivity"]) + float(metrics["specificity"]) - 1.0
        current = selections.get("youden")
        current_youden = (
            float(current[1]["recall_sensitivity"]) + float(current[1]["specificity"]) - 1.0
            if current is not None
            else -1.0
        )
        if current is None or youden > current_youden:
            selections["youden"] = (float(threshold), metrics)

        if float(metrics["recall_sensitivity"]) >= min_sensitivity:
            current = selections.get("sensitivity_90")
            if current is None or float(metrics["specificity"]) > float(current[1]["specificity"]):
                selections["sensitivity_90"] = (float(threshold), metrics)

    if "sensitivity_90" not in selections:
        selections["sensitivity_90"] = selections["balanced_accuracy"]

    return {
        name: {"selected_threshold": threshold, **metrics}
        for name, (threshold, metrics) in selections.items()
    }
