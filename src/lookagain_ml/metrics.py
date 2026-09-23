"""Task-appropriate metric registry and implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    task: str
    higher_is_better: bool
    description: str


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "r2": MetricSpec("r2", "regression", True, "Coefficient of determination"),
    "mse": MetricSpec("mse", "regression", False, "Mean squared error"),
    "mae": MetricSpec("mae", "regression", False, "Mean absolute error"),
    "correlation": MetricSpec("correlation", "regression", True, "Pearson correlation"),
    "calibration_slope": MetricSpec(
        "calibration_slope", "regression", False, "Slope from outcome on prediction"
    ),
    "accuracy": MetricSpec("accuracy", "classification", True, "Accuracy"),
    "balanced_accuracy": MetricSpec(
        "balanced_accuracy", "classification", True, "Class-balanced accuracy"
    ),
    "macro_f1": MetricSpec("macro_f1", "classification", True, "Macro-averaged F1"),
    "log_loss": MetricSpec("log_loss", "classification", False, "Multiclass log loss"),
    "roc_auc": MetricSpec("roc_auc", "classification", True, "Binary ROC AUC"),
    "ece": MetricSpec("ece", "classification", False, "Expected calibration error"),
    "brier": MetricSpec("brier", "classification", False, "Binary Brier score"),
}


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == y_true).astype(float)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (low, high) in enumerate(pairwise(edges)):
        selected = (confidence > low) & (
            confidence <= high if index == bins - 1 else confidence < high
        )
        if selected.any():
            value += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(value)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_value = np.asarray(y_true, dtype=float)
    pred_value = np.asarray(prediction, dtype=float)
    if not np.isfinite(y_value).all() or not np.isfinite(pred_value).all():
        raise ValueError("Regression metrics require finite outcomes and predictions.")
    if np.std(y_value) > 1e-12 and np.std(pred_value) > 1e-12:
        correlation = float(np.corrcoef(y_value, pred_value)[0, 1])
    else:
        correlation = 0.0
    if np.var(pred_value) > 1e-12:
        calibration_slope = float(
            np.cov(pred_value, y_value, ddof=0)[0, 1] / np.var(pred_value)
        )
        calibration_intercept = float(
            y_value.mean() - calibration_slope * pred_value.mean()
        )
    else:
        calibration_slope = 0.0
        calibration_intercept = float(y_value.mean())
    return {
        "r2": float(r2_score(y_value, pred_value)),
        "mse": float(mean_squared_error(y_value, pred_value)),
        "mae": float(mean_absolute_error(y_value, pred_value)),
        "correlation": correlation,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    y_value = np.asarray(y_true, dtype=int)
    probability_value = np.asarray(probabilities, dtype=float)
    if probability_value.ndim != 2 or probability_value.shape[0] != len(y_value):
        raise ValueError("Classification probabilities must have shape (observations, classes).")
    if not np.isfinite(probability_value).all() or (probability_value < 0).any():
        raise ValueError("Classification probabilities must be finite and nonnegative.")
    if not np.allclose(probability_value.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Classification probabilities must sum to one for every observation.")
    predictions = probability_value.argmax(axis=1)
    class_count = probability_value.shape[1]
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_value, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_value, predictions)),
        "macro_f1": float(f1_score(y_value, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_value, probability_value, labels=np.arange(class_count))),
        "ece": expected_calibration_error(y_value, probability_value),
    }
    if class_count == 2:
        if len(np.unique(y_value)) == 2:
            result["roc_auc"] = float(roc_auc_score(y_value, probability_value[:, 1]))
        else:
            result["roc_auc"] = None
        result["brier"] = float(brier_score_loss(y_value, probability_value[:, 1]))
    return result


MetricFunction = Callable[..., dict[str, float | None]]


def metric_function(task: str) -> MetricFunction:
    if task == "regression":
        return regression_metrics
    if task == "classification":
        return classification_metrics
    raise ValueError("task must be 'regression' or 'classification'.")


