"""Paired uncertainty on an already locked test set."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .metrics import METRIC_REGISTRY, metric_function


def paired_locked_test_bootstrap(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    task: str,
    comparisons: Iterable[tuple[str, str]],
    metrics: Iterable[str],
    repetitions: int = 2000,
    random_state: int = 20260858,
) -> dict[str, Any]:
    """Use identical group-resampled locked-test rows for every fitted prediction."""

    if repetitions < 100:
        raise ValueError("bootstrap_repetitions must be at least 100 when enabled.")
    y_value = np.asarray(y_test)
    group_values = np.asarray(groups, dtype=object)
    if len(y_value) != len(group_values):
        raise ValueError("y_test and groups must have the same number of rows.")
    prediction_values = {name: np.asarray(value) for name, value in predictions.items()}
    if not prediction_values:
        raise ValueError("At least one locked-test prediction is required.")
    for name, value in prediction_values.items():
        if len(value) != len(y_value):
            raise ValueError(f"Prediction {name!r} is not row-aligned with y_test.")
    metric_names = list(metrics)
    allowed = {name for name, spec in METRIC_REGISTRY.items() if spec.task == task}
    unknown = set(metric_names) - allowed
    if unknown:
        raise ValueError(f"Unsupported {task} bootstrap metrics: {sorted(unknown)}.")
    pairs = list(comparisons)
    missing = {
        name for pair in pairs for name in pair if name not in prediction_values
    }
    if missing:
        raise ValueError(f"Bootstrap comparisons reference missing methods: {sorted(missing)}.")
    unique = np.unique(group_values)
    rows = {value: np.flatnonzero(group_values == value) for value in unique}
    compute = metric_function(task)
    point_metrics = {
        name: compute(y_value, prediction)
        for name, prediction in prediction_values.items()
    }
    draws = {
        name: {metric: [] for metric in metric_names} for name in prediction_values
    }
    pair_lookup = {f"{left}_minus_{right}": (left, right) for left, right in pairs}
    differences = {
        f"{left}_minus_{right}": {metric: [] for metric in metric_names}
        for left, right in pairs
    }
    rng = np.random.default_rng(random_state)
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows[value] for value in sampled])
        observed = {
            name: compute(y_value[indices], prediction[indices])
            for name, prediction in prediction_values.items()
        }
        for name in prediction_values:
            for metric in metric_names:
                value = observed[name].get(metric)
                if value is not None and np.isfinite(value):
                    draws[name][metric].append(float(value))
        for left, right in pairs:
            key = f"{left}_minus_{right}"
            for metric in metric_names:
                left_value = observed[left].get(metric)
                right_value = observed[right].get(metric)
                if (
                    left_value is not None
                    and right_value is not None
                    and np.isfinite(left_value)
                    and np.isfinite(right_value)
                ):
                    differences[key][metric].append(float(left_value - right_value))

    def summarize(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "lower_95": float(np.percentile(array, 2.5)),
            "upper_95": float(np.percentile(array, 97.5)),
            "positive_share": float(np.mean(array > 0)),
        }

    return {
        "method": "paired grouped bootstrap on locked-test predictions",
        "repetitions": repetitions,
        "resampling_unit": "effective leakage-safe group",
        "group_count": len(unique),
        "same_sample_for_all_methods": True,
        "conditioning_statement": (
            "Intervals condition on the fitted models and locked split; they do not include "
            "retraining, repartitioning, or encoder-pretraining uncertainty."
        ),
        "model_intervals": {
            name: {
                metric: {
                    **(summarize(values) or {}),
                    "point_estimate": point_metrics[name].get(metric),
                }
                for metric, values in by_metric.items()
            }
            for name, by_metric in draws.items()
        },
        "difference_intervals": {
            pair: {
                metric: {
                    **(summarize(values) or {}),
                    "point_estimate": (
                        float(point_metrics[pair_lookup[pair][0]][metric])
                        - float(point_metrics[pair_lookup[pair][1]][metric])
                        if point_metrics[pair_lookup[pair][0]].get(metric) is not None
                        and point_metrics[pair_lookup[pair][1]].get(metric) is not None
                        else None
                    ),
                    "higher_is_better": METRIC_REGISTRY[metric].higher_is_better,
                }
                for metric, values in by_metric.items()
            }
            for pair, by_metric in differences.items()
        },
    }
