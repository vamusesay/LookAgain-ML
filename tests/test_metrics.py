import numpy as np

from lookagain_ml.metrics import (
    METRIC_REGISTRY,
    classification_metrics,
    regression_metrics,
)


def test_regression_metrics_include_required_outputs():
    values = regression_metrics(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([0.1, 0.9, 2.1, 2.9]),
    )
    assert {"r2", "mse", "mae", "correlation", "calibration_slope"} <= set(values)
    assert values["r2"] > 0.9


def test_binary_classification_metrics_are_probability_aware():
    values = classification_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
    )
    assert values["accuracy"] == 1.0
    assert values["roc_auc"] == 1.0
    assert {"balanced_accuracy", "macro_f1", "log_loss", "ece", "brier"} <= set(values)
    assert METRIC_REGISTRY["r2"].task == "regression"
    assert METRIC_REGISTRY["log_loss"].task == "classification"


def test_multiclass_metrics_never_report_r2():
    values = classification_metrics(
        np.asarray([0, 1, 2]),
        np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    )
    assert "r2" not in values
    assert "roc_auc" not in values



