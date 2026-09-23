"""Validation-tuned linear downstream heads."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .exceptions import InputValidationError

DEFAULT_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_LOGISTIC_CS = (0.03, 0.1, 0.3, 1.0, 3.0)


@dataclass
class FittedLinearHead:
    """One frozen validation-selected linear head."""

    task: str
    pipeline: Pipeline
    selected_hyperparameter: dict[str, float]
    validation_grid: list[dict[str, float]]
    selection_metric: str
    working_dtype: str = "float64"

    def predict(self, features: np.ndarray) -> np.ndarray:
        dtype = np.float32 if self.working_dtype == "float32" else np.float64
        transformed = self.pipeline.named_steps["scale"].transform(features).astype(dtype)
        pca = self.pipeline.named_steps.get("pca")
        if pca is not None:
            transformed = pca.transform(transformed).astype(dtype)
        if self.task == "regression":
            return np.asarray(self.pipeline.named_steps["ridge"].predict(transformed), dtype=float)
        probability = np.asarray(
            self.pipeline.named_steps["logistic"].predict_proba(transformed), dtype=float
        )
        return probability / probability.sum(axis=1, keepdims=True)


def numeric_state_sha256(*values: np.ndarray) -> str:
    """Hash fitted numeric state without expanding large arrays into reports."""

    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def fitted_linear_head_audit(head: FittedLinearHead) -> dict[str, Any]:
    """Return a stable lock for training preprocessing and the fitted head."""

    scaler = head.pipeline.named_steps["scale"]
    model_name = "ridge" if head.task == "regression" else "logistic"
    model = head.pipeline.named_steps[model_name]
    pca = head.pipeline.named_steps.get("pca")
    state = [scaler.mean_, scaler.scale_]
    if pca is not None:
        state.extend([pca.mean_, pca.components_])
    return {
        "preprocessing_fit_partition": "train",
        "hyperparameter_selection_partition": "validation",
        "selected_hyperparameter": dict(head.selected_hyperparameter),
        "working_dtype": head.working_dtype,
        "stabilized_near_null_columns": int(
            np.sum(getattr(scaler, "lookagain_stabilized_columns_", np.asarray([], dtype=bool)))
        ),
        "training_preprocessing_sha256": numeric_state_sha256(*state),
        "pca": (
            {
                "applied": True,
                "fit_partition": "train",
                "input_dimension": int(pca.n_features_in_),
                "output_dimension": int(pca.n_components_),
                "svd_solver": str(pca.svd_solver),
                "iterated_power": int(pca.iterated_power),
                "random_state": pca.random_state,
            }
            if pca is not None
            else {"applied": False}
        ),
        "fitted_head_sha256": numeric_state_sha256(model.coef_, model.intercept_),
    }


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise InputValidationError(f"{name} must be a finite two-dimensional feature matrix.")
    return array


def _fit_numerically_stable_scaler(train: np.ndarray) -> StandardScaler:
    """Fit a train-only scaler without amplifying numerically null columns.

    Randomized PCA can return trailing directions whose training variance is at
    floating-point noise scale.  A later validation observation may project
    nontrivially onto such a direction; dividing by a roughly 1e-15 training
    standard deviation then creates meaningless 1e14-scale inputs.  Treat a
    standard deviation below sqrt(float64 epsilon), relative to the column's
    observed magnitude, as numerically null and leave that column unscaled.
    """

    scaler = StandardScaler().fit(train)
    magnitude = np.maximum(1.0, np.max(np.abs(np.asarray(train, dtype=float)), axis=0))
    floor = np.sqrt(np.finfo(np.float64).eps) * magnitude
    stabilized = np.asarray(scaler.scale_ < floor, dtype=bool)
    scaler.scale_[stabilized] = 1.0
    scaler.lookagain_scale_floor_ = floor
    scaler.lookagain_stabilized_columns_ = stabilized
    return scaler


def _fit_preprocessor(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    max_components: int | None,
    pca_iterated_power: int,
    random_state: int,
    working_dtype: str,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA | None]:
    if working_dtype not in {"float32", "float64"}:
        raise ValueError("working_dtype must be 'float32' or 'float64'.")
    dtype = np.float32 if working_dtype == "float32" else np.float64
    train = np.asarray(train, dtype=dtype)
    validation = np.asarray(validation, dtype=dtype)
    scaler = _fit_numerically_stable_scaler(train)
    transformed_train = scaler.transform(train).astype(dtype)
    transformed_validation = scaler.transform(validation).astype(dtype)
    pca = None
    if max_components is not None and train.shape[1] > max_components:
        component_count = min(max_components, train.shape[0] - 1, train.shape[1])
        if component_count < 1:
            raise InputValidationError("PCA requires at least two training observations.")
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=random_state,
            iterated_power=pca_iterated_power,
        ).fit(transformed_train)
        # The research workflow used fit followed by transform (not PCA.fit_transform).
        transformed_train = pca.transform(transformed_train).astype(dtype)
        transformed_validation = pca.transform(transformed_validation).astype(dtype)
    return transformed_train, transformed_validation, scaler, pca


def fit_ridge_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
    max_components: int | None = None,
    pca_iterated_power: int = 3,
    random_state: int = 20260818,
    working_dtype: str = "float64",
) -> FittedLinearHead:
    """Select ridge alpha using validation MSE; never accepts test outcomes."""

    train = _finite_matrix(x_train, "x_train")
    validation = _finite_matrix(x_validation, "x_validation")
    targets = np.asarray(y_train, dtype=float)
    validation_targets = np.asarray(y_validation, dtype=float)
    candidates: list[tuple[float, Ridge]] = []
    grid: list[dict[str, float]] = []
    if max_components is not None and max_components < 1:
        raise ValueError("max_components must be positive when supplied.")
    if pca_iterated_power < 0:
        raise ValueError("pca_iterated_power must be nonnegative.")
    transformed_train, transformed_validation, scaler, pca = _fit_preprocessor(
        train,
        validation,
        max_components=max_components,
        pca_iterated_power=pca_iterated_power,
        random_state=random_state,
        working_dtype=working_dtype,
    )
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        if alpha <= 0:
            raise ValueError("All ridge alphas must be positive.")
        model = Ridge(alpha=alpha).fit(transformed_train, targets)
        loss = float(mean_squared_error(validation_targets, model.predict(transformed_validation)))
        candidates.append((loss, model))
        grid.append({"alpha": alpha, "validation_mse": loss})
    if not candidates:
        raise ValueError("At least one ridge alpha is required.")
    best_index = min(range(len(candidates)), key=lambda index: candidates[index][0])
    steps: list[tuple[str, Any]] = [("scale", scaler)]
    if pca is not None:
        steps.append(("pca", pca))
    steps.append(("ridge", candidates[best_index][1]))
    return FittedLinearHead(
        task="regression",
        pipeline=Pipeline(steps),
        selected_hyperparameter={"alpha": grid[best_index]["alpha"]},
        validation_grid=grid,
        selection_metric="mse",
        working_dtype=working_dtype,
    )


def _class_weight(y: np.ndarray) -> str | None:
    _, counts = np.unique(y, return_counts=True)
    return "balanced" if counts.max() / counts.min() > 2.0 else None


def fit_logistic_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    c_values: Iterable[float] = DEFAULT_LOGISTIC_CS,
    random_state: int = 20260818,
    max_components: int | None = None,
    pca_iterated_power: int = 3,
    max_iter: int = 2000,
    working_dtype: str = "float64",
) -> FittedLinearHead:
    """Select regularized logistic C using validation log loss only."""

    train = _finite_matrix(x_train, "x_train")
    validation = _finite_matrix(x_validation, "x_validation")
    targets = np.asarray(y_train, dtype=int)
    validation_targets = np.asarray(y_validation, dtype=int)
    expected_classes = np.arange(len(np.unique(targets)))
    if not np.array_equal(np.unique(targets), expected_classes):
        raise InputValidationError(
            "Training class IDs must be contiguous from zero. Use LookAgain-ML's analyze() encoder."
        )
    if max_components is not None and max_components < 1:
        raise ValueError("max_components must be positive when supplied.")
    if pca_iterated_power < 0:
        raise ValueError("pca_iterated_power must be nonnegative.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    candidates: list[tuple[float, LogisticRegression]] = []
    grid: list[dict[str, float]] = []
    transformed_train, transformed_validation, scaler, pca = _fit_preprocessor(
        train,
        validation,
        max_components=max_components,
        pca_iterated_power=pca_iterated_power,
        random_state=random_state,
        working_dtype=working_dtype,
    )
    for raw_c in c_values:
        c_value = float(raw_c)
        if c_value <= 0:
            raise ValueError("All logistic C values must be positive.")
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=max_iter,
            class_weight=_class_weight(targets),
            random_state=random_state,
        )
        model.fit(transformed_train, targets)
        probability = np.asarray(model.predict_proba(transformed_validation), dtype=float)
        probability /= probability.sum(axis=1, keepdims=True)
        loss = float(log_loss(validation_targets, probability, labels=expected_classes))
        candidates.append((loss, model))
        grid.append({"C": c_value, "validation_log_loss": loss})
    if not candidates:
        raise ValueError("At least one logistic C value is required.")
    best_index = min(range(len(candidates)), key=lambda index: candidates[index][0])
    steps = [("scale", scaler)]
    if pca is not None:
        steps.append(("pca", pca))
    steps.append(("logistic", candidates[best_index][1]))
    return FittedLinearHead(
        task="classification",
        pipeline=Pipeline(steps),
        selected_hyperparameter={"C": grid[best_index]["C"]},
        validation_grid=grid,
        selection_metric="log_loss",
        working_dtype=working_dtype,
    )


def fit_linear_head(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    ridge_alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
    logistic_cs: Iterable[float] = DEFAULT_LOGISTIC_CS,
    random_state: int = 20260818,
    max_components: int | None = None,
    pca_iterated_power: int = 3,
    logistic_max_iter: int = 2000,
    working_dtype: str = "float64",
) -> FittedLinearHead:
    if task == "regression":
        return fit_ridge_head(
            x_train,
            y_train,
            x_validation,
            y_validation,
            alphas=ridge_alphas,
            max_components=max_components,
            pca_iterated_power=pca_iterated_power,
            random_state=random_state,
            working_dtype=working_dtype,
        )
    if task == "classification":
        return fit_logistic_head(
            x_train,
            y_train,
            x_validation,
            y_validation,
            c_values=logistic_cs,
            random_state=random_state,
            max_components=max_components,
            pca_iterated_power=pca_iterated_power,
            max_iter=logistic_max_iter,
            working_dtype=working_dtype,
        )
    raise ValueError("task must be 'regression' or 'classification'.")
