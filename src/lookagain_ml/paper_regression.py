"""Reusable validation-locked regression helpers used by paper workflows.

These functions expose the exact estimator families needed by the updated
Images-as-Covariates replication without changing :func:`lookagain_ml.analyze`
defaults.  They accept no test outcomes.  Paper-specific menus, seed offsets,
and selection orchestration remain in replication configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .exceptions import InputValidationError
from .hashing import digest_strings
from .heads import numeric_state_sha256


def _finite_matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise InputValidationError(f"{name} must be a finite two-dimensional matrix.")
    return matrix


def _finite_outcome(values: Any, name: str) -> np.ndarray:
    outcome = np.asarray(values, dtype=float)
    if outcome.ndim != 1 or not np.isfinite(outcome).all():
        raise InputValidationError(f"{name} must be a finite one-dimensional outcome.")
    return outcome


@dataclass
class FittedValidationMLPRegressor:
    """One training-fitted scikit-learn MLP with validation diagnostics."""

    scaler: StandardScaler
    model: MLPRegressor
    y_mean: float
    y_scale: float
    validation_prediction: np.ndarray
    validation_mse: float
    configuration: dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = _finite_matrix(features, "features")
        raw = self.model.predict(self.scaler.transform(matrix))
        return np.asarray(raw * self.y_scale + self.y_mean, dtype=float)

    def audit(self) -> dict[str, Any]:
        state = [self.scaler.mean_, self.scaler.scale_]
        state.extend(self.model.coefs_)
        state.extend(self.model.intercepts_)
        return {
            "model": "sklearn_mlp_regressor",
            "fit_partition": "train",
            "validation_partition_role": "reported comparison only",
            "internal_early_stopping": True,
            "test_outcomes_used": False,
            "configuration": dict(self.configuration),
            "iterations": int(self.model.n_iter_),
            "validation_mse": float(self.validation_mse),
            "fitted_state_sha256": numeric_state_sha256(*state),
        }


def fit_validation_mlp_regressor(
    x_train: np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    x_validation: np.ndarray,
    y_validation: Sequence[float] | np.ndarray,
    *,
    hidden_layer_sizes: Sequence[int] = (512, 128),
    alpha: float = 1e-4,
    learning_rate_init: float = 5e-4,
    batch_size: int = 128,
    max_iter: int = 300,
    patience: int = 20,
    validation_fraction: float = 0.15,
    random_state: int,
) -> FittedValidationMLPRegressor:
    """Fit the source-defined ReLU/Adam image or tabular regression head.

    The authoritative sklearn backend does not implement dropout.  Callers
    must not describe an unused historical ``dropout`` configuration field as
    active regularization.
    """

    train = _finite_matrix(x_train, "x_train")
    validation = _finite_matrix(x_validation, "x_validation")
    y_fit = _finite_outcome(y_train, "y_train")
    y_tune = _finite_outcome(y_validation, "y_validation")
    if len(train) != len(y_fit) or len(validation) != len(y_tune):
        raise InputValidationError("MLP features and outcomes must align by partition.")
    hidden = tuple(int(value) for value in hidden_layer_sizes)
    if not hidden or any(value < 1 for value in hidden):
        raise InputValidationError("hidden_layer_sizes must contain positive integers.")
    if alpha < 0 or learning_rate_init <= 0 or batch_size < 1 or max_iter < 1 or patience < 1:
        raise InputValidationError("MLP regularization, learning, batch, iteration, and patience settings are invalid.")
    if not 0 < validation_fraction < 1:
        raise InputValidationError("validation_fraction must be strictly between zero and one.")

    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train).astype(np.float32)
    validation_scaled = scaler.transform(validation).astype(np.float32)
    y_mean = float(y_fit.mean())
    y_scale = float(y_fit.std()) or 1.0
    standardized = ((y_fit - y_mean) / y_scale).astype(np.float32)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        learning_rate_init=float(learning_rate_init),
        batch_size=int(batch_size),
        max_iter=int(max_iter),
        early_stopping=True,
        validation_fraction=float(validation_fraction),
        n_iter_no_change=int(patience),
        random_state=int(random_state),
    ).fit(train_scaled, standardized)
    validation_prediction = (
        model.predict(validation_scaled) * y_scale + y_mean
    )
    configuration = {
        "hidden_layer_sizes": list(hidden),
        "activation": "relu",
        "solver": "adam",
        "alpha": float(alpha),
        "learning_rate_init": float(learning_rate_init),
        "batch_size": int(batch_size),
        "max_iter": int(max_iter),
        "early_stopping": True,
        "validation_fraction": float(validation_fraction),
        "n_iter_no_change": int(patience),
        "random_state": int(random_state),
        "dropout_applied": False,
    }
    return FittedValidationMLPRegressor(
        scaler=scaler,
        model=model,
        y_mean=y_mean,
        y_scale=y_scale,
        validation_prediction=np.asarray(validation_prediction, dtype=float),
        validation_mse=float(mean_squared_error(y_tune, validation_prediction)),
        configuration=configuration,
    )


@dataclass
class FittedRefitRidgeRegressor:
    """Validation-selected ridge with a locked train+validation test refit."""

    selected_alpha: float
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    validation_grid: list[dict[str, float]]
    audit: dict[str, Any]


def fit_validation_refit_ridge_regressor(
    x_train: np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    x_validation: np.ndarray,
    y_validation: Sequence[float] | np.ndarray,
    x_test: np.ndarray,
    *,
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
) -> FittedRefitRidgeRegressor:
    """Select alpha on validation and refit on train+validation for test."""

    train = _finite_matrix(x_train, "x_train")
    validation = _finite_matrix(x_validation, "x_validation")
    test = _finite_matrix(x_test, "x_test")
    y_fit = _finite_outcome(y_train, "y_train")
    y_tune = _finite_outcome(y_validation, "y_validation")
    if train.shape[1] != validation.shape[1] or train.shape[1] != test.shape[1]:
        raise InputValidationError("Ridge feature columns must align across partitions.")
    scaler = StandardScaler().fit(train)
    z_train = scaler.transform(train)
    z_validation = scaler.transform(validation)
    candidates: list[tuple[float, float, Ridge, np.ndarray]] = []
    grid: list[dict[str, float]] = []
    for value in alphas:
        alpha = float(value)
        if alpha < 0 or not np.isfinite(alpha):
            raise InputValidationError("Ridge alphas must be finite and nonnegative.")
        model = Ridge(alpha=alpha).fit(z_train, y_fit)
        prediction = model.predict(z_validation)
        loss = float(mean_squared_error(y_tune, prediction))
        candidates.append((loss, alpha, model, prediction))
        grid.append({"alpha": alpha, "validation_mse": loss})
    if not candidates:
        raise InputValidationError("At least one ridge alpha is required.")
    loss, selected_alpha, _, validation_prediction = min(candidates, key=lambda item: item[0])
    combined = np.vstack([train, validation])
    combined_y = np.concatenate([y_fit, y_tune])
    refit_scaler = StandardScaler().fit(combined)
    refit = Ridge(alpha=selected_alpha).fit(refit_scaler.transform(combined), combined_y)
    test_prediction = refit.predict(refit_scaler.transform(test))
    return FittedRefitRidgeRegressor(
        selected_alpha=float(selected_alpha),
        validation_prediction=np.asarray(validation_prediction, dtype=float),
        test_prediction=np.asarray(test_prediction, dtype=float),
        validation_grid=grid,
        audit={
            "model": "validation_selected_ridge_with_locked_refit",
            "training_scaler_fit_partition": "train",
            "selection_partition": "validation",
            "test_refit_partition": "train+validation after alpha lock",
            "test_outcomes_used": False,
            "selected_alpha": float(selected_alpha),
            "validation_mse": float(loss),
            "training_state_sha256": numeric_state_sha256(scaler.mean_, scaler.scale_),
            "refit_state_sha256": numeric_state_sha256(
                refit_scaler.mean_, refit_scaler.scale_, refit.coef_, np.atleast_1d(refit.intercept_)
            ),
        },
    )


@dataclass
class FittedBlockPCA:
    """Train-fitted per-block scaling/PCA plus aligned transformed matrices."""

    training: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    audit: dict[str, Any]


def fit_train_only_block_pca(
    training_blocks: Sequence[np.ndarray],
    validation_blocks: Sequence[np.ndarray],
    test_blocks: Sequence[np.ndarray],
    *,
    max_components: int = 32,
    random_state: int = 7001,
) -> FittedBlockPCA:
    """Scale and reduce each block using training rows only."""

    if not (
        len(training_blocks) == len(validation_blocks) == len(test_blocks)
        and len(training_blocks) > 0
    ):
        raise InputValidationError("Block lists must be nonempty and have matching lengths.")
    if max_components < 1:
        raise InputValidationError("max_components must be positive.")
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for index, (raw_train, raw_validation, raw_test) in enumerate(
        zip(training_blocks, validation_blocks, test_blocks)
    ):
        train = _finite_matrix(raw_train, f"training_blocks[{index}]")
        validation = _finite_matrix(raw_validation, f"validation_blocks[{index}]")
        test = _finite_matrix(raw_test, f"test_blocks[{index}]")
        if train.shape[1] != validation.shape[1] or train.shape[1] != test.shape[1]:
            raise InputValidationError(f"Block {index} columns do not align.")
        scaler = StandardScaler().fit(train)
        z_train = scaler.transform(train)
        z_validation = scaler.transform(validation)
        z_test = scaler.transform(test)
        components = min(max_components, train.shape[1], train.shape[0] - 1)
        if components < 1:
            raise InputValidationError("Block PCA requires at least two training rows.")
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=int(random_state),
        ).fit(z_train)
        train_parts.append(pca.transform(z_train).astype(np.float32))
        validation_parts.append(pca.transform(z_validation).astype(np.float32))
        test_parts.append(pca.transform(z_test).astype(np.float32))
        records.append(
            {
                "block_index": index,
                "input_dimension": int(train.shape[1]),
                "output_dimension": int(components),
                "scaler_fit_partition": "train",
                "pca_fit_partition": "train",
                "random_state": int(random_state),
                "state_sha256": numeric_state_sha256(
                    scaler.mean_, scaler.scale_, pca.mean_, pca.components_
                ),
            }
        )
    return FittedBlockPCA(
        training=np.column_stack(train_parts).astype(np.float32),
        validation=np.column_stack(validation_parts).astype(np.float32),
        test=np.column_stack(test_parts).astype(np.float32),
        audit={
            "method": "train_only_blockwise_pca",
            "max_components_per_block": int(max_components),
            "blocks": records,
            "test_outcomes_used": False,
        },
    )


def _fit_meta_nn_ensemble(
    features: np.ndarray,
    outcome: np.ndarray,
    *,
    seeds: tuple[int, ...],
) -> list[tuple[MLPRegressor, StandardScaler, float, float]]:
    fitted: list[tuple[MLPRegressor, StandardScaler, float, float]] = []
    for seed in seeds:
        scaler = StandardScaler().fit(features)
        y_mean = float(outcome.mean())
        y_scale = float(outcome.std()) or 1.0
        model = MLPRegressor(
            hidden_layer_sizes=(8, 4),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.003,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=int(seed),
        ).fit(scaler.transform(features), (outcome - y_mean) / y_scale)
        fitted.append((model, scaler, y_mean, y_scale))
    return fitted


def _predict_meta_nn(
    fitted: list[tuple[MLPRegressor, StandardScaler, float, float]],
    features: np.ndarray,
) -> np.ndarray:
    return np.mean(
        [
            model.predict(scaler.transform(features)) * y_scale + y_mean
            for model, scaler, y_mean, y_scale in fitted
        ],
        axis=0,
    )


@dataclass
class FittedGroupCrossFittedRegressor:
    """Group-cross-fitted validation predictions and a full-validation model."""

    method: str
    oof_prediction: np.ndarray
    fitted: Any
    fold_assignments: np.ndarray
    audit: dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = _finite_matrix(features, "features")
        if self.method == "linear":
            return np.asarray(self.fitted.predict(matrix), dtype=float)
        return np.asarray(_predict_meta_nn(self.fitted, matrix), dtype=float)


def fit_group_cross_fitted_regression_stack(
    features: np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    method: str = "linear",
    folds: int = 5,
    fold_random_state: int = 20260827,
    nn_seeds: tuple[int, ...] = (101, 202, 303),
) -> FittedGroupCrossFittedRegressor:
    """Fit a cross-fitted validation meta-model using group-held-out predictions."""

    matrix = _finite_matrix(features, "features")
    y = _finite_outcome(outcome, "outcome")
    group_values = np.asarray(groups, dtype=object)
    if len(matrix) != len(y) or len(group_values) != len(y):
        raise InputValidationError("Meta features, outcomes, and groups must align.")
    if method not in {"linear", "nn"}:
        raise InputValidationError("method must be 'linear' or 'nn'.")
    unique = np.unique(group_values.astype(str))
    if folds < 2 or len(unique) < folds:
        raise InputValidationError("Cross-fitting requires at least two folds and one group per fold.")
    shuffled = unique.copy()
    np.random.default_rng(int(fold_random_state)).shuffle(shuffled)
    fold_map = {group: index % int(folds) for index, group in enumerate(shuffled)}
    assignments = np.asarray([fold_map[str(group)] for group in group_values], dtype=int)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_records: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        fit_rows = np.flatnonzero(assignments != fold)
        hold_rows = np.flatnonzero(assignments == fold)
        fit_groups = set(group_values[fit_rows].astype(str))
        hold_groups = set(group_values[hold_rows].astype(str))
        overlap = fit_groups & hold_groups
        if overlap:
            raise InputValidationError("A group crossed an out-of-fold meta split.")
        if method == "linear":
            local = LinearRegression().fit(matrix[fit_rows], y[fit_rows])
            oof[hold_rows] = local.predict(matrix[hold_rows])
        else:
            local = _fit_meta_nn_ensemble(matrix[fit_rows], y[fit_rows], seeds=nn_seeds)
            oof[hold_rows] = _predict_meta_nn(local, matrix[hold_rows])
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": len(fit_rows),
                "hold_rows": len(hold_rows),
                "fit_groups": len(fit_groups),
                "hold_groups": len(hold_groups),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("Cross-fitted meta predictions are incomplete.")
    fitted: Any
    if method == "linear":
        fitted = LinearRegression().fit(matrix, y)
        fitted_state = numeric_state_sha256(fitted.coef_, np.atleast_1d(fitted.intercept_))
    else:
        fitted = _fit_meta_nn_ensemble(matrix, y, seeds=nn_seeds)
        state: list[np.ndarray] = []
        for model, scaler, _, _ in fitted:
            state.extend([scaler.mean_, scaler.scale_, *model.coefs_, *model.intercepts_])
        fitted_state = numeric_state_sha256(*state)
    return FittedGroupCrossFittedRegressor(
        method=method,
        oof_prediction=oof,
        fitted=fitted,
        fold_assignments=assignments,
        audit={
            "method": f"group_cross_fitted_{method}_regression_stack",
            "fit_partition": "validation inner folds; final model uses all validation rows",
            "folds": int(folds),
            "fold_random_state": int(fold_random_state),
            "fold_assignment_sha256": digest_strings(assignments.astype(str).tolist()),
            "fold_records": fold_records,
            "nn_seeds": list(nn_seeds) if method == "nn" else None,
            "each_oof_row_excluded_from_its_model": True,
            "groups_disjoint_within_each_fold": True,
            "test_outcomes_used": False,
            "fitted_state_sha256": fitted_state,
        },
    )
