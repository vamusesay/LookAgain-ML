"""Leakage-safe cross-fitted random-forest score integration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .exceptions import InputValidationError
from .structured import DEFAULT_RF_CONFIGURATIONS, DEFAULT_RF_SEEDS


@dataclass
class RFScoreResult:
    """Predictions and complete decision audit for one locked RF-score fit."""

    training_prediction: np.ndarray
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    train_oof_image_score: np.ndarray
    validation_image_score: np.ndarray
    test_image_score: np.ndarray
    selected_image_alpha: float
    selected_rf_configuration: dict[str, Any]
    validation_grid: list[dict[str, Any]]
    audit: dict[str, Any]


def _reduce_image_block(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    *,
    max_components: int,
    random_state: int,
    working_dtype: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    StandardScaler,
    PCA | None,
]:
    dtype = np.float32 if working_dtype == "float32" else np.float64
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train).astype(dtype)
    validation_scaled = scaler.transform(validation).astype(dtype)
    test_scaled = scaler.transform(test).astype(dtype)
    components = min(max_components, train.shape[1], train.shape[0] - 1)
    pca = None
    if components < train.shape[1]:
        pca = PCA(n_components=components, svd_solver="randomized", random_state=random_state)
        pca.fit(train_scaled)
        train_scaled = pca.transform(train_scaled).astype(dtype)
        validation_scaled = pca.transform(validation_scaled).astype(dtype)
        test_scaled = pca.transform(test_scaled).astype(dtype)
    return train_scaled, validation_scaled, test_scaled, {
        "scaler_fit_partition": "train",
        "pca_fit_partition": "train" if pca is not None else "not applied",
        "input_dimension": int(train.shape[1]),
        "output_dimension": int(train_scaled.shape[1]),
    }, scaler, pca


def _group_folds(groups: np.ndarray, folds: int, random_state: int) -> np.ndarray:
    unique = np.unique(groups.astype(str))
    if len(unique) < folds:
        raise InputValidationError(f"RF-score OOF generation requires at least {folds} training groups.")
    shuffled = unique.copy()
    np.random.default_rng(random_state).shuffle(shuffled)
    mapping = {group: index % folds for index, group in enumerate(shuffled)}
    return np.asarray([mapping[str(group)] for group in groups], dtype=int)


def fit_rf_score(
    structured_train: np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    groups_train: Sequence[Any] | np.ndarray,
    image_train: np.ndarray,
    structured_validation: np.ndarray,
    y_validation: Sequence[float] | np.ndarray,
    image_validation: np.ndarray,
    structured_test: np.ndarray,
    image_test: np.ndarray,
    *,
    ridge_alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    rf_configurations: Iterable[dict[str, Any]] = DEFAULT_RF_CONFIGURATIONS,
    rf_seeds: tuple[int, ...] = DEFAULT_RF_SEEDS,
    n_estimators: int = 500,
    oof_folds: int = 5,
    image_max_components: int = 32,
    random_state: int = 20260821,
    reduction_random_state: int | None = None,
    fold_random_state: int | None = None,
    standardize_reduced_scores: bool = False,
    refit_image_reduction: bool = True,
    working_dtype: str = "float64",
    n_jobs: int | None = -1,
) -> RFScoreResult:
    """Fit source-defined regression RF-score without accepting test outcomes.

    Training RF rows receive group-safe out-of-fold image scores. Validation
    scores come from a training-only image model. After validation locks ridge
    alpha and forest configuration, test scores and the test RF are refitted on
    train+validation. This function cannot inspect test outcomes by design.
    """

    if working_dtype not in {"float32", "float64"}:
        raise InputValidationError("working_dtype must be 'float32' or 'float64'.")
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1
    ):
        raise InputValidationError("n_jobs must be None, -1, or a positive integer.")
    dtype = np.float32 if working_dtype == "float32" else np.float64
    x_train = np.asarray(structured_train, dtype=dtype)
    x_validation = np.asarray(structured_validation, dtype=dtype)
    x_test = np.asarray(structured_test, dtype=dtype)
    z_train = np.asarray(image_train, dtype=dtype)
    z_validation = np.asarray(image_validation, dtype=dtype)
    z_test = np.asarray(image_test, dtype=dtype)
    y_train_array = np.asarray(y_train, dtype=float)
    y_validation_array = np.asarray(y_validation, dtype=float)
    groups = np.asarray(groups_train, dtype=object)
    matrices = (x_train, x_validation, x_test, z_train, z_validation, z_test)
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in matrices):
        raise InputValidationError("RF-score inputs must be finite two-dimensional matrices.")
    if x_train.shape[1] != x_validation.shape[1] or x_train.shape[1] != x_test.shape[1]:
        raise InputValidationError("Structured RF-score columns must align across partitions.")
    if z_train.shape[1] != z_validation.shape[1] or z_train.shape[1] != z_test.shape[1]:
        raise InputValidationError("Image RF-score columns must align across partitions.")
    if len(y_train_array) != len(x_train) or len(groups) != len(x_train) or len(y_validation_array) != len(x_validation):
        raise InputValidationError("RF-score outcomes/groups must align with their partitions.")
    alphas = [float(value) for value in ridge_alphas]
    if not alphas or any(value < 0 or not np.isfinite(value) for value in alphas):
        raise InputValidationError("ridge_alphas must contain finite nonnegative values.")
    if not isinstance(standardize_reduced_scores, bool) or not isinstance(refit_image_reduction, bool):
        raise InputValidationError(
            "standardize_reduced_scores and refit_image_reduction must be booleans."
        )
    reduction_seed = int(random_state if reduction_random_state is None else reduction_random_state)
    fold_seed = int(random_state if fold_random_state is None else fold_random_state)
    reduced_train, reduced_validation, reduced_test, reduction_audit, _, _ = _reduce_image_block(
        z_train, z_validation, z_test,
        max_components=image_max_components,
        random_state=reduction_seed,
        working_dtype=working_dtype,
    )
    if standardize_reduced_scores:
        score_scaler = StandardScaler().fit(reduced_train)
        score_train = score_scaler.transform(reduced_train)
        score_validation = score_scaler.transform(reduced_validation)
    else:
        score_scaler = None
        score_train = reduced_train
        score_validation = reduced_validation
    ridge_grid: list[dict[str, float]] = []
    for alpha in alphas:
        prediction = Ridge(alpha=alpha).fit(score_train, y_train_array).predict(score_validation)
        ridge_grid.append({"alpha": alpha, "validation_mse": float(np.mean((y_validation_array - prediction) ** 2))})
    selected_alpha = min(ridge_grid, key=lambda row: row["validation_mse"])["alpha"]
    folds = _group_folds(groups, oof_folds, fold_seed)
    train_oof = np.full(len(x_train), np.nan, dtype=float)
    for fold in range(oof_folds):
        fit_rows = folds != fold
        hold_rows = folds == fold
        train_oof[hold_rows] = Ridge(alpha=selected_alpha).fit(
            score_train[fit_rows], y_train_array[fit_rows]
        ).predict(score_train[hold_rows])
    validation_score = Ridge(alpha=selected_alpha).fit(
        score_train, y_train_array
    ).predict(score_validation)
    rf_train = np.column_stack([x_train, train_oof])
    rf_validation = np.column_stack([x_validation, validation_score])
    configs = list(rf_configurations)
    if not configs or not rf_seeds or n_estimators < 1:
        raise InputValidationError("RF-score requires RF configurations, seeds, and positive trees.")
    validation_grid: list[dict[str, Any]] = []
    for configuration in configs:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            bootstrap=True,
            criterion="squared_error",
            random_state=rf_seeds[0],
            **configuration,
        ).fit(rf_train, y_train_array)
        prediction = model.predict(rf_validation)
        validation_grid.append({
            **configuration,
            "validation_mse": float(np.mean((y_validation_array - prediction) ** 2)),
            "selection_seed": rf_seeds[0],
        })
    best_index = int(np.argmin([row["validation_mse"] for row in validation_grid]))
    selected_configuration = dict(configs[best_index])
    validation_models = [
        RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                bootstrap=True,
                criterion="squared_error",
                random_state=seed,
                **selected_configuration,
            ).fit(rf_train, y_train_array)
        for seed in rf_seeds
    ]
    training_prediction = np.mean([model.predict(rf_train) for model in validation_models], axis=0)
    validation_prediction = np.mean([model.predict(rf_validation) for model in validation_models], axis=0)
    combined_image = np.vstack([z_train, z_validation])
    combined_y = np.concatenate([y_train_array, y_validation_array])
    if refit_image_reduction:
        scaler_full = StandardScaler().fit(combined_image)
        combined_reduced = scaler_full.transform(combined_image)
        test_reduced = scaler_full.transform(z_test)
        components = min(image_max_components, combined_image.shape[1], combined_image.shape[0] - 1)
        if components < combined_image.shape[1]:
            pca_full = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=reduction_seed,
            )
            combined_reduced = pca_full.fit_transform(combined_reduced)
            test_reduced = pca_full.transform(test_reduced)
    else:
        combined_reduced = np.vstack([reduced_train, reduced_validation])
        test_reduced = reduced_test
    if standardize_reduced_scores:
        refit_score_scaler = StandardScaler().fit(combined_reduced)
        combined_reduced = refit_score_scaler.transform(combined_reduced)
        test_reduced = refit_score_scaler.transform(test_reduced)
    test_score = Ridge(alpha=selected_alpha).fit(combined_reduced, combined_y).predict(test_reduced)
    rf_combined = np.column_stack([
        np.vstack([x_train, x_validation]),
        np.concatenate([train_oof, validation_score]),
    ])
    rf_test = np.column_stack([x_test, test_score])
    test_prediction = np.mean(
        [
            RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                bootstrap=True,
                criterion="squared_error",
                random_state=seed,
                **selected_configuration,
            ).fit(rf_combined, combined_y).predict(rf_test)
            for seed in rf_seeds
        ],
        axis=0,
    )
    return RFScoreResult(
        training_prediction=training_prediction,
        validation_prediction=validation_prediction,
        test_prediction=test_prediction,
        train_oof_image_score=train_oof,
        validation_image_score=validation_score,
        test_image_score=test_score,
        selected_image_alpha=float(selected_alpha),
        selected_rf_configuration=selected_configuration,
        validation_grid=validation_grid,
        audit={
            "method": "RF-score",
            "task": "regression",
            "image_score": "ridge prediction from one frozen representation",
            "image_reduction": reduction_audit,
            "image_ridge_grid": ridge_grid,
            "reduction_random_state": reduction_seed,
            "fold_random_state": fold_seed,
            "reduced_scores_standardized": standardize_reduced_scores,
            "test_reduction_refit_on_train_validation": refit_image_reduction,
            "working_dtype": working_dtype,
            "n_jobs": n_jobs,
            "selected_image_alpha": float(selected_alpha),
            "training_image_scores": f"{oof_folds}-fold group-safe out-of-fold",
            "validation_image_scores": "training-only ridge model",
            "test_image_scores": "train+validation refit after decisions locked",
            "rf_trees_per_seed": int(n_estimators),
            "rf_seeds": list(rf_seeds),
            "rf_seed_role": "forest fitting only; not partition seeds",
            "rf_configuration_selection_partition": "validation",
            "selected_rf_configuration": selected_configuration,
            "test_outcomes_used": False,
            "causal_interpretation": False,
        },
    )
