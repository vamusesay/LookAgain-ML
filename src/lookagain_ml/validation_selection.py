"""Leakage-safe nested selection for encoder combinations.

The outer validation outcomes are used only to compare already-frozen candidates.
Base-head hyperparameters and prediction-level combinations are learned from
group-safe out-of-fold predictions inside the outer training partition.  After
outer-validation selection, locked decisions are refit on train+validation and
evaluated on the test partition, whose outcomes are excluded from fitting and selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .combination import FittedCombination, _FeatureBlockTransform
from .exceptions import InputValidationError
from .heads import (
    FittedLinearHead,
    _class_weight,
    _fit_numerically_stable_scaler,
    _fit_preprocessor,
)
from .metrics import metric_function


@dataclass
class NestedValidationSelectionResult:
    """Result of one outer train/validation/test nested selection run."""

    selected_encoder: str
    preferred_method: str
    selection_metric: str
    validation_metrics: dict[str, dict[str, Any]]
    test_metrics: dict[str, dict[str, Any]]
    validation_predictions: dict[str, np.ndarray]
    test_predictions: dict[str, np.ndarray]
    selected_hyperparameters: dict[str, dict[str, float]]
    audit: dict[str, Any]


def _loss(task: str, metric: str, y: np.ndarray, prediction: np.ndarray) -> float:
    if task == "classification":
        return float(log_loss(y, prediction, labels=np.arange(prediction.shape[1])))
    if metric == "mae":
        return float(mean_absolute_error(y, prediction))
    if metric == "mse":
        return float(mean_squared_error(y, prediction))
    raise InputValidationError("Regression selection_metric must be 'mse' or 'mae'.")


def _balanced_group_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    values = np.asarray(groups, dtype=object).astype(str)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if folds < 2 or len(unique) < folds:
        raise InputValidationError("Nested cross-fitting requires at least one group per fold.")
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(unique))
    rng.shuffle(order)
    order = order[np.argsort(-counts[order], kind="stable")]
    totals = np.zeros(folds, dtype=int)
    group_fold = np.empty(len(unique), dtype=int)
    for group_index in order:
        fold = int(np.argmin(totals))
        group_fold[group_index] = fold
        totals[fold] += int(counts[group_index])
    return group_fold[inverse]


def _fixed_head(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    parameter: float,
    *,
    random_state: int,
    max_components: int | None,
    pca_iterated_power: int,
    logistic_max_iter: int,
    working_dtype: str,
) -> FittedLinearHead:
    transformed, _, scaler, pca = _fit_preprocessor(
        x_train,
        x_train,
        max_components=max_components,
        pca_iterated_power=pca_iterated_power,
        random_state=random_state,
        working_dtype=working_dtype,
    )
    steps: list[tuple[str, Any]] = [("scale", scaler)]
    if pca is not None:
        steps.append(("pca", pca))
    if task == "regression":
        model = Ridge(alpha=float(parameter)).fit(transformed, np.asarray(y_train, float))
        steps.append(("ridge", model))
        selected = {"alpha": float(parameter)}
        metric = "mse"
    else:
        target = np.asarray(y_train, int)
        classes = np.arange(len(np.unique(target)))
        if not np.array_equal(np.unique(target), classes):
            raise InputValidationError("Every inner training fold must contain all classes.")
        model = LogisticRegression(
            C=float(parameter),
            solver="lbfgs",
            max_iter=int(logistic_max_iter),
            class_weight=_class_weight(target),
            random_state=int(random_state),
        ).fit(transformed, target)
        steps.append(("logistic", model))
        selected = {"C": float(parameter)}
        metric = "log_loss"
    return FittedLinearHead(
        task=task,
        pipeline=Pipeline(steps),
        selected_hyperparameter=selected,
        validation_grid=[],
        selection_metric=metric,
        working_dtype=working_dtype,
    )


def _cross_fitted_head(
    task: str,
    features: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    candidate_values: tuple[float, ...],
    *,
    selection_metric: str,
    folds: int,
    random_state: int,
    max_components: int | None,
    pca_iterated_power: int,
    logistic_max_iter: int,
    working_dtype: str,
) -> tuple[np.ndarray, FittedLinearHead, dict[str, float], list[dict[str, float]], np.ndarray]:
    fold_id = _balanced_group_folds(groups, folds, random_state)
    class_count = len(np.unique(outcome)) if task == "classification" else None
    shape = (len(outcome), class_count) if task == "classification" else (len(outcome),)
    candidates = {value: np.full(shape, np.nan, dtype=float) for value in candidate_values}
    records: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = fold_id != fold
        hold = fold_id == fold
        fit_groups = set(np.asarray(groups)[fit].astype(str))
        hold_groups = set(np.asarray(groups)[hold].astype(str))
        if fit_groups & hold_groups:
            raise InputValidationError("A group crossed an inner base-head fold.")
        transformed_fit, transformed_hold, _, _ = _fit_preprocessor(
            features[fit],
            features[hold],
            max_components=max_components,
            pca_iterated_power=pca_iterated_power,
            random_state=random_state + fold,
            working_dtype=working_dtype,
        )
        for value in candidate_values:
            if task == "regression":
                model = Ridge(alpha=float(value)).fit(transformed_fit, outcome[fit])
                candidates[value][hold] = model.predict(transformed_hold)
            else:
                target = np.asarray(outcome[fit], dtype=int)
                if not np.array_equal(np.unique(target), np.arange(int(class_count))):
                    raise InputValidationError("Every inner training fold must contain all classes.")
                model = LogisticRegression(
                    C=float(value),
                    solver="lbfgs",
                    max_iter=int(logistic_max_iter),
                    class_weight=_class_weight(target),
                    random_state=int(random_state + fold),
                ).fit(transformed_fit, target)
                probability = np.asarray(model.predict_proba(transformed_hold), dtype=float)
                candidates[value][hold] = probability / probability.sum(axis=1, keepdims=True)
        records.append(
            {
                "fold": fold,
                "fit_rows": int(fit.sum()),
                "hold_rows": int(hold.sum()),
                "group_overlap": 0,
            }
        )
    grid: list[dict[str, float]] = []
    for value in candidate_values:
        if not np.isfinite(candidates[value]).all():
            raise RuntimeError("Nested base-head OOF predictions are incomplete.")
        key = "alpha" if task == "regression" else "C"
        grid.append({key: float(value), "inner_oof_loss": _loss(task, selection_metric, outcome, candidates[value])})
    selected_row = min(grid, key=lambda row: row["inner_oof_loss"])
    selected_value = float(selected_row["alpha" if task == "regression" else "C"])
    full = _fixed_head(
        task,
        features,
        outcome,
        selected_value,
        random_state=random_state,
        max_components=max_components,
        pca_iterated_power=pca_iterated_power,
        logistic_max_iter=logistic_max_iter,
        working_dtype=working_dtype,
    )
    return candidates[selected_value], full, dict(full.selected_hyperparameter), grid, fold_id


def _locked_oof(
    task: str,
    features: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    parameter: float,
    *,
    folds: int,
    random_state: int,
    max_components: int | None,
    pca_iterated_power: int,
    logistic_max_iter: int,
    working_dtype: str,
) -> tuple[np.ndarray, FittedLinearHead, np.ndarray]:
    fold_id = _balanced_group_folds(groups, folds, random_state)
    class_count = len(np.unique(outcome)) if task == "classification" else None
    shape = (len(outcome), class_count) if task == "classification" else (len(outcome),)
    oof = np.full(shape, np.nan, dtype=float)
    for fold in range(folds):
        fit, hold = fold_id != fold, fold_id == fold
        head = _fixed_head(
            task,
            features[fit],
            outcome[fit],
            parameter,
            random_state=random_state + fold,
            max_components=max_components,
            pca_iterated_power=pca_iterated_power,
            logistic_max_iter=logistic_max_iter,
            working_dtype=working_dtype,
        )
        oof[hold] = head.predict(features[hold])
    full = _fixed_head(
        task,
        features,
        outcome,
        parameter,
        random_state=random_state,
        max_components=max_components,
        pca_iterated_power=pca_iterated_power,
        logistic_max_iter=logistic_max_iter,
        working_dtype=working_dtype,
    )
    return oof, full, fold_id


def _prediction_matrix(task: str, values: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    return np.column_stack([values[name] for name in order]) if task == "regression" else np.hstack([values[name] for name in order])


def _fit_cross_fitted_stack(task: str, predictions: dict[str, np.ndarray], y: np.ndarray, order: list[str], *, alpha: float, c_value: float, random_state: int) -> FittedCombination:
    matrix = _prediction_matrix(task, predictions, order)
    if task == "regression":
        estimator = Pipeline([("scale", StandardScaler()), ("stack", Ridge(alpha=float(alpha)))])
    else:
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("stack", LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000, random_state=int(random_state))),
        ])
    estimator.fit(matrix, y)
    return FittedCombination("linear_stack", list(order), task, estimator, "Stack fitted only on group-safe outer-training OOF predictions.")


def _greedy_subset(task: str, predictions: dict[str, np.ndarray], y: np.ndarray, order: list[str], metric: str) -> tuple[list[str], list[dict[str, Any]]]:
    remaining = list(order)
    sequence: list[str] = []
    path: list[dict[str, Any]] = []
    while remaining:
        scored: list[tuple[float, int, str]] = []
        for position, name in enumerate(remaining):
            names = [*sequence, name]
            prediction = np.mean(np.stack([predictions[item] for item in names]), axis=0)
            scored.append((_loss(task, metric, y, prediction), position, name))
        loss, _, selected = min(scored)
        sequence.append(selected)
        remaining.remove(selected)
        path.append({"encoders": list(sequence), "inner_oof_loss": float(loss)})
    best = min(range(len(path)), key=lambda index: path[index]["inner_oof_loss"])
    return sequence[: best + 1], path


@dataclass
class _FeatureUnion:
    order: list[str]
    transforms: dict[str, _FeatureBlockTransform]
    head: FittedLinearHead

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        blocks = [self.transforms[name].transform(features[name]) for name in self.order]
        return self.head.predict(np.hstack(blocks))


def _fit_union_blocks(
    features: dict[str, np.ndarray],
    order: list[str],
    *,
    max_components_per_encoder: int,
    random_state: int,
) -> tuple[dict[str, _FeatureBlockTransform], np.ndarray]:
    transforms: dict[str, _FeatureBlockTransform] = {}
    blocks: list[np.ndarray] = []
    for offset, name in enumerate(order):
        values = np.asarray(features[name], dtype=float)
        scaler = _fit_numerically_stable_scaler(values)
        transformed = scaler.transform(values)
        pca = None
        if values.shape[1] > max_components_per_encoder:
            components = min(max_components_per_encoder, len(values) - 1, values.shape[1])
            pca = PCA(n_components=components, svd_solver="randomized", random_state=random_state + offset).fit(transformed)
            transformed = pca.transform(transformed)
        transforms[name] = _FeatureBlockTransform(scaler, pca)
        blocks.append(transformed)
    return transforms, np.hstack(blocks)


def _fit_union_locked(
    task: str,
    features: dict[str, np.ndarray],
    outcome: np.ndarray,
    order: list[str],
    parameter: float,
    *,
    max_components_per_encoder: int,
    random_state: int,
    logistic_max_iter: int,
    working_dtype: str,
) -> _FeatureUnion:
    transforms, union = _fit_union_blocks(
        features,
        order,
        max_components_per_encoder=max_components_per_encoder,
        random_state=random_state,
    )
    head = _fixed_head(
        task,
        union,
        outcome,
        parameter,
        random_state=random_state,
        max_components=None,
        pca_iterated_power=3,
        logistic_max_iter=logistic_max_iter,
        working_dtype=working_dtype,
    )
    return _FeatureUnion(list(order), transforms, head)


def _inner_holdout(groups: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    fold_id = _balanced_group_folds(groups, folds, seed)
    return fold_id != 0, fold_id == 0


def nested_validation_selection(
    features: dict[str, np.ndarray],
    outcome: np.ndarray,
    groups: np.ndarray,
    split_labels: np.ndarray,
    *,
    task: str,
    encoder_order: Iterable[str],
    methods: Iterable[str] = ("equal_average", "linear_stack", "greedy_average", "feature_concatenation"),
    regression_selection_metric: str = "mse",
    inner_folds: int = 5,
    random_state: int = 20260818,
    ridge_alphas: Iterable[float] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    logistic_cs: Iterable[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
    stack_alpha: float = 10.0,
    stack_c: float = 0.1,
    head_max_components: int | None = 256,
    feature_concatenation_max_components: int = 128,
    pca_iterated_power: int = 3,
    logistic_max_iter: int = 2000,
    working_dtype: str = "float64",
) -> NestedValidationSelectionResult:
    """Run validation-based procedure selection and test-set-isolated evaluation."""

    order = list(encoder_order)
    method_order = list(methods)
    allowed = {"equal_average", "linear_stack", "greedy_average", "feature_concatenation"}
    if set(method_order) - allowed:
        raise InputValidationError(f"Unknown nested combination methods: {sorted(set(method_order) - allowed)}")
    labels = np.asarray(split_labels).astype(str)
    y = np.asarray(outcome, dtype=float if task == "regression" else int)
    group_values = np.asarray(groups, dtype=object)
    masks = {name: labels == name for name in ("train", "validation", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise InputValidationError("Outer train, validation, and test partitions must all be nonempty.")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(group_values[masks[left]].astype(str)) & set(group_values[masks[right]].astype(str)):
            raise InputValidationError("A group crosses outer partitions.")
    selection_metric = "log_loss" if task == "classification" else regression_selection_metric
    candidate_values = tuple(float(v) for v in (ridge_alphas if task == "regression" else logistic_cs))
    compute = metric_function(task)

    train_oof: dict[str, np.ndarray] = {}
    outer_validation: dict[str, np.ndarray] = {}
    parameters: dict[str, dict[str, float]] = {}
    grids: dict[str, list[dict[str, float]]] = {}
    base_fold_hashes: dict[str, list[int]] = {}
    for offset, name in enumerate(order):
        oof, head, parameter, grid, fold_id = _cross_fitted_head(
            task,
            np.asarray(features[name])[masks["train"]],
            y[masks["train"]],
            group_values[masks["train"]],
            candidate_values,
            selection_metric=selection_metric,
            folds=inner_folds,
            random_state=random_state + 100 * offset,
            max_components=head_max_components,
            pca_iterated_power=pca_iterated_power,
            logistic_max_iter=logistic_max_iter,
            working_dtype=working_dtype,
        )
        train_oof[name] = oof
        outer_validation[name] = head.predict(np.asarray(features[name])[masks["validation"]])
        parameters[name] = parameter
        grids[name] = grid
        base_fold_hashes[name] = fold_id.tolist()

    validation_metrics = {name: compute(y[masks["validation"]], prediction) for name, prediction in outer_validation.items()}
    selected_encoder = min(order, key=lambda name: float(validation_metrics[name][selection_metric]))
    candidate_validation_predictions: dict[str, np.ndarray] = {selected_encoder: outer_validation[selected_encoder]}
    combination_audit: dict[str, Any] = {}
    if "equal_average" in method_order:
        candidate_validation_predictions["equal_average"] = np.mean(np.stack([outer_validation[name] for name in order]), axis=0)
        combination_audit["equal_average"] = {"fit_source": "prespecified; no outcomes used"}
    stack = None
    if "linear_stack" in method_order:
        stack = _fit_cross_fitted_stack(task, train_oof, y[masks["train"]], order, alpha=stack_alpha, c_value=stack_c, random_state=random_state)
        candidate_validation_predictions["linear_stack"] = stack.predict(outer_validation)
        combination_audit["linear_stack"] = {"fit_source": "outer-training group-safe OOF predictions/outcomes"}
    subset: list[str] | None = None
    if "greedy_average" in method_order:
        subset, path = _greedy_subset(task, train_oof, y[masks["train"]], order, selection_metric)
        candidate_validation_predictions["greedy_average"] = np.mean(np.stack([outer_validation[name] for name in subset]), axis=0)
        combination_audit["greedy_average"] = {"fit_source": "outer-training group-safe OOF predictions/outcomes", "selected_encoders": subset, "path": path}
    union_parameter: float | None = None
    if "feature_concatenation" in method_order:
        fit, tune = _inner_holdout(group_values[masks["train"]], inner_folds, random_state + 8001)
        inner_features = {name: np.asarray(features[name])[masks["train"]] for name in order}
        inner_transforms, inner_fit_union = _fit_union_blocks(
            {name: block[fit] for name, block in inner_features.items()},
            order,
            max_components_per_encoder=feature_concatenation_max_components,
            random_state=random_state + 8100,
        )
        inner_tune_union = np.hstack(
            [inner_transforms[name].transform(inner_features[name][tune]) for name in order]
        )
        union_grid: list[dict[str, float]] = []
        for value in candidate_values:
            local_head = _fixed_head(
                task,
                inner_fit_union,
                y[masks["train"]][fit],
                value,
                random_state=random_state + 8100,
                max_components=None,
                pca_iterated_power=3,
                logistic_max_iter=logistic_max_iter,
                working_dtype=working_dtype,
            )
            pred = local_head.predict(inner_tune_union)
            union_grid.append({"parameter": value, "inner_validation_loss": _loss(task, selection_metric, y[masks["train"]][tune], pred)})
        union_parameter = float(min(union_grid, key=lambda row: row["inner_validation_loss"])["parameter"])
        union = _fit_union_locked(
            task,
            inner_features,
            y[masks["train"]],
            order,
            union_parameter,
            max_components_per_encoder=feature_concatenation_max_components,
            random_state=random_state + 8200,
            logistic_max_iter=logistic_max_iter,
            working_dtype=working_dtype,
        )
        candidate_validation_predictions["feature_concatenation"] = union.predict({name: np.asarray(features[name])[masks["validation"]] for name in order})
        combination_audit["feature_concatenation"] = {"fit_source": "outer-training deterministic group-safe inner holdout; refit on full outer training", "selected_parameter": union_parameter, "inner_grid": union_grid}

    for name, prediction in candidate_validation_predictions.items():
        validation_metrics[name] = compute(y[masks["validation"]], prediction)
    decision_order = [selected_encoder, *method_order]
    preferred_method = min(decision_order, key=lambda name: float(validation_metrics[name][selection_metric]))

    refit_mask = masks["train"] | masks["validation"]
    refit_oof: dict[str, np.ndarray] = {}
    refit_test: dict[str, np.ndarray] = {}
    refit_folds: dict[str, list[int]] = {}
    for offset, name in enumerate(order):
        key = "alpha" if task == "regression" else "C"
        oof, head, fold_id = _locked_oof(
            task,
            np.asarray(features[name])[refit_mask],
            y[refit_mask],
            group_values[refit_mask],
            parameters[name][key],
            folds=inner_folds,
            random_state=random_state + 10000 + 100 * offset,
            max_components=head_max_components,
            pca_iterated_power=pca_iterated_power,
            logistic_max_iter=logistic_max_iter,
            working_dtype=working_dtype,
        )
        refit_oof[name] = oof
        refit_test[name] = head.predict(np.asarray(features[name])[masks["test"]])
        refit_folds[name] = fold_id.tolist()

    test_predictions: dict[str, np.ndarray] = {name: refit_test[name] for name in order}
    if "equal_average" in method_order:
        test_predictions["equal_average"] = np.mean(np.stack([refit_test[name] for name in order]), axis=0)
    if "linear_stack" in method_order:
        refit_stack = _fit_cross_fitted_stack(task, refit_oof, y[refit_mask], order, alpha=stack_alpha, c_value=stack_c, random_state=random_state + 9000)
        test_predictions["linear_stack"] = refit_stack.predict(refit_test)
    if "greedy_average" in method_order:
        assert subset is not None
        test_predictions["greedy_average"] = np.mean(np.stack([refit_test[name] for name in subset]), axis=0)
    if "feature_concatenation" in method_order:
        assert union_parameter is not None
        refit_union = _fit_union_locked(
            task,
            {name: np.asarray(features[name])[refit_mask] for name in order},
            y[refit_mask],
            order,
            union_parameter,
            max_components_per_encoder=feature_concatenation_max_components,
            random_state=random_state + 9200,
            logistic_max_iter=logistic_max_iter,
            working_dtype=working_dtype,
        )
        test_predictions["feature_concatenation"] = refit_union.predict({name: np.asarray(features[name])[masks["test"]] for name in order})
    test_metrics = {name: compute(y[masks["test"]], prediction) for name, prediction in test_predictions.items()}
    audit = {
        "protocol": "honest_nested_outer_validation_v1",
        "outer_validation_role": "comparison of candidates frozen without outer-validation outcomes",
        "outer_validation_outcomes_used_to_fit_candidates": False,
        "test_outcomes_used_for_fitting_or_selection": False,
        "base_head_tuning": "group-safe OOF loss inside outer training",
        "meta_model_fit": "group-safe OOF base predictions and outer-training outcomes",
        "feature_union_tuning": "deterministic group-safe inner holdout inside outer training",
        "test_refit": "locked decisions; base heads refit on train+validation; meta-model refit on leakage-safe group-OOF train+validation predictions",
        "inner_folds": inner_folds,
        "selected_encoder": selected_encoder,
        "preferred_method": preferred_method,
        "base_head_grids": grids,
        "base_inner_fold_assignments": base_fold_hashes,
        "refit_inner_fold_assignments": refit_folds,
        "combination_audit": combination_audit,
        "near_null_rule": "fixed float64 training-only floor sqrt(eps) times max(1, training column magnitude); stabilized columns are left unscaled",
    }
    return NestedValidationSelectionResult(
        selected_encoder=selected_encoder,
        preferred_method=preferred_method,
        selection_metric=selection_metric,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_predictions=candidate_validation_predictions,
        test_predictions=test_predictions,
        selected_hyperparameters=parameters,
        audit=audit,
    )
