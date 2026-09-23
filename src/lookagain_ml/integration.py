"""Validation-locked integration of structured and image information."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .exceptions import InputValidationError
from .heads import fit_linear_head, fitted_linear_head_audit, numeric_state_sha256
from .metrics import metric_function
from .structured import FittedNeuralHead, fit_neural_head

INTEGRATION_METHODS = ("linear_score", "nn_score", "rf_score", "joint_linear", "joint_nn")


def normalize_integration_methods(methods: Iterable[str] | None) -> list[str]:
    if methods is None:
        return ["linear_score"]
    if isinstance(methods, str):
        raise InputValidationError(
            "integration_methods must be an iterable such as ['linear_score', 'rf_score', 'joint_linear']."
        )
    normalized: list[str] = []
    for raw in methods:
        name = str(raw).strip().lower()
        if name not in INTEGRATION_METHODS:
            raise InputValidationError(
                f"Unknown integration method {raw!r}; choose from {list(INTEGRATION_METHODS)}."
            )
        if name in normalized:
            raise InputValidationError(f"Integration method {name!r} was requested more than once.")
        normalized.append(name)
    if not normalized:
        raise InputValidationError("integration_methods must contain at least one method.")
    return normalized


def _score_matrix(task: str, structured_prediction: np.ndarray, image_prediction: np.ndarray) -> np.ndarray:
    structured = np.asarray(structured_prediction, dtype=float)
    image = np.asarray(image_prediction, dtype=float)
    if structured.shape[0] != image.shape[0]:
        raise InputValidationError("Structured and image predictions must have aligned rows.")
    if task == "regression":
        if structured.ndim != 1 or image.ndim != 1:
            raise InputValidationError("Regression score integration requires one score per row.")
        return np.column_stack([structured, image])
    if structured.ndim != 2 or image.ndim != 2 or structured.shape[1] != image.shape[1]:
        raise InputValidationError(
            "Classification score integration requires aligned class-probability matrices."
        )
    return np.hstack([structured, image])


def _inner_score_split(
    y: np.ndarray, groups: np.ndarray, task: str, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find a group-disjoint inner split of the original validation partition."""

    indices = np.arange(len(y))
    if len(indices) < 8 or len(np.unique(groups)) < 4:
        raise InputValidationError(
            "nn_score requires at least 8 validation rows in at least 4 independent groups."
        )
    for offset in range(20):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=random_state + offset
        )
        inner_train, inner_tune = next(splitter.split(indices, y, groups))
        if task == "regression" or (
            set(np.asarray(y)[inner_train].tolist()) == set(np.asarray(y).tolist())
            and set(np.asarray(y)[inner_tune].tolist()) == set(np.asarray(y).tolist())
        ):
            return inner_train, inner_tune
    raise InputValidationError(
        "nn_score could not form a group-safe validation-internal split containing every class."
    )


@dataclass
class FittedScoreIntegration:
    name: str
    task: str
    estimator: Pipeline | FittedNeuralHead
    configuration: dict[str, Any]
    selection_metrics: dict[str, Any] | None = None

    def predict(self, structured_prediction: np.ndarray, image_prediction: np.ndarray) -> np.ndarray:
        matrix = _score_matrix(self.task, structured_prediction, image_prediction)
        if isinstance(self.estimator, FittedNeuralHead):
            return self.estimator.predict(matrix)
        if self.task == "regression":
            return np.asarray(self.estimator.predict(matrix), dtype=float)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=float)

    def metadata(self) -> dict[str, Any]:
        if isinstance(self.estimator, FittedNeuralHead):
            fitted_state = self.estimator.audit()["fitted_state_sha256"]
            neural_audit = self.estimator.audit()
        else:
            scaler = self.estimator.named_steps["scale"]
            model = self.estimator.named_steps["integration"]
            fitted_state = numeric_state_sha256(
                scaler.mean_, scaler.scale_, model.coef_, model.intercept_
            )
            neural_audit = None
        return {
            "method": self.name,
            "integration_level": "separately fitted outcome scores",
            "inputs": "structured prediction and image prediction only",
            "fit_partition": "validation",
            "test_outcomes_used": False,
            "configuration": self.configuration,
            "neural_audit": neural_audit,
            "fitted_state_sha256": fitted_state,
        }


def fit_score_integration(
    task: str,
    structured_validation_prediction: np.ndarray,
    image_validation_prediction: np.ndarray,
    y_validation: np.ndarray,
    *,
    validation_groups: np.ndarray,
    method: str,
    random_state: int,
    alpha: float,
    c_value: float,
    nn_max_iter: int,
) -> FittedScoreIntegration:
    """Fit one score integration using validation rows only; no test argument is accepted."""

    matrix = _score_matrix(task, structured_validation_prediction, image_validation_prediction)
    if not np.isfinite(matrix).all():
        raise InputValidationError("Score-integration inputs must be finite.")
    y_values = np.asarray(y_validation)
    groups = np.asarray(validation_groups, dtype=object)
    if len(groups) != len(y_values):
        raise InputValidationError("validation_groups must align with validation scores.")
    if task == "classification":
        expected_classes = set(range(matrix.shape[1] // 2))
        observed_classes = set(np.asarray(y_values, dtype=int).tolist())
        if observed_classes != expected_classes:
            missing = sorted(expected_classes - observed_classes)
            unexpected = sorted(observed_classes - expected_classes)
            details = []
            if missing:
                details.append(f"missing encoded classes {missing}")
            if unexpected:
                details.append(f"unexpected encoded classes {unexpected}")
            raise InputValidationError(
                "Classification score integration is fitted on validation outcomes, so the "
                "validation partition must contain every encoded class ("
                + "; ".join(details)
                + "). Supply more independent groups, explicit safe split labels, or use a "
                "joint integration method."
            )
    if method == "linear_score":
        if task == "regression":
            if alpha < 0:
                raise InputValidationError("score_integration_alpha must be nonnegative.")
            if alpha == 0:
                model: Any = LinearRegression()
                configuration = {"model": "ordinary_least_squares", "alpha": 0.0}
            else:
                model = Ridge(alpha=float(alpha))
                configuration = {"model": "ridge", "alpha": float(alpha)}
        else:
            if c_value <= 0:
                raise InputValidationError("score_integration_c must be positive.")
            model = LogisticRegression(
                C=float(c_value), solver="lbfgs", max_iter=2000, random_state=random_state
            )
            configuration = {"model": "logistic", "C": float(c_value)}
        pipeline = Pipeline([("scale", StandardScaler()), ("integration", model)])
        pipeline.fit(matrix, y_values)
        return FittedScoreIntegration(method, task, pipeline, configuration)
    if method == "nn_score":
        inner_train, inner_tune = _inner_score_split(y_values, groups, task, random_state)
        neural = fit_neural_head(
            task,
            matrix[inner_train],
            y_values[inner_train],
            matrix[inner_tune],
            y_values[inner_tune],
            architectures=((8,), (8, 4)),
            random_state=random_state,
            max_iter=nn_max_iter,
        )
        configuration = {
            "model": "small_adamw_mlp",
            "architecture_candidates": [[8], [8, 4]],
            "fit_partition": "validation-inner-train",
            "early_stopping_and_selection_partition": "validation-inner-tune",
            "reported_selection_metric_partition": "validation-inner-tune",
            "inner_split_group_safe": True,
            "inner_train_n": len(inner_train),
            "inner_tune_n": len(inner_tune),
        }
        selection_metrics = metric_function(task)(
            y_values[inner_tune], neural.predict(matrix[inner_tune])
        )
        return FittedScoreIntegration(
            method, task, neural, configuration, selection_metrics=selection_metrics
        )
    raise InputValidationError("fit_score_integration accepts linear_score or nn_score.")


@dataclass
class _ImageBlockTransform:
    scaler: StandardScaler
    pca: PCA | None

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = self.scaler.transform(np.asarray(values, dtype=float))
        if self.pca is not None:
            result = self.pca.transform(result)
        return np.asarray(result, dtype=float)


@dataclass
class FittedJointIntegration:
    name: str
    task: str
    image_encoder_names: list[str]
    image_transforms: dict[str, _ImageBlockTransform]
    head: Any
    max_components_per_encoder: int

    def _matrix(self, structured_features: np.ndarray, image_features: dict[str, np.ndarray]) -> np.ndarray:
        missing = set(self.image_encoder_names) - set(image_features)
        if missing:
            raise InputValidationError(
                f"Joint integration is missing image representations for {sorted(missing)}."
            )
        blocks = [np.asarray(structured_features, dtype=float)]
        blocks.extend(
            self.image_transforms[name].transform(image_features[name])
            for name in self.image_encoder_names
        )
        return np.hstack(blocks)

    def predict(self, structured_features: np.ndarray, image_features: dict[str, np.ndarray]) -> np.ndarray:
        return self.head.predict(self._matrix(structured_features, image_features))

    def metadata(self) -> dict[str, Any]:
        blocks: dict[str, Any] = {}
        for name in self.image_encoder_names:
            transform = self.image_transforms[name]
            state = [transform.scaler.mean_, transform.scaler.scale_]
            if transform.pca is not None:
                state.extend([transform.pca.mean_, transform.pca.components_])
            blocks[name] = {
                "input_dimension": len(transform.scaler.mean_),
                "output_dimension": int(
                    transform.pca.n_components_
                    if transform.pca is not None
                    else len(transform.scaler.mean_)
                ),
                "pca_applied": transform.pca is not None,
                "fit_partition": "train",
                "fitted_state_sha256": numeric_state_sha256(*state),
            }
        head_audit = (
            fitted_linear_head_audit(self.head)
            if self.name == "joint_linear"
            else self.head.audit()
        )
        return {
            "method": self.name,
            "integration_level": "structured covariates plus image representations",
            "image_encoders": list(self.image_encoder_names),
            "image_block_transforms": blocks,
            "max_components_per_encoder": self.max_components_per_encoder,
            "head": head_audit,
            "test_outcomes_used": False,
        }


def fit_joint_integration(
    task: str,
    structured_train: np.ndarray,
    image_train: dict[str, np.ndarray],
    y_train: np.ndarray,
    structured_validation: np.ndarray,
    image_validation: dict[str, np.ndarray],
    y_validation: np.ndarray,
    image_encoder_names: list[str],
    *,
    method: str,
    max_components_per_encoder: int,
    ridge_alphas: Iterable[float],
    logistic_cs: Iterable[float],
    random_state: int,
    nn_max_iter: int,
) -> FittedJointIntegration:
    """Fit train-only image reduction and a validation-tuned joint head."""

    if max_components_per_encoder < 1:
        raise InputValidationError("joint_image_max_components must be positive.")
    transforms: dict[str, _ImageBlockTransform] = {}
    train_blocks = [np.asarray(structured_train, dtype=float)]
    validation_blocks = [np.asarray(structured_validation, dtype=float)]
    for offset, name in enumerate(image_encoder_names):
        train = np.asarray(image_train[name], dtype=float)
        validation = np.asarray(image_validation[name], dtype=float)
        if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
            raise InputValidationError(f"Image feature block {name!r} is not column-aligned.")
        scaler = StandardScaler().fit(train)
        train_value = scaler.transform(train)
        validation_value = scaler.transform(validation)
        pca = None
        if train.shape[1] > max_components_per_encoder:
            components = min(max_components_per_encoder, train.shape[0] - 1, train.shape[1])
            if components < 1:
                raise InputValidationError("Joint PCA requires at least two training rows.")
            pca = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=random_state + offset,
            ).fit(train_value)
            train_value = pca.transform(train_value)
            validation_value = pca.transform(validation_value)
        transforms[name] = _ImageBlockTransform(scaler, pca)
        train_blocks.append(train_value)
        validation_blocks.append(validation_value)
    train_matrix = np.hstack(train_blocks)
    validation_matrix = np.hstack(validation_blocks)
    if method == "joint_linear":
        head = fit_linear_head(
            task,
            train_matrix,
            y_train,
            validation_matrix,
            y_validation,
            ridge_alphas=ridge_alphas,
            logistic_cs=logistic_cs,
            random_state=random_state,
        )
    elif method == "joint_nn":
        head = fit_neural_head(
            task,
            train_matrix,
            y_train,
            validation_matrix,
            y_validation,
            random_state=random_state,
            max_iter=nn_max_iter,
        )
    else:
        raise InputValidationError("fit_joint_integration accepts joint_linear or joint_nn.")
    return FittedJointIntegration(
        method,
        task,
        list(image_encoder_names),
        transforms,
        head,
        max_components_per_encoder,
    )


def incremental_metrics(
    task: str, structured_metrics: dict[str, Any], combined_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Report paired fitted-procedure differences with direction made explicit."""

    if task == "regression":
        baseline_mse = float(structured_metrics["mse"])
        combined_mse = float(combined_metrics["mse"])
        return {
            "delta_r2": float(combined_metrics["r2"]) - float(structured_metrics["r2"]),
            "mse_reduction": baseline_mse - combined_mse,
            "relative_mse_reduction": (
                (baseline_mse - combined_mse) / baseline_mse if baseline_mse > 0 else None
            ),
            "interpretation": (
                "Fitted X+I minus fitted X performance on the same rows. A negative fitted "
                "increment does not imply negative population information."
            ),
        }
    return {
        "delta_accuracy": float(combined_metrics["accuracy"]) - float(structured_metrics["accuracy"]),
        "delta_balanced_accuracy": (
            float(combined_metrics["balanced_accuracy"])
            - float(structured_metrics["balanced_accuracy"])
        ),
        "delta_macro_f1": float(combined_metrics["macro_f1"]) - float(structured_metrics["macro_f1"]),
        "log_loss_reduction": float(structured_metrics["log_loss"]) - float(combined_metrics["log_loss"]),
        "interpretation": (
            "Fitted X+I minus fitted X performance on the same rows; image-only prediction "
            "is not evidence of incremental information."
        ),
    }
