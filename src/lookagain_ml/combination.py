"""Validation-only combination of frozen image representations."""

from __future__ import annotations

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
from .heads import (
    FittedLinearHead,
    fit_linear_head,
    fitted_linear_head_audit,
    numeric_state_sha256,
)

COMBINATION_METHODS = (
    "equal_average",
    "linear_stack",
    "greedy_average",
    "feature_concatenation",
)
PREDICTION_COMBINATION_METHODS = (
    "equal_average",
    "linear_stack",
    "greedy_average",
)


@dataclass
class FittedCombination:
    """A prediction-level rule whose membership/weights are already frozen."""

    name: str
    encoder_names: list[str]
    task: str
    estimator: Any = None
    description: str = ""

    def predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        missing = set(self.encoder_names) - set(predictions)
        if missing:
            raise InputValidationError(
                f"Combination is missing predictions for {sorted(missing)}."
            )
        values = [np.asarray(predictions[name]) for name in self.encoder_names]
        if self.name in {"equal_average", "greedy_average"}:
            return np.mean(np.stack(values, axis=0), axis=0)
        matrix = (
            np.column_stack(values) if self.task == "regression" else np.hstack(values)
        )
        if self.task == "regression":
            return np.asarray(self.estimator.predict(matrix), dtype=float)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=float)

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": self.name,
            "encoders": list(self.encoder_names),
            "fit_partition": "validation" if self.name == "linear_stack" else "none",
            "description": self.description,
        }
        if self.name in {"equal_average", "greedy_average"}:
            result["weights"] = {
                name: 1.0 / len(self.encoder_names) for name in self.encoder_names
            }
        if self.name == "linear_stack":
            scaler = self.estimator.named_steps["scale"]
            model = self.estimator.named_steps["stack"]
            result["fixed_regularization"] = (
                {"alpha": float(model.alpha)}
                if self.task == "regression"
                else {"C": float(model.C)}
            )
            class_count = len(model.classes_) if self.task == "classification" else None
            result["input_columns"] = (
                list(self.encoder_names)
                if self.task == "regression"
                else [
                    f"{name}__probability_class_{class_index}"
                    for name in self.encoder_names
                    for class_index in range(int(class_count))
                ]
            )
            result["input_scaler_mean"] = np.asarray(scaler.mean_).tolist()
            result["input_scaler_scale"] = np.asarray(scaler.scale_).tolist()
            result["coefficients_on_standardized_inputs"] = np.asarray(
                model.coef_
            ).tolist()
            result["intercept"] = np.asarray(model.intercept_).tolist()
            if self.task == "classification":
                result["classes"] = np.asarray(model.classes_).tolist()
        return result


@dataclass
class _FeatureBlockTransform:
    scaler: StandardScaler
    pca: PCA | None

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(values, dtype=float))
        if self.pca is not None:
            transformed = self.pca.transform(transformed)
        return np.asarray(transformed, dtype=float)


@dataclass
class FittedFeatureConcatenation:
    """Train-fitted block transforms plus a validation-tuned common head."""

    encoder_names: list[str]
    task: str
    transforms: dict[str, _FeatureBlockTransform]
    head: FittedLinearHead
    max_components_per_encoder: int
    name: str = "feature_concatenation"

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        missing = set(self.encoder_names) - set(features)
        if missing:
            raise InputValidationError(
                f"Feature concatenation is missing representations for {sorted(missing)}."
            )
        blocks = [
            self.transforms[name].transform(features[name]) for name in self.encoder_names
        ]
        return self.head.predict(np.hstack(blocks))

    def metadata(self) -> dict[str, Any]:
        block_metadata: dict[str, Any] = {}
        for name in self.encoder_names:
            transform = self.transforms[name]
            state = [transform.scaler.mean_, transform.scaler.scale_]
            if transform.pca is not None:
                state.extend([transform.pca.mean_, transform.pca.components_])
            block_metadata[name] = {
                "input_dimension": len(transform.scaler.mean_),
                "output_dimension": int(
                    transform.pca.n_components_
                    if transform.pca is not None
                    else len(transform.scaler.mean_)
                ),
                "pca_applied": transform.pca is not None,
                "training_transform_sha256": numeric_state_sha256(*state),
            }
        return {
            "method": self.name,
            "encoders": list(self.encoder_names),
            "fit_partition": "training features/outcomes; validation tunes the common head",
            "description": (
                "Standardized encoder blocks with train-only PCA when needed, then a "
                "validation-tuned common linear head."
            ),
            "max_components_per_encoder": self.max_components_per_encoder,
            "block_transforms": block_metadata,
            "head": fitted_linear_head_audit(self.head),
        }


def normalize_combination_methods(methods: Iterable[str] | None) -> list[str]:
    if methods is None:
        return []
    if isinstance(methods, str):
        raise InputValidationError(
            "combinations must be an iterable such as ['equal_average', 'linear_stack']."
        )
    normalized: list[str] = []
    for raw in methods:
        name = str(raw).strip().lower()
        if name not in COMBINATION_METHODS:
            raise InputValidationError(
                f"Unknown combination {raw!r}; choose from {list(COMBINATION_METHODS)}."
            )
        if name in normalized:
            raise InputValidationError(
                f"Combination {name!r} was requested more than once."
            )
        normalized.append(name)
    return normalized


def _average_loss(task: str, y: np.ndarray, values: list[np.ndarray]) -> float:
    prediction = np.mean(np.stack(values, axis=0), axis=0)
    if task == "regression":
        return float(mean_squared_error(y, prediction))
    return float(log_loss(y, prediction, labels=np.arange(prediction.shape[1])))


def _greedy_encoder_prefix(
    task: str,
    validation_predictions: dict[str, np.ndarray],
    y_validation: np.ndarray,
    encoder_order: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    remaining = list(encoder_order)
    sequence: list[str] = []
    path: list[dict[str, Any]] = []
    while remaining:
        candidates = []
        for position, name in enumerate(remaining):
            names = [*sequence, name]
            loss = _average_loss(
                task,
                y_validation,
                [validation_predictions[item] for item in names],
            )
            candidates.append((loss, position, name))
        loss, _, selected = min(candidates)
        sequence.append(selected)
        remaining.remove(selected)
        path.append({"encoders": list(sequence), "validation_loss": float(loss)})
    best_length = min(
        range(len(path)), key=lambda index: path[index]["validation_loss"]
    ) + 1
    return sequence[:best_length], path


def fit_validation_combinations(
    task: str,
    validation_predictions: dict[str, np.ndarray],
    y_validation: np.ndarray,
    encoder_order: list[str],
    methods: list[str],
    *,
    random_state: int,
    stack_alpha: float = 10.0,
    stack_c: float = 0.1,
) -> tuple[dict[str, FittedCombination], dict[str, dict[str, Any]]]:
    """Fit combination decisions using validation outcomes only.

    This boundary intentionally has no test-outcome or test-prediction argument.
    """

    unsupported = set(methods) - set(PREDICTION_COMBINATION_METHODS)
    if unsupported:
        raise InputValidationError(
            "fit_validation_combinations only accepts prediction-level methods; "
            f"got {sorted(unsupported)}."
        )
    if len(encoder_order) < 2 and methods:
        raise InputValidationError(
            "At least two encoders are required for combination methods."
        )
    fitted: dict[str, FittedCombination] = {}
    audit: dict[str, dict[str, Any]] = {}
    if "equal_average" in methods:
        fitted["equal_average"] = FittedCombination(
            "equal_average",
            list(encoder_order),
            task,
            description="Equal average fixed without outcomes.",
        )
        audit["equal_average"] = {
            "selection_source": "prespecified",
            "path": None,
        }
    if "linear_stack" in methods:
        matrix = (
            np.column_stack([validation_predictions[name] for name in encoder_order])
            if task == "regression"
            else np.hstack([validation_predictions[name] for name in encoder_order])
        )
        if task == "regression":
            if stack_alpha <= 0:
                raise InputValidationError("stack_alpha must be positive.")
            estimator = Pipeline(
                [("scale", StandardScaler()), ("stack", Ridge(alpha=float(stack_alpha)))]
            ).fit(matrix, y_validation)
        else:
            if stack_c <= 0:
                raise InputValidationError("stack_c must be positive.")
            class_count = next(iter(validation_predictions.values())).shape[1]
            if not np.array_equal(np.unique(y_validation), np.arange(class_count)):
                raise InputValidationError(
                    "Linear stacking requires every class in the validation partition. "
                    "Supply more independent validation groups or omit linear_stack."
                )
            estimator = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "stack",
                        LogisticRegression(
                            C=float(stack_c),
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ).fit(matrix, y_validation)
        fitted["linear_stack"] = FittedCombination(
            "linear_stack",
            list(encoder_order),
            task,
            estimator,
            "Fixed-regularization linear stack fitted on validation predictions/outcomes.",
        )
        audit["linear_stack"] = {
            "selection_source": "validation outcomes only",
            "regularization_prespecified": True,
            "path": None,
        }
    if "greedy_average" in methods:
        selected, path = _greedy_encoder_prefix(
            task, validation_predictions, y_validation, encoder_order
        )
        fitted["greedy_average"] = FittedCombination(
            "greedy_average",
            selected,
            task,
            description="Greedy equal-average prefix selected by validation loss.",
        )
        audit["greedy_average"] = {
            "selection_source": "validation outcomes only",
            "path": path,
            "selected_encoders": selected,
        }
    return fitted, audit


def fit_feature_concatenation(
    task: str,
    training_features: dict[str, np.ndarray],
    y_train: np.ndarray,
    validation_features: dict[str, np.ndarray],
    y_validation: np.ndarray,
    encoder_order: list[str],
    *,
    max_components_per_encoder: int = 128,
    ridge_alphas: Iterable[float],
    logistic_cs: Iterable[float],
    random_state: int,
) -> tuple[FittedFeatureConcatenation, dict[str, Any]]:
    """Fit feature union without accepting test features or test outcomes."""

    if len(encoder_order) < 2:
        raise InputValidationError(
            "At least two encoders are required for feature_concatenation."
        )
    if max_components_per_encoder < 1:
        raise InputValidationError(
            "feature_concatenation_max_components must be positive."
        )
    transforms: dict[str, _FeatureBlockTransform] = {}
    train_blocks: list[np.ndarray] = []
    validation_blocks: list[np.ndarray] = []
    for name in encoder_order:
        train = np.asarray(training_features[name], dtype=float)
        validation = np.asarray(validation_features[name], dtype=float)
        if (
            train.ndim != 2
            or validation.ndim != 2
            or train.shape[1] != validation.shape[1]
        ):
            raise InputValidationError(
                f"Feature block {name!r} must have aligned two-dimensional train/validation columns."
            )
        if not np.isfinite(train).all() or not np.isfinite(validation).all():
            raise InputValidationError(f"Feature block {name!r} must be finite.")
        scaler = StandardScaler().fit(train)
        train_transformed = scaler.transform(train)
        validation_transformed = scaler.transform(validation)
        pca = None
        if train.shape[1] > max_components_per_encoder:
            component_count = min(
                max_components_per_encoder, train.shape[0] - 1, train.shape[1]
            )
            if component_count < 1:
                raise InputValidationError(
                    "Feature concatenation PCA requires at least two training observations."
                )
            pca = PCA(
                n_components=component_count,
                svd_solver="randomized",
                random_state=random_state,
            ).fit(train_transformed)
            train_transformed = pca.transform(train_transformed)
            validation_transformed = pca.transform(validation_transformed)
        transforms[name] = _FeatureBlockTransform(scaler=scaler, pca=pca)
        train_blocks.append(train_transformed)
        validation_blocks.append(validation_transformed)
    head = fit_linear_head(
        task,
        np.hstack(train_blocks),
        y_train,
        np.hstack(validation_blocks),
        y_validation,
        ridge_alphas=ridge_alphas,
        logistic_cs=logistic_cs,
        random_state=random_state,
    )
    fitted = FittedFeatureConcatenation(
        encoder_names=list(encoder_order),
        task=task,
        transforms=transforms,
        head=head,
        max_components_per_encoder=max_components_per_encoder,
    )
    return fitted, {
        "selection_source": "training fit and validation-only head tuning",
        "test_information_used": False,
        "selected_encoders": list(encoder_order),
        "head_validation_grid": head.validation_grid,
        "head_selected_hyperparameter": head.selected_hyperparameter,
    }
