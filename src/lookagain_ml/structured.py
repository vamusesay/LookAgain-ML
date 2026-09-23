"""Leakage-safe preprocessing and models for structured covariates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .exceptions import InputValidationError
from .heads import FittedLinearHead, fit_linear_head, fitted_linear_head_audit
from .metrics import metric_function

STRUCTURED_MODELS = ("linear", "nn", "rf")
DEFAULT_NN_ARCHITECTURES = ((32,), (64, 32))
DEFAULT_RF_CONFIGURATIONS = (
    {"max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
    {"max_features": "sqrt", "min_samples_leaf": 5, "max_depth": 20},
    {"max_features": 0.33, "min_samples_leaf": 20, "max_depth": None},
    {"max_features": 0.67, "min_samples_leaf": 1, "max_depth": None},
)
DEFAULT_RF_SEEDS = (2026082601, 2026082602, 2026082603)
_MISSING_CATEGORY = "__LOOKAGAIN_MISSING__"


def _state_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_structured_models(models: Iterable[str] | None) -> list[str]:
    if models is None:
        return ["linear"]
    if isinstance(models, str):
        raise InputValidationError(
            "structured_models must be an iterable such as ['linear', 'nn', 'rf']."
        )
    normalized: list[str] = []
    for raw in models:
        name = str(raw).strip().lower()
        if name not in STRUCTURED_MODELS:
            raise InputValidationError(
                f"Unknown structured model {raw!r}; choose from {list(STRUCTURED_MODELS)}."
            )
        if name in normalized:
            raise InputValidationError(f"Structured model {name!r} was requested more than once.")
        normalized.append(name)
    if not normalized:
        raise InputValidationError("structured_models must contain at least one method.")
    return normalized


def subset_covariates(covariates: Any, source_positions: np.ndarray, requested_n: int) -> Any:
    """Align covariates to image-manifest rows without silently dropping independently."""

    if isinstance(covariates, pd.DataFrame):
        if len(covariates) != requested_n:
            raise InputValidationError(
                "covariates must have one row per requested image before image validation; "
                f"got {len(covariates)} and {requested_n}."
            )
        return covariates.iloc[np.asarray(source_positions, dtype=int)].reset_index(drop=True)
    values = np.asarray(covariates)
    if values.ndim != 2:
        raise InputValidationError(
            "covariates must be a pandas DataFrame or a two-dimensional numeric matrix."
        )
    if values.shape[0] != requested_n:
        raise InputValidationError(
            "covariates must have one row per requested image before image validation; "
            f"got {values.shape[0]} and {requested_n}."
        )
    return values[np.asarray(source_positions, dtype=int)]


@dataclass
class FittedStructuredPreprocessor:
    """A serializable train-fitted numeric/categorical transformation."""

    input_kind: str
    input_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    missing_indicator_columns: list[str]
    categories: dict[str, list[str]]
    output_feature_names: list[str]
    preprocessed_numeric_matrix: bool

    def transform(self, covariates: Any) -> np.ndarray:
        if self.preprocessed_numeric_matrix:
            values = np.asarray(covariates, dtype=float)
            if values.ndim != 2 or values.shape[1] != len(self.output_feature_names):
                raise InputValidationError(
                    "The preprocessed covariate matrix must retain the fitted number of columns."
                )
            if not np.isfinite(values).all():
                raise InputValidationError(
                    "Preprocessed covariate matrices must be finite; impute them before analyze()."
                )
            return values

        frame = _as_frame(covariates, self.input_columns)
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            numeric_values = []
            for name in self.numeric_columns:
                raw = _numeric_array(frame[name])
                missing = ~np.isfinite(raw)
                clean = np.where(missing, self.numeric_medians[name], raw)
                scaled = (clean - self.numeric_means[name]) / self.numeric_scales[name]
                numeric_values.append(scaled)
                if name in self.missing_indicator_columns:
                    numeric_values.append(missing.astype(float))
            blocks.append(np.column_stack(numeric_values))
        if self.categorical_columns:
            categorical_values = []
            for name in self.categorical_columns:
                values = _categorical_strings(frame[name])
                for category in self.categories[name]:
                    categorical_values.append((values == category).astype(float))
            blocks.append(np.column_stack(categorical_values))
        if not blocks:
            raise InputValidationError("covariates contain no usable columns.")
        result = np.column_stack(blocks).astype(float, copy=False)
        if not np.isfinite(result).all():
            raise InputValidationError("Structured preprocessing produced non-finite values.")
        return result

    def metadata(self) -> dict[str, Any]:
        state = {
            "input_kind": self.input_kind,
            "input_columns": self.input_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "missing_indicator_columns": self.missing_indicator_columns,
            "categories": self.categories,
            "output_feature_names": self.output_feature_names,
            "preprocessed_numeric_matrix": self.preprocessed_numeric_matrix,
        }
        return {
            **state,
            "fit_partition": "train",
            "unseen_category_handling": (
                "all-zero one-hot block" if self.categorical_columns else "not applicable"
            ),
            "target_encoding": "not used",
            "fitted_state_sha256": _state_digest(state),
            "output_dimension": len(self.output_feature_names),
        }


def _categorical_strings(series: pd.Series) -> np.ndarray:
    values = series.astype(object).where(~series.isna(), _MISSING_CATEGORY)
    return values.map(str).to_numpy(dtype=str)


def _numeric_array(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        return numeric.to_numpy(dtype=float, na_value=np.nan)
    except TypeError:
        return numeric.astype(float).to_numpy()


def _as_frame(covariates: Any, expected_columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(covariates, pd.DataFrame):
        frame = covariates.copy()
        frame.columns = [str(column) for column in frame.columns]
    else:
        values = np.asarray(covariates)
        if values.ndim != 2:
            raise InputValidationError("covariates must be two-dimensional.")
        columns = expected_columns or [f"x{index}" for index in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=columns)
    if expected_columns is not None:
        missing = [name for name in expected_columns if name not in frame]
        if missing:
            raise InputValidationError(f"Covariates are missing fitted columns: {missing}.")
        frame = frame[expected_columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise InputValidationError(f"Covariate column names must be unique; duplicates: {duplicates}.")
    if frame.shape[1] == 0:
        raise InputValidationError("covariates must contain at least one column.")
    return frame


def fit_structured_preprocessor(
    covariates_train: Any,
    *,
    missing_indicators: bool = True,
    preprocessed_numeric_matrix: bool = False,
) -> FittedStructuredPreprocessor:
    """Fit preprocessing on training rows only; this function accepts no outcomes."""

    if preprocessed_numeric_matrix:
        values = np.asarray(covariates_train, dtype=float)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise InputValidationError(
                "A preprocessed covariate matrix must be finite, two-dimensional, and nonempty."
            )
        names = [f"x{index}" for index in range(values.shape[1])]
        return FittedStructuredPreprocessor(
            input_kind="preprocessed_numeric_matrix",
            input_columns=names,
            numeric_columns=names,
            categorical_columns=[],
            numeric_medians={},
            numeric_means={},
            numeric_scales={},
            missing_indicator_columns=[],
            categories={},
            output_feature_names=names,
            preprocessed_numeric_matrix=True,
        )

    frame = _as_frame(covariates_train)
    numeric_columns = [
        str(name)
        for name in frame.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(frame[name])
    ]
    categorical_columns = [name for name in frame.columns if name not in numeric_columns]
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    indicator_columns: list[str] = []
    output_names: list[str] = []
    for name in numeric_columns:
        raw = _numeric_array(frame[name])
        finite = raw[np.isfinite(raw)]
        median = float(np.median(finite)) if finite.size else 0.0
        clean = np.where(np.isfinite(raw), raw, median)
        mean = float(clean.mean())
        scale = float(clean.std())
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        medians[name], means[name], scales[name] = median, mean, scale
        output_names.append(name)
        if missing_indicators and (~np.isfinite(raw)).any():
            indicator_columns.append(name)
            output_names.append(f"{name}__missing")
    categories: dict[str, list[str]] = {}
    for name in categorical_columns:
        categories[name] = sorted(np.unique(_categorical_strings(frame[name])).tolist())
        output_names.extend([f"{name}=={value}" for value in categories[name]])
    if not output_names:
        raise InputValidationError("covariates contain no usable columns.")
    return FittedStructuredPreprocessor(
        input_kind="pandas_dataframe" if isinstance(covariates_train, pd.DataFrame) else "numeric_matrix",
        input_columns=list(frame.columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_medians=medians,
        numeric_means=means,
        numeric_scales=scales,
        missing_indicator_columns=indicator_columns,
        categories=categories,
        output_feature_names=output_names,
        preprocessed_numeric_matrix=False,
    )


def _build_mlp(input_dimension: int, architecture: tuple[int, ...], output_dimension: int) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    previous = input_dimension
    for width in architecture:
        layers.extend([torch.nn.Linear(previous, width), torch.nn.ReLU()])
        previous = width
    layers.append(torch.nn.Linear(previous, output_dimension))
    return torch.nn.Sequential(*layers)


@dataclass
class FittedNeuralHead:
    """A frozen CPU MLP represented by scaler state and NumPy weights."""

    task: str
    scaler: StandardScaler
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    selected_architecture: tuple[int, ...]
    validation_grid: list[dict[str, Any]]
    class_count: int | None
    training_seed: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = self.scaler.transform(np.asarray(features, dtype=float))
        hidden = values
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ weight.T + bias
            if index < len(self.weights) - 1:
                hidden = np.maximum(hidden, 0.0)
        if self.task == "regression":
            return np.asarray(hidden[:, 0], dtype=float)
        shifted = hidden - hidden.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return np.asarray(probabilities, dtype=float)

    def audit(self) -> dict[str, Any]:
        state = {
            "architecture": list(self.selected_architecture),
            "weights": [value.tolist() for value in self.weights],
            "biases": [value.tolist() for value in self.biases],
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        return {
            "preprocessing_fit_partition": "train",
            "optimizer": "AdamW",
            "activation": "ReLU",
            "early_stopping_partition": "validation",
            "architecture_selection_partition": "validation",
            "test_outcomes_used": False,
            "selected_architecture": list(self.selected_architecture),
            "training_seed": self.training_seed,
            "validation_grid": self.validation_grid,
            "fitted_state_sha256": _state_digest(state),
        }


@dataclass
class FittedRandomForestEnsemble:
    """A validation-selected 500-tree forest averaged over fixed fitting seeds."""

    task: str
    estimators: list[Any]
    selected_configuration: dict[str, Any]
    validation_grid: list[dict[str, Any]]
    seeds: tuple[int, ...]
    n_estimators: int = 500
    n_jobs: int | None = -1
    refit_estimators: list[Any] | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if self.task == "regression":
            return np.mean([model.predict(values) for model in self.estimators], axis=0)
        return np.mean([model.predict_proba(values) for model in self.estimators], axis=0)

    def refit(self, features: np.ndarray, outcome: np.ndarray) -> None:
        """Refit with locked configuration after validation decisions are frozen."""

        model_type: Any = RandomForestRegressor if self.task == "regression" else RandomForestClassifier
        self.refit_estimators = [
            model_type(
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                bootstrap=True,
                criterion="squared_error" if self.task == "regression" else "gini",
                random_state=seed,
                **self.selected_configuration,
            ).fit(np.asarray(features, dtype=float), np.asarray(outcome))
            for seed in self.seeds
        ]

    def predict_refit(self, features: np.ndarray) -> np.ndarray:
        if not self.refit_estimators:
            raise RuntimeError("Random forest has not been refit on train+validation rows.")
        values = np.asarray(features, dtype=float)
        if self.task == "regression":
            return np.mean([model.predict(values) for model in self.refit_estimators], axis=0)
        return np.mean([model.predict_proba(values) for model in self.refit_estimators], axis=0)

    def audit(self) -> dict[str, Any]:
        return {
            "model": "random_forest_ensemble",
            "n_estimators_per_seed": self.n_estimators,
            "n_jobs": self.n_jobs,
            "seeds": list(self.seeds),
            "selected_configuration": self.selected_configuration,
            "configuration_selection_partition": "validation",
            "fit_partition": "train",
            "test_prediction_refit_partition": (
                "train+validation after validation lock" if self.refit_estimators else None
            ),
            "test_outcomes_used": False,
            "validation_grid": self.validation_grid,
        }


def fit_random_forest_head(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    configurations: Iterable[dict[str, Any]] = DEFAULT_RF_CONFIGURATIONS,
    seeds: tuple[int, ...] = DEFAULT_RF_SEEDS,
    n_estimators: int = 500,
    n_jobs: int | None = -1,
) -> FittedRandomForestEnsemble:
    """Validation-select a source-compatible RF; no test arguments are accepted."""

    train = np.asarray(x_train, dtype=float)
    validation = np.asarray(x_validation, dtype=float)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise InputValidationError("Random-forest inputs must be aligned two-dimensional matrices.")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise InputValidationError("Random-forest inputs must be finite after train-fit preprocessing.")
    if not seeds or n_estimators < 1:
        raise InputValidationError("RF seeds and n_estimators must be nonempty and positive.")
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1
    ):
        raise InputValidationError("n_jobs must be None, -1, or a positive integer.")
    compute = metric_function(task)
    primary = "mse" if task == "regression" else "log_loss"
    candidates = list(configurations)
    if not candidates:
        raise InputValidationError("At least one random-forest configuration is required.")
    grid: list[dict[str, Any]] = []
    model_type: Any = RandomForestRegressor if task == "regression" else RandomForestClassifier
    for configuration in candidates:
        model = model_type(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            bootstrap=True,
            criterion="squared_error" if task == "regression" else "gini",
            random_state=seeds[0],
            **configuration,
        ).fit(train, y_train)
        prediction = model.predict(validation) if task == "regression" else model.predict_proba(validation)
        grid.append({
            **configuration,
            f"validation_{primary}": float(compute(y_validation, prediction)[primary]),
            "selection_seed": seeds[0],
        })
    best_index = int(np.argmin([row[f"validation_{primary}"] for row in grid]))
    selected = dict(candidates[best_index])
    estimators = [
        model_type(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            bootstrap=True,
            criterion="squared_error" if task == "regression" else "gini",
            random_state=seed,
            **selected,
        ).fit(train, y_train)
        for seed in seeds
    ]
    return FittedRandomForestEnsemble(
        task, estimators, selected, grid, seeds, n_estimators, n_jobs
    )


def _train_one_neural_candidate(
    task: str,
    train: np.ndarray,
    y_train: np.ndarray,
    validation: np.ndarray,
    y_validation: np.ndarray,
    architecture: tuple[int, ...],
    *,
    seed: int,
    max_iter: int,
    patience: int,
) -> tuple[FittedNeuralHead, float, int]:
    torch.manual_seed(seed)
    output_dimension = 1 if task == "regression" else int(np.max(y_train)) + 1
    model = _build_mlp(train.shape[1], architecture, output_dimension).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function: Any = torch.nn.MSELoss() if task == "regression" else torch.nn.CrossEntropyLoss()
    train_x = torch.as_tensor(train, dtype=torch.float32)
    validation_x = torch.as_tensor(validation, dtype=torch.float32)
    if task == "regression":
        train_y = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        validation_y = torch.as_tensor(y_validation, dtype=torch.float32).reshape(-1, 1)
    else:
        train_y = torch.as_tensor(y_train, dtype=torch.long)
        validation_y = torch.as_tensor(y_validation, dtype=torch.long)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    epochs = 0
    for epoch in range(max_iter):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y).item())
        epochs = epoch + 1
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    linear_layers = [module for module in model if isinstance(module, torch.nn.Linear)]
    weights = [module.weight.detach().cpu().numpy().astype(float, copy=True) for module in linear_layers]
    biases = [module.bias.detach().cpu().numpy().astype(float, copy=True) for module in linear_layers]
    fitted = FittedNeuralHead(
        task=task,
        scaler=StandardScaler(),
        weights=weights,
        biases=biases,
        selected_architecture=architecture,
        validation_grid=[],
        class_count=output_dimension if task == "classification" else None,
        training_seed=seed,
    )
    return fitted, best_loss, epochs


def fit_neural_head(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    architectures: Iterable[tuple[int, ...]] = DEFAULT_NN_ARCHITECTURES,
    random_state: int,
    max_iter: int = 300,
    patience: int = 25,
) -> FittedNeuralHead:
    """Select a modest AdamW MLP by validation loss; no test input is accepted."""

    train_raw = np.asarray(x_train, dtype=float)
    validation_raw = np.asarray(x_validation, dtype=float)
    if (
        train_raw.ndim != 2
        or validation_raw.ndim != 2
        or train_raw.shape[1] != validation_raw.shape[1]
        or not np.isfinite(train_raw).all()
        or not np.isfinite(validation_raw).all()
    ):
        raise InputValidationError("Neural-head inputs must be aligned finite two-dimensional matrices.")
    if max_iter < 1 or patience < 1:
        raise InputValidationError("Neural-head max_iter and patience must be positive.")
    scaler = StandardScaler().fit(train_raw)
    train = scaler.transform(train_raw)
    validation = scaler.transform(validation_raw)
    if task == "classification":
        train_classes = np.unique(np.asarray(y_train, dtype=int))
        all_classes = np.unique(np.concatenate([np.asarray(y_train, dtype=int), np.asarray(y_validation, dtype=int)]))
        if not np.array_equal(train_classes, all_classes) or not np.array_equal(train_classes, np.arange(len(train_classes))):
            raise InputValidationError("Neural classification training rows must contain every encoded class.")
    candidates: list[tuple[float, FittedNeuralHead]] = []
    grid: list[dict[str, Any]] = []
    compute = metric_function(task)
    primary = "mse" if task == "regression" else "log_loss"
    for index, raw_architecture in enumerate(architectures):
        architecture = tuple(int(width) for width in raw_architecture)
        if not architecture or len(architecture) > 2 or any(width < 1 for width in architecture):
            raise InputValidationError("Every NN architecture must contain one or two positive hidden widths.")
        fitted, validation_objective, epochs = _train_one_neural_candidate(
            task,
            train,
            np.asarray(y_train),
            validation,
            np.asarray(y_validation),
            architecture,
            seed=random_state + index,
            max_iter=max_iter,
            patience=patience,
        )
        fitted.scaler = scaler
        prediction = fitted.predict(validation_raw)
        validation_metric = float(compute(np.asarray(y_validation), prediction)[primary])
        candidates.append((validation_metric, fitted))
        grid.append(
            {
                "hidden_layer_sizes": list(architecture),
                f"validation_{primary}": validation_metric,
                "validation_training_objective": validation_objective,
                "epochs": epochs,
                "optimizer": "AdamW",
                "early_stopping_partition": "validation",
            }
        )
    if not candidates:
        raise InputValidationError("At least one NN architecture is required.")
    best = min(candidates, key=lambda item: item[0])[1]
    best.validation_grid = grid
    return best


def fit_structured_models(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    models: list[str],
    ridge_alphas: Iterable[float],
    logistic_cs: Iterable[float],
    random_state: int,
    nn_max_iter: int,
    rf_configurations: Iterable[dict[str, Any]] = DEFAULT_RF_CONFIGURATIONS,
    rf_seeds: tuple[int, ...] = DEFAULT_RF_SEEDS,
    rf_n_estimators: int = 500,
    rf_n_jobs: int | None = -1,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    """Fit and validation-select structured candidates without test arguments."""

    fitted: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    compute = metric_function(task)
    primary = "mse" if task == "regression" else "log_loss"
    if "linear" in models:
        head: FittedLinearHead = fit_linear_head(
            task,
            x_train,
            y_train,
            x_validation,
            y_validation,
            ridge_alphas=ridge_alphas,
            logistic_cs=logistic_cs,
            random_state=random_state,
        )
        prediction = head.predict(x_validation)
        fitted["linear"] = head
        records["linear"] = {
            "model": "ridge" if task == "regression" else "logistic",
            "validation_metrics": compute(y_validation, prediction),
            "selected_hyperparameters": head.selected_hyperparameter,
            "validation_grid": head.validation_grid,
            "decision_audit": fitted_linear_head_audit(head),
        }
    if "nn" in models:
        neural = fit_neural_head(
            task,
            x_train,
            y_train,
            x_validation,
            y_validation,
            random_state=random_state,
            max_iter=nn_max_iter,
        )
        prediction = neural.predict(x_validation)
        fitted["nn"] = neural
        records["nn"] = {
            "model": "small_adamw_mlp",
            "validation_metrics": compute(y_validation, prediction),
            "selected_hyperparameters": {"hidden_layer_sizes": list(neural.selected_architecture)},
            "validation_grid": neural.validation_grid,
            "decision_audit": neural.audit(),
        }
    if "rf" in models:
        forest = fit_random_forest_head(
            task,
            x_train,
            y_train,
            x_validation,
            y_validation,
            configurations=rf_configurations,
            seeds=rf_seeds,
            n_estimators=rf_n_estimators,
            n_jobs=rf_n_jobs,
        )
        prediction = forest.predict(x_validation)
        fitted["rf"] = forest
        records["rf"] = {
            "model": "random_forest_seed_averaged",
            "validation_metrics": compute(y_validation, prediction),
            "selected_hyperparameters": forest.selected_configuration,
            "validation_grid": forest.validation_grid,
            "decision_audit": forest.audit(),
        }
    selected = min(models, key=lambda name: records[name]["validation_metrics"][primary])
    return fitted, records, selected
