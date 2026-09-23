"""Orchestration for one validation-locked X/I/X+I comparison."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import InputValidationError
from .integration import fit_joint_integration, fit_score_integration
from .metrics import metric_function
from .rf_score import fit_rf_score
from .structured import (
    FittedStructuredPreprocessor,
    fit_structured_models,
    fit_structured_preprocessor,
)


def take_covariate_rows(covariates: Any, rows: np.ndarray) -> Any:
    indices = np.asarray(rows)
    if indices.dtype == bool:
        indices = np.flatnonzero(indices)
    if isinstance(covariates, pd.DataFrame):
        return covariates.iloc[indices].reset_index(drop=True)
    return np.asarray(covariates)[indices]


def joint_image_encoder_scope(
    image_source_method: str,
    selected_encoder: str,
    combination_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    """Match joint X+I features to the image information used by I."""

    if image_source_method == selected_encoder:
        return [selected_encoder]
    try:
        members = combination_metadata[image_source_method]["metadata"]["encoders"]
    except KeyError as error:
        raise InputValidationError(
            f"Missing encoder-membership metadata for image method {image_source_method!r}."
        ) from error
    normalized = [str(name) for name in members]
    if not normalized:
        raise InputValidationError(
            f"Image method {image_source_method!r} has no recorded encoder members."
        )
    return normalized


@dataclass
class FittedCovariateAnalysis:
    preprocessor: FittedStructuredPreprocessor
    structured_models: dict[str, Any]
    structured_records: dict[str, dict[str, Any]]
    selected_structured_model: str
    integration_models: dict[str, Any]
    integration_records: dict[str, dict[str, Any]]
    selected_integration_method: str
    structured_features: np.ndarray
    structured_predictions: dict[str, np.ndarray]
    integration_predictions: dict[str, np.ndarray]
    image_source_method: str
    image_feature_encoders: list[str]


def fit_covariate_analysis(
    covariates_train: Any,
    covariates_validation: Any,
    covariates_all: Any,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    training_groups: np.ndarray,
    validation_groups: np.ndarray,
    all_partition_labels: np.ndarray,
    image_validation_prediction: np.ndarray,
    image_all_prediction: np.ndarray,
    image_features_train: dict[str, np.ndarray],
    image_features_validation: dict[str, np.ndarray],
    image_features_all: dict[str, np.ndarray],
    *,
    task: str,
    structured_model_names: list[str],
    integration_method_names: list[str],
    image_source_method: str,
    image_feature_encoders: list[str],
    rf_score_encoder: str,
    missing_indicators: bool,
    preprocessed_numeric_matrix: bool,
    ridge_alphas: Iterable[float],
    logistic_cs: Iterable[float],
    random_state: int,
    score_alpha: float,
    score_c: float,
    joint_image_max_components: int,
    nn_max_iter: int,
    rf_structured_configurations: Iterable[dict[str, Any]],
    rf_score_configurations: Iterable[dict[str, Any]],
    rf_seeds: tuple[int, ...],
    rf_n_estimators: int,
    rf_score_folds: int,
    rf_score_image_max_components: int,
    rf_n_jobs: int | None,
    rf_score_reduction_random_state: int | None,
    rf_score_fold_random_state: int | None,
    rf_score_standardize_reduced_scores: bool,
    rf_score_refit_image_reduction: bool,
    rf_score_working_dtype: str,
) -> FittedCovariateAnalysis:
    """Fit every X and X+I decision without accepting locked-test outcomes."""

    preprocessor = fit_structured_preprocessor(
        covariates_train,
        missing_indicators=missing_indicators,
        preprocessed_numeric_matrix=preprocessed_numeric_matrix,
    )
    x_train = preprocessor.transform(covariates_train)
    x_validation = preprocessor.transform(covariates_validation)
    x_all = preprocessor.transform(covariates_all)
    structured_models, structured_records, selected_structured = fit_structured_models(
        task,
        x_train,
        y_train,
        x_validation,
        y_validation,
        models=structured_model_names,
        ridge_alphas=ridge_alphas,
        logistic_cs=logistic_cs,
        random_state=random_state,
        nn_max_iter=nn_max_iter,
        rf_configurations=rf_structured_configurations,
        rf_seeds=rf_seeds,
        rf_n_estimators=rf_n_estimators,
        rf_n_jobs=rf_n_jobs,
    )
    structured_predictions = {
        name: model.predict(x_all) for name, model in structured_models.items()
    }
    labels = np.asarray(all_partition_labels, dtype=str)
    train_mask = labels == "train"
    validation_mask = labels == "validation"
    test_mask = labels == "test"
    if not (
        int(train_mask.sum()) == len(x_train)
        and int(validation_mask.sum()) == len(x_validation)
        and set(labels) == {"train", "validation", "test"}
    ):
        raise InputValidationError("all_partition_labels do not align with the fitted partitions.")
    if "rf" in structured_models:
        forest = structured_models["rf"]
        forest.refit(
            np.vstack([x_train, x_validation]),
            np.concatenate([np.asarray(y_train), np.asarray(y_validation)]),
        )
        structured_predictions["rf"][test_mask] = forest.predict_refit(x_all[test_mask])
        structured_records["rf"]["decision_audit"] = forest.audit()
    selected_x_all = structured_predictions[selected_structured]
    selected_x_validation = structured_models[selected_structured].predict(x_validation)

    integration_models: dict[str, Any] = {}
    integration_records: dict[str, dict[str, Any]] = {}
    integration_predictions: dict[str, np.ndarray] = {}
    compute = metric_function(task)
    for offset, method in enumerate(integration_method_names):
        if method in {"linear_score", "nn_score"}:
            model = fit_score_integration(
                task,
                selected_x_validation,
                image_validation_prediction,
                y_validation,
                validation_groups=np.asarray(validation_groups, dtype=object),
                method=method,
                random_state=random_state + 101 + offset,
                alpha=score_alpha,
                c_value=score_c,
                nn_max_iter=nn_max_iter,
            )
            prediction_all = model.predict(selected_x_all, image_all_prediction)
            prediction_validation = model.predict(
                selected_x_validation, image_validation_prediction
            )
        elif method == "rf_score":
            if task != "regression":
                raise InputValidationError(
                    "rf_score is source-defined for regression only; no classification substitute is made."
                )
            if rf_score_encoder not in image_features_all:
                raise InputValidationError(
                    f"rf_score_encoder {rf_score_encoder!r} is not among the extracted representations."
                )
            model = fit_rf_score(
                x_train,
                y_train,
                training_groups,
                image_features_train[rf_score_encoder],
                x_validation,
                y_validation,
                image_features_validation[rf_score_encoder],
                x_all[test_mask],
                image_features_all[rf_score_encoder][test_mask],
                random_state=random_state + 307 + offset,
                rf_configurations=rf_score_configurations,
                rf_seeds=rf_seeds,
                n_estimators=rf_n_estimators,
                oof_folds=rf_score_folds,
                image_max_components=rf_score_image_max_components,
                reduction_random_state=rf_score_reduction_random_state,
                fold_random_state=rf_score_fold_random_state,
                standardize_reduced_scores=rf_score_standardize_reduced_scores,
                refit_image_reduction=rf_score_refit_image_reduction,
                working_dtype=rf_score_working_dtype,
                n_jobs=rf_n_jobs,
            )
            if task == "regression":
                prediction_all = np.empty(len(x_all), dtype=float)
            prediction_all[train_mask] = model.training_prediction
            prediction_all[validation_mask] = model.validation_prediction
            prediction_all[test_mask] = model.test_prediction
            prediction_validation = model.validation_prediction
        else:
            model = fit_joint_integration(
                task,
                x_train,
                image_features_train,
                y_train,
                x_validation,
                image_features_validation,
                y_validation,
                image_feature_encoders,
                method=method,
                max_components_per_encoder=joint_image_max_components,
                ridge_alphas=ridge_alphas,
                logistic_cs=logistic_cs,
                random_state=random_state + 211 + offset,
                nn_max_iter=nn_max_iter,
            )
            prediction_all = model.predict(x_all, image_features_all)
            prediction_validation = model.predict(x_validation, image_features_validation)
        integration_models[method] = model
        integration_predictions[method] = prediction_all
        integration_records[method] = {
            "validation_metrics": (
                model.selection_metrics
                if method == "nn_score" and model.selection_metrics is not None
                else compute(y_validation, prediction_validation)
            ),
            "metadata": (
                {**model.audit, "rf_score_encoder": rf_score_encoder}
                if method == "rf_score"
                else model.metadata()
            ),
        }
    primary = "mse" if task == "regression" else "log_loss"
    selected_integration = min(
        integration_method_names,
        key=lambda name: integration_records[name]["validation_metrics"][primary],
    )
    return FittedCovariateAnalysis(
        preprocessor=preprocessor,
        structured_models=structured_models,
        structured_records=structured_records,
        selected_structured_model=selected_structured,
        integration_models=integration_models,
        integration_records=integration_records,
        selected_integration_method=selected_integration,
        structured_features=x_all,
        structured_predictions=structured_predictions,
        integration_predictions=integration_predictions,
        image_source_method=image_source_method,
        image_feature_encoders=list(image_feature_encoders),
    )
