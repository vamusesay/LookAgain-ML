"""Repeated group-safe X/I/X+I analysis with complete decision refitting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .combination import fit_feature_concatenation, fit_validation_combinations
from .covariate_analysis import (
    fit_covariate_analysis,
    joint_image_encoder_scope,
    take_covariate_rows,
)
from .exceptions import InputValidationError
from .hashing import digest_strings
from .heads import fit_linear_head, fitted_linear_head_audit
from .integration import incremental_metrics
from .metrics import metric_function
from .splitting import make_split, validate_partition_safety
from .structured import DEFAULT_RF_CONFIGURATIONS, DEFAULT_RF_SEEDS

if TYPE_CHECKING:
    from .checkpointing import CheckpointStore
    from .progress import ProgressReporter


def repeated_xi_analysis(
    covariates: Any,
    features: dict[str, np.ndarray],
    outcome: np.ndarray,
    groups: np.ndarray,
    image_hashes: np.ndarray,
    observation_ids: np.ndarray,
    *,
    task: str,
    repetitions: int,
    encoder_order: list[str],
    combination_methods: list[str],
    structured_model_names: list[str],
    integration_method_names: list[str],
    missing_indicators: bool,
    preprocessed_numeric_matrix: bool,
    random_state: int,
    ridge_alphas: Iterable[float],
    logistic_cs: Iterable[float],
    stack_alpha: float,
    stack_c: float,
    feature_concatenation_max_components: int,
    score_alpha: float,
    score_c: float,
    joint_image_max_components: int,
    nn_max_iter: int,
    rf_score_encoder: str | None = None,
    rf_structured_configurations: Iterable[dict[str, Any]] = DEFAULT_RF_CONFIGURATIONS,
    rf_score_configurations: Iterable[dict[str, Any]] = DEFAULT_RF_CONFIGURATIONS,
    rf_seeds: tuple[int, ...] = DEFAULT_RF_SEEDS,
    rf_n_estimators: int = 500,
    rf_score_folds: int = 5,
    rf_score_image_max_components: int = 32,
    rf_n_jobs: int | None = -1,
    rf_score_reduction_random_state: int | None = None,
    rf_score_fold_random_state: int | None = None,
    rf_score_standardize_reduced_scores: bool = False,
    rf_score_refit_image_reduction: bool = True,
    rf_score_working_dtype: str = "float64",
    head_max_components: int | None = None,
    head_pca_iterated_power: int = 3,
    logistic_max_iter: int = 2000,
    head_working_dtype: str = "float64",
    checkpoint_store: CheckpointStore | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Rerun all image, X, and X+I decisions for every generated split."""

    if repetitions < 2:
        raise InputValidationError("repeated_splits must be at least 2 when enabled.")
    compute = metric_function(task)
    selection_metric = "mse" if task == "regression" else "log_loss"
    performance_metric = "r2" if task == "regression" else "accuracy"
    rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for split_index in range(repetitions):
        unit = f"repeated_xi:split_{split_index:04d}"
        restored = checkpoint_store.load_unit(unit) if checkpoint_store else None
        if restored is not None:
            rows.extend(restored["rows"])
            assignments.extend(restored["assignments"])
            audits.append(restored["audit"])
            selected_rows.append(restored["selected_row"])
            if progress_reporter is not None:
                progress_reporter.emit(
                    "REUSED",
                    "repeated_xi_split",
                    "validated completed X/I/X+I split reused",
                    split_index=split_index,
                    current=split_index + 1,
                    total=repetitions,
                    details={"unit": unit},
                )
                progress_reporter.emit(
                    "SKIPPED",
                    "repeated_xi_fitting",
                    "completed compatible split did not require refitting",
                    split_index=split_index,
                    details={"unit": unit},
                )
            continue
        if progress_reporter is not None:
            progress_reporter.emit(
                "STARTED",
                "repeated_xi_split",
                "repeated X/I/X+I split started",
                split_index=split_index,
                current=split_index + 1,
                total=repetitions,
            )
        seed = random_state + 1009 * (split_index + 1)
        split = make_split(outcome, groups, image_hashes, task=task, random_state=seed)
        labels = split.labels
        validate_partition_safety(labels, groups, image_hashes)
        masks = {name: labels == name for name in ("train", "validation", "test")}
        if task == "classification" and set(outcome[masks["train"]]) != set(outcome):
            raise InputValidationError(
                f"Repeated split {split_index} training partition omitted a class."
            )

        image_predictions: dict[str, np.ndarray] = {}
        image_validation_metrics: dict[str, dict[str, Any]] = {}
        image_head_audits: dict[str, Any] = {}
        for name in encoder_order:
            head = fit_linear_head(
                task,
                features[name][masks["train"]],
                outcome[masks["train"]],
                features[name][masks["validation"]],
                outcome[masks["validation"]],
                ridge_alphas=ridge_alphas,
                logistic_cs=logistic_cs,
                random_state=seed,
                max_components=head_max_components,
                pca_iterated_power=head_pca_iterated_power,
                logistic_max_iter=logistic_max_iter,
                working_dtype=head_working_dtype,
            )
            prediction = head.predict(features[name])
            image_predictions[name] = prediction
            image_validation_metrics[name] = compute(
                outcome[masks["validation"]], prediction[masks["validation"]]
            )
            image_head_audits[name] = {
                **fitted_linear_head_audit(head),
                "validation_grid": head.validation_grid,
            }
        selected_encoder = min(
            encoder_order,
            key=lambda name: image_validation_metrics[name][selection_metric],
        )
        prediction_methods = [
            name for name in combination_methods if name != "feature_concatenation"
        ]
        fitted_combinations, combination_audits = fit_validation_combinations(
            task,
            {
                name: prediction[masks["validation"]]
                for name, prediction in image_predictions.items()
            },
            outcome[masks["validation"]],
            encoder_order,
            prediction_methods,
            random_state=seed,
            stack_alpha=stack_alpha,
            stack_c=stack_c,
        )
        combination_predictions: dict[str, np.ndarray] = {}
        combination_validation_metrics: dict[str, dict[str, Any]] = {}
        combination_metadata: dict[str, Any] = {}
        for name, model in fitted_combinations.items():
            prediction = model.predict(image_predictions)
            combination_predictions[name] = prediction
            combination_validation_metrics[name] = compute(
                outcome[masks["validation"]], prediction[masks["validation"]]
            )
            combination_metadata[name] = {
                "metadata": model.metadata(),
                "validation_audit": combination_audits[name],
            }
        if "feature_concatenation" in combination_methods:
            model, combination_audit = fit_feature_concatenation(
                task,
                {name: value[masks["train"]] for name, value in features.items()},
                outcome[masks["train"]],
                {name: value[masks["validation"]] for name, value in features.items()},
                outcome[masks["validation"]],
                encoder_order,
                max_components_per_encoder=feature_concatenation_max_components,
                ridge_alphas=ridge_alphas,
                logistic_cs=logistic_cs,
                random_state=seed,
            )
            prediction = model.predict(features)
            combination_predictions["feature_concatenation"] = prediction
            combination_validation_metrics["feature_concatenation"] = compute(
                outcome[masks["validation"]], prediction[masks["validation"]]
            )
            combination_metadata["feature_concatenation"] = {
                "metadata": model.metadata(),
                "validation_audit": combination_audit,
            }
        image_candidates = {
            selected_encoder: image_validation_metrics[selected_encoder],
            **combination_validation_metrics,
        }
        image_order = [selected_encoder, *combination_methods]
        image_source = min(
            image_order, key=lambda name: image_candidates[name][selection_metric]
        )
        image_source_prediction = (
            image_predictions[selected_encoder]
            if image_source == selected_encoder
            else combination_predictions[image_source]
        )
        joint_feature_encoders = joint_image_encoder_scope(
            image_source, selected_encoder, combination_metadata
        )

        fitted = fit_covariate_analysis(
            take_covariate_rows(covariates, masks["train"]),
            take_covariate_rows(covariates, masks["validation"]),
            covariates,
            outcome[masks["train"]],
            outcome[masks["validation"]],
            np.asarray(split.effective_groups, dtype=object)[masks["train"]],
            np.asarray(split.effective_groups, dtype=object)[masks["validation"]],
            labels,
            image_source_prediction[masks["validation"]],
            image_source_prediction,
            {name: value[masks["train"]] for name, value in features.items()},
            {name: value[masks["validation"]] for name, value in features.items()},
            features,
            task=task,
            structured_model_names=structured_model_names,
            integration_method_names=integration_method_names,
            image_source_method=image_source,
            image_feature_encoders=joint_feature_encoders,
            rf_score_encoder=rf_score_encoder or selected_encoder,
            missing_indicators=missing_indicators,
            preprocessed_numeric_matrix=preprocessed_numeric_matrix,
            ridge_alphas=ridge_alphas,
            logistic_cs=logistic_cs,
            random_state=seed,
            score_alpha=score_alpha,
            score_c=score_c,
            joint_image_max_components=joint_image_max_components,
            nn_max_iter=nn_max_iter,
            rf_structured_configurations=rf_structured_configurations,
            rf_score_configurations=rf_score_configurations,
            rf_seeds=rf_seeds,
            rf_n_estimators=rf_n_estimators,
            rf_score_folds=rf_score_folds,
            rf_score_image_max_components=rf_score_image_max_components,
            rf_n_jobs=rf_n_jobs,
            rf_score_reduction_random_state=rf_score_reduction_random_state,
            rf_score_fold_random_state=rf_score_fold_random_state,
            rf_score_standardize_reduced_scores=rf_score_standardize_reduced_scores,
            rf_score_refit_image_reduction=rf_score_refit_image_reduction,
            rf_score_working_dtype=rf_score_working_dtype,
        )
        selected_x_prediction = fitted.structured_predictions[fitted.selected_structured_model]
        y_test = outcome[masks["test"]]
        x_metrics = compute(y_test, selected_x_prediction[masks["test"]])
        i_metrics = compute(y_test, image_source_prediction[masks["test"]])
        integration_test_metrics = {
            name: compute(y_test, prediction[masks["test"]])
            for name, prediction in fitted.integration_predictions.items()
        }
        test_best = (
            max(
                integration_method_names,
                key=lambda name: integration_test_metrics[name][performance_metric],
            )
            if performance_metric != "log_loss"
            else min(
                integration_method_names,
                key=lambda name: integration_test_metrics[name][performance_metric],
            )
        )
        for name in integration_method_names:
            metrics = integration_test_metrics[name]
            increment = incremental_metrics(task, x_metrics, metrics)
            rows.append(
                {
                    "split_index": split_index,
                    "seed": seed,
                    "method": name,
                    "validation_selected": name == fitted.selected_integration_method,
                    "validation_loss": fitted.integration_records[name]["validation_metrics"][selection_metric],
                    "test_primary": metrics[performance_metric],
                    "structured_test_primary": x_metrics[performance_metric],
                    "image_test_primary": i_metrics[performance_metric],
                    "increment": increment["delta_r2"] if task == "regression" else increment["delta_accuracy"],
                }
            )
        selected_metrics = integration_test_metrics[fitted.selected_integration_method]
        selected_increment = incremental_metrics(task, x_metrics, selected_metrics)
        selected_rows.append(
            {
                "split_index": split_index,
                "selected_structured_model": fitted.selected_structured_model,
                "selected_image_method": image_source,
                "joint_image_feature_encoders": fitted.image_feature_encoders,
                "selected_integration_method": fitted.selected_integration_method,
                "test_best_integration_method_descriptive": test_best,
                "validation_selection_hit_descriptive": fitted.selected_integration_method == test_best,
                "structured_test_primary": x_metrics[performance_metric],
                "image_test_primary": i_metrics[performance_metric],
                "combined_test_primary": selected_metrics[performance_metric],
                "increment": selected_increment["delta_r2"] if task == "regression" else selected_increment["delta_accuracy"],
            }
        )
        audits.append(
            {
                "split_index": split_index,
                "seed": seed,
                "selected_encoder": selected_encoder,
                "selected_image_method": image_source,
                "joint_image_feature_encoders": fitted.image_feature_encoders,
                "image_head_decisions": image_head_audits,
                "image_combination_decisions": combination_metadata,
                "structured_preprocessing": fitted.preprocessor.metadata(),
                "structured_model_decisions": {
                    name: {
                        "decision_audit": record["decision_audit"],
                        "validation_grid": record["validation_grid"],
                        "validation_metrics": record["validation_metrics"],
                    }
                    for name, record in fitted.structured_records.items()
                },
                "selected_structured_model": fitted.selected_structured_model,
                "integration_decisions": {
                    name: {
                        "metadata": record["metadata"],
                        "validation_metrics": record["validation_metrics"],
                    }
                    for name, record in fitted.integration_records.items()
                },
                "selected_integration_method": fitted.selected_integration_method,
                "test_outcomes_used_for_decisions": False,
                "split_assignment_sha256": digest_strings(
                    (
                        pd.Series(observation_ids).astype(str)
                        + "|"
                        + pd.Series(labels).astype(str)
                    ).tolist()
                ),
            }
        )
        assignments.extend(
            {
                "split_index": split_index,
                "seed": seed,
                "observation_id": str(observation_ids[row]),
                "split": str(labels[row]),
            }
            for row in range(len(labels))
        )
        if checkpoint_store is not None:
            path = checkpoint_store.save_unit(
                unit,
                {
                    "rows": [row for row in rows if row["split_index"] == split_index],
                    "assignments": [
                        row for row in assignments if row["split_index"] == split_index
                    ],
                    "audit": audits[-1],
                    "selected_row": selected_rows[-1],
                },
            )
            if progress_reporter is not None:
                progress_reporter.emit(
                    "CHECKPOINTED",
                    "repeated_xi_split",
                    "completed X/I/X+I split checkpointed atomically",
                    split_index=split_index,
                    current=split_index + 1,
                    total=repetitions,
                    location=str(path),
                    details={"unit": unit},
                )
        if progress_reporter is not None:
            progress_reporter.emit(
                "COMPLETED",
                "repeated_xi_split",
                "repeated X/I/X+I split completed",
                split_index=split_index,
                current=split_index + 1,
                total=repetitions,
                details={"unit": unit},
            )

    selected_frame = pd.DataFrame(selected_rows)
    values = selected_frame["increment"].to_numpy(dtype=float)
    method_frame = pd.DataFrame(rows)
    method_frame["test_rank"] = method_frame.groupby("split_index")["test_primary"].rank(
        ascending=False, method="min"
    )
    method_summary = []
    for name, subset in method_frame.groupby("method", sort=False):
        method_values = subset["increment"].to_numpy(dtype=float)
        method_summary.append(
            {
                "method": name,
                "mean_increment": float(method_values.mean()),
                "sd_increment": float(method_values.std(ddof=1)) if len(method_values) > 1 else 0.0,
                "empirical_2_5_percentile": float(np.percentile(method_values, 2.5)),
                "empirical_97_5_percentile": float(np.percentile(method_values, 97.5)),
                "positive_frequency": float(np.mean(method_values > 0)),
                "validation_selection_frequency": float(subset["validation_selected"].mean()),
                "mean_test_rank": float(subset["test_rank"].mean()),
                "test_rank_sd": (
                    float(subset["test_rank"].std(ddof=1)) if len(subset) > 1 else 0.0
                ),
                "descriptive_test_win_frequency": float(np.mean(subset["test_rank"] == 1)),
            }
        )
    information_columns = {
        "X_structured_only": "structured_test_primary",
        "I_image_only": "image_test_primary",
        "X_plus_I_selected": "combined_test_primary",
    }
    information_frame = selected_frame[["split_index", *information_columns.values()]].melt(
        id_vars="split_index", var_name="information_column", value_name="test_primary"
    )
    column_to_name = {column: name for name, column in information_columns.items()}
    information_frame["information_set"] = information_frame["information_column"].map(
        column_to_name
    )
    information_frame["test_rank"] = information_frame.groupby("split_index")[
        "test_primary"
    ].rank(ascending=False, method="min")
    information_set_summary = []
    for name in information_columns:
        subset = information_frame[information_frame["information_set"] == name]
        information_values = subset["test_primary"].to_numpy(dtype=float)
        information_set_summary.append(
            {
                "information_set": name,
                "performance_metric": performance_metric,
                "mean": float(information_values.mean()),
                "sd": (
                    float(information_values.std(ddof=1))
                    if len(information_values) > 1
                    else 0.0
                ),
                "empirical_2_5_percentile": float(np.percentile(information_values, 2.5)),
                "empirical_97_5_percentile": float(np.percentile(information_values, 97.5)),
                "mean_test_rank": float(subset["test_rank"].mean()),
                "test_rank_sd": (
                    float(subset["test_rank"].std(ddof=1)) if len(subset) > 1 else 0.0
                ),
                "descriptive_test_win_frequency": float(np.mean(subset["test_rank"] == 1)),
            }
        )
    return {
        "repetitions": repetitions,
        "performance_metric": performance_metric,
        "increment_metric": "delta_r2" if task == "regression" else "delta_accuracy",
        "split_percentiles_label": (
            "Empirical split-to-split X/I/X+I variation; these percentiles are not confidence intervals."
        ),
        "selected_procedure_summary": {
            "mean_increment": float(values.mean()),
            "sd_increment": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "empirical_2_5_percentile": float(np.percentile(values, 2.5)),
            "empirical_97_5_percentile": float(np.percentile(values, 97.5)),
            "positive_frequency": float(np.mean(values > 0)),
            "validation_selection_hit_frequency_descriptive": float(
                selected_frame["validation_selection_hit_descriptive"].mean()
            ),
        },
        "information_set_summary": information_set_summary,
        "information_set_rank_stability": {
            row["information_set"]: {
                "mean_test_rank": row["mean_test_rank"],
                "test_rank_sd": row["test_rank_sd"],
                "descriptive_test_win_frequency": row[
                    "descriptive_test_win_frequency"
                ],
            }
            for row in information_set_summary
        },
        "method_summary": method_summary,
        "integration_method_rank_stability": {
            row["method"]: {
                "mean_test_rank": row["mean_test_rank"],
                "test_rank_sd": row["test_rank_sd"],
                "descriptive_test_win_frequency": row[
                    "descriptive_test_win_frequency"
                ],
            }
            for row in method_summary
        },
        "selected_split_results": selected_frame.to_dict(orient="records"),
        "method_split_results": method_frame.to_dict(orient="records"),
        "split_assignments": assignments,
        "decision_audits": audits,
        "test_outcomes_used_for_decisions": False,
    }
