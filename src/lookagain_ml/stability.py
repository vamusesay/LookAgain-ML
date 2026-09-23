"""Repeated group-safe split evaluation and stability summaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .combination import fit_feature_concatenation, fit_validation_combinations
from .exceptions import InputValidationError
from .hashing import digest_strings
from .heads import fit_linear_head, fitted_linear_head_audit
from .metrics import metric_function
from .splitting import make_split, validate_partition_safety

if TYPE_CHECKING:
    from .checkpointing import CheckpointStore
    from .progress import ProgressReporter


def repeated_split_analysis(
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
    random_state: int,
    ridge_alphas: Iterable[float],
    logistic_cs: Iterable[float],
    stack_alpha: float,
    stack_c: float,
    feature_concatenation_max_components: int,
    head_max_components: int | None = None,
    head_pca_iterated_power: int = 3,
    logistic_max_iter: int = 2000,
    head_working_dtype: str = "float64",
    checkpoint_store: CheckpointStore | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Rerun the complete image-only selection workflow on fresh safe splits."""

    if repetitions < 2:
        raise InputValidationError("repeated_splits must be at least 2 when enabled.")
    compute = metric_function(task)
    selection_metric = "mse" if task == "regression" else "log_loss"
    performance_metric = "r2" if task == "regression" else "accuracy"
    rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    split_audits: list[dict[str, Any]] = []
    for split_index in range(repetitions):
        unit = f"repeated_image:split_{split_index:04d}"
        restored = checkpoint_store.load_unit(unit) if checkpoint_store else None
        if restored is not None:
            rows.extend(restored["rows"])
            assignments.extend(restored["assignments"])
            split_audits.append(restored["audit"])
            if progress_reporter is not None:
                progress_reporter.emit(
                    "REUSED",
                    "repeated_split",
                    "validated completed repeated split reused",
                    split_index=split_index,
                    current=split_index + 1,
                    total=repetitions,
                    details={"unit": unit},
                )
                progress_reporter.emit(
                    "SKIPPED",
                    "repeated_split_fitting",
                    "completed compatible split did not require refitting",
                    split_index=split_index,
                    details={"unit": unit},
                )
            continue
        if progress_reporter is not None:
            progress_reporter.emit(
                "STARTED",
                "repeated_split",
                "repeated image-only split started",
                split_index=split_index,
                current=split_index + 1,
                total=repetitions,
            )
        seed = random_state + 1009 * (split_index + 1)
        split = make_split(
            outcome, groups, image_hashes, task=task, random_state=seed
        )
        labels = split.labels
        validate_partition_safety(labels, groups, image_hashes)
        masks = {
            name: labels == name for name in ("train", "validation", "test")
        }
        if task == "classification" and set(outcome[masks["train"]]) != set(outcome):
            raise InputValidationError(
                f"Repeated split {split_index} training partition omitted a class. "
                "Use more independent groups or fewer repeated splits."
            )
        predictions: dict[str, np.ndarray] = {}
        validation_metrics: dict[str, dict[str, Any]] = {}
        head_audits: dict[str, dict[str, Any]] = {}
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
            predictions[name] = head.predict(features[name])
            validation_metrics[name] = compute(
                outcome[masks["validation"]], predictions[name][masks["validation"]]
            )
            head_audits[name] = {
                **fitted_linear_head_audit(head),
                "validation_grid": head.validation_grid,
            }
        selected_single = min(
            encoder_order,
            key=lambda name: validation_metrics[name][selection_metric],
        )
        validation_predictions = {
            name: prediction[masks["validation"]]
            for name, prediction in predictions.items()
        }
        prediction_methods = [
            name
            for name in combination_methods
            if name != "feature_concatenation"
        ]
        combinations, combination_audit = fit_validation_combinations(
            task,
            validation_predictions,
            outcome[masks["validation"]],
            encoder_order,
            prediction_methods,
            random_state=seed,
            stack_alpha=stack_alpha,
            stack_c=stack_c,
        )
        combination_predictions: dict[str, np.ndarray] = {}
        combination_validation_metrics: dict[str, dict[str, Any]] = {}
        for name, fitted in combinations.items():
            combination_predictions[name] = fitted.predict(predictions)
            combination_validation_metrics[name] = compute(
                outcome[masks["validation"]],
                combination_predictions[name][masks["validation"]],
            )
            combination_audit[name]["metadata"] = fitted.metadata()
        if "feature_concatenation" in combination_methods:
            fitted_union, union_audit = fit_feature_concatenation(
                task,
                {
                    name: features[name][masks["train"]] for name in encoder_order
                },
                outcome[masks["train"]],
                {
                    name: features[name][masks["validation"]]
                    for name in encoder_order
                },
                outcome[masks["validation"]],
                encoder_order,
                max_components_per_encoder=feature_concatenation_max_components,
                ridge_alphas=ridge_alphas,
                logistic_cs=logistic_cs,
                random_state=seed,
            )
            union_prediction = fitted_union.predict(features)
            combination_predictions["feature_concatenation"] = union_prediction
            combination_validation_metrics["feature_concatenation"] = compute(
                outcome[masks["validation"]],
                union_prediction[masks["validation"]],
            )
            combination_audit["feature_concatenation"] = {
                **union_audit,
                "metadata": fitted_union.metadata(),
            }
        decision_order = [selected_single, *combination_methods]
        decision_losses = {
            selected_single: validation_metrics[selected_single][selection_metric],
            **{
                name: combination_validation_metrics[name][selection_metric]
                for name in combination_methods
            },
        }
        preferred = min(decision_order, key=lambda name: decision_losses[name])

        # Decisions are frozen above. Test outcomes enter only descriptive metrics below.
        all_predictions = {**predictions, **combination_predictions}
        all_validation = {**validation_metrics, **combination_validation_metrics}
        for name, prediction in all_predictions.items():
            test_metrics = compute(
                outcome[masks["test"]], prediction[masks["test"]]
            )
            rows.append(
                {
                    "split_index": split_index,
                    "seed": seed,
                    "method": name,
                    "method_type": (
                        "encoder" if name in encoder_order else "combination"
                    ),
                    "validation_loss": float(
                        all_validation[name][selection_metric]
                    ),
                    "test_primary": float(test_metrics[performance_metric]),
                    "performance_metric": performance_metric,
                    "validation_selected_single": name == selected_single,
                    "validation_preferred_method": name == preferred,
                    "selected_single": selected_single,
                    "preferred_method": preferred,
                    "test_metrics": test_metrics,
                }
            )
        for observation_id, label in zip(observation_ids, labels):
            assignments.append(
                {
                    "split_index": split_index,
                    "seed": seed,
                    "observation_id": str(observation_id),
                    "split": str(label),
                }
            )
        split_audits.append(
            {
                "split_index": split_index,
                "seed": seed,
                "split_counts": {
                    name: int(masks[name].sum())
                    for name in ("train", "validation", "test")
                },
                "assignment_sha256": digest_strings(
                    [
                        f"{obs}|{label}"
                        for obs, label in zip(observation_ids, labels)
                    ]
                ),
                "selected_single": selected_single,
                "preferred_method": preferred,
                "group_split_overlap": 0,
                "hash_split_overlap": 0,
                "training_preprocessing_and_heads_refit": True,
                "head_hyperparameters_selected_on": "validation",
                "encoder_selected_on": "validation",
                "combination_decisions_selected_on": (
                    "validation" if combination_methods else None
                ),
                "head_decision_audits": head_audits,
                "combination_audit": combination_audit,
                "test_outcomes_used_for_decisions": False,
            }
        )
        if checkpoint_store is not None:
            path = checkpoint_store.save_unit(
                unit,
                {
                    "rows": [row for row in rows if row["split_index"] == split_index],
                    "assignments": [
                        row for row in assignments if row["split_index"] == split_index
                    ],
                    "audit": split_audits[-1],
                },
            )
            if progress_reporter is not None:
                progress_reporter.emit(
                    "CHECKPOINTED",
                    "repeated_split",
                    "completed repeated split checkpointed atomically",
                    split_index=split_index,
                    current=split_index + 1,
                    total=repetitions,
                    location=str(path),
                    details={"unit": unit},
                )
        if progress_reporter is not None:
            progress_reporter.emit(
                "COMPLETED",
                "repeated_split",
                "repeated image-only split completed",
                split_index=split_index,
                current=split_index + 1,
                total=repetitions,
                details={"unit": unit},
            )

    frame = pd.DataFrame(rows)
    frame["test_rank"] = frame.groupby("split_index")["test_primary"].rank(
        ascending=False, method="min"
    )
    summaries: list[dict[str, Any]] = []
    for name, group in frame.groupby("method", sort=False):
        values = group["test_primary"].to_numpy(float)
        summaries.append(
            {
                "method": name,
                "method_type": group["method_type"].iloc[0],
                "performance_metric": performance_metric,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "empirical_2_5_percentile": float(np.percentile(values, 2.5)),
                "empirical_97_5_percentile": float(np.percentile(values, 97.5)),
                "mean_test_rank": float(group["test_rank"].mean()),
                "test_rank_sd": float(group["test_rank"].std(ddof=1)),
                "descriptive_test_win_frequency": float(
                    np.mean(group["test_rank"] == 1)
                ),
                "validation_selection_frequency": float(
                    group["validation_selected_single"].mean()
                    if group["method_type"].iloc[0] == "encoder"
                    else 0.0
                ),
                "validation_preferred_method_frequency": float(
                    group["validation_preferred_method"].mean()
                ),
            }
        )
    stack_gains: list[float] = []
    if "linear_stack" in combination_methods:
        for split_index in range(repetitions):
            subset = frame[frame.split_index == split_index]
            selected = str(subset.selected_single.iloc[0])
            selected_value = float(
                subset.loc[subset.method == selected, "test_primary"].iloc[0]
            )
            stack_value = float(
                subset.loc[subset.method == "linear_stack", "test_primary"].iloc[0]
            )
            stack_gains.append(stack_value - selected_value)
    gain_summary = None
    if stack_gains:
        values = np.asarray(stack_gains)
        gain_summary = {
            "performance_metric": performance_metric,
            "orientation": (
                "linear_stack minus validation-selected single; positive favors stack"
            ),
            "values": stack_gains,
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "empirical_2_5_percentile": float(np.percentile(values, 2.5)),
            "empirical_97_5_percentile": float(np.percentile(values, 97.5)),
            "positive_frequency": float(np.mean(values > 0)),
        }
    return {
        "repetitions": repetitions,
        "split_percentiles_label": (
            "Empirical split-to-split variation; these percentiles are not confidence intervals."
        ),
        "performance_metric": performance_metric,
        "split_level_results": frame.to_dict(orient="records"),
        "summary": summaries,
        "validation_selection_frequency": {
            row["method"]: row["validation_selection_frequency"]
            for row in summaries
            if row["method_type"] == "encoder"
        },
        "rank_stability": {
            row["method"]: {
                "mean_test_rank": row["mean_test_rank"],
                "test_rank_sd": row["test_rank_sd"],
                "descriptive_test_win_frequency": row[
                    "descriptive_test_win_frequency"
                ],
            }
            for row in summaries
        },
        "stack_gain_distribution": gain_summary,
        "split_assignments": assignments,
        "split_audits": split_audits,
    }
