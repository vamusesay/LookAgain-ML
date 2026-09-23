"""Structured, auditable results for the implemented LookAgain-ML stages."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._version import RESULT_SCHEMA_VERSION, __version__


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _shareable_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Remove source references from failure records before serialization."""

    shareable = dict(audit)
    shareable["failures"] = [
        {
            "source_position": item.get("source_position"),
            "reason": item.get("reason"),
        }
        for item in audit.get("failures", [])
    ]
    return shareable


def _token_digest(value: Any) -> str:
    """Return a stable anonymized digest for a potentially identifying token."""

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass
class LookAgainResults:
    """Machine-readable outputs from one validation-locked analysis.

    Stage 4 can additionally populate structured-covariate and X/I/X+I
    comparisons while preserving the image-only result surface.
    """

    task: str
    selected_encoder: str
    encoder_results: dict[str, Any]
    metrics: dict[str, dict[str, float | None]]
    selected_hyperparameters: dict[str, Any]
    manifest: pd.DataFrame
    predictions: pd.DataFrame
    audit: dict[str, Any]
    qc: dict[str, Any]
    costs: dict[str, Any]
    configuration: dict[str, Any]
    classes: list[Any] | None = None
    fitted_head: Any = field(default=None, repr=False)
    stack_results: Any = None
    covariate_results: Any = None
    integration_results: Any = None
    repeated_split_results: Any = None
    uncertainty: Any = None
    cache_metadata: Any = None
    logme_results: Any = None
    method_predictions: dict[str, np.ndarray] | None = field(default=None, repr=False)
    fitted_covariate_preprocessor: Any = field(default=None, repr=False)
    xi_predictions: dict[str, np.ndarray] | None = field(default=None, repr=False)
    representation_features: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def summary(self) -> pd.DataFrame:
        """Return one concise metric row per evaluated partition."""

        return pd.DataFrame(
            [{"partition": partition, **values} for partition, values in self.metrics.items()]
        ).set_index("partition")

    def cost_summary(self) -> pd.DataFrame:
        """Return one compact row per encoder joining performance and run cost."""

        primary_metric = "mse" if self.task == "regression" else "log_loss"
        rows = []
        for name, result in self.encoder_results.items():
            cache = result.get("cache", {})
            rows.append(
                {
                    "encoder": name,
                    "selected": name == self.selected_encoder,
                    "feature_dimension": result["metadata"]["feature_dimension"],
                    "validation_metric": primary_metric,
                    "validation_metric_value": result["validation_metrics"][primary_metric],
                    "test_metric": primary_metric,
                    "test_metric_value": result["test_metrics"][primary_metric],
                    "cache_status": cache.get("status"),
                    "cache_bytes": cache.get("cache_bytes"),
                    **result.get("costs", {}),
                }
            )
        return pd.DataFrame(rows)

    def comparison(self) -> pd.DataFrame:
        """Return the validation-selection comparison with descriptive locked-test metrics."""

        rows: list[dict[str, Any]] = []
        for name, result in self.encoder_results.items():
            metadata = result["metadata"]
            cache = result.get("cache", {})
            row: dict[str, Any] = {
                "encoder": name,
                "selected": name == self.selected_encoder,
                "checkpoint": metadata["checkpoint"],
                "checkpoint_revision": metadata.get("checkpoint_revision"),
                "feature_dimension": metadata["feature_dimension"],
                "pooling": metadata["pooling"],
                "cache_status": cache.get("status"),
            }
            row.update(
                {f"validation_{key}": value for key, value in result["validation_metrics"].items()}
            )
            row.update(
                {f"test_{key}": value for key, value in result["test_metrics"].items()}
            )
            rows.append(row)
        return pd.DataFrame(rows).set_index("encoder")

    def plot_encoder_performance(
        self,
        path: str | Path | None = None,
        *,
        partition: str = "validation",
    ):
        """Plot primary loss by encoder; validation is the selection view."""

        if partition not in {"validation", "test"}:
            raise ValueError("partition must be 'validation' or 'test'.")
        from matplotlib.figure import Figure

        primary_metric = "mse" if self.task == "regression" else "log_loss"
        names = list(self.encoder_results)
        values = [
            self.encoder_results[name][f"{partition}_metrics"][primary_metric]
            for name in names
        ]
        colors = ["#1f77b4" if name == self.selected_encoder else "#b9c1ca" for name in names]
        figure = Figure(figsize=(max(6.0, 1.25 * len(names)), 4.0))
        axes = figure.subplots()
        axes.bar(names, values, color=colors)
        axes.set_ylabel(primary_metric.replace("_", " ").upper())
        suffix = "selection partition" if partition == "validation" else "descriptive locked test"
        axes.set_title(f"{partition.title()} encoder comparison ({suffix}; lower is better)")
        axes.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=160, bbox_inches="tight")
        return axes

    def plot_cost_performance(self, path: str | Path | None = None):
        """Plot validation loss against observed representation-extraction time.

        Cache-hit runs use the recorded cache-creation extraction time when it
        is available. Missing timings are omitted rather than treated as zero.
        """

        from matplotlib.figure import Figure

        primary_metric = "mse" if self.task == "regression" else "log_loss"
        points: list[tuple[str, float, float, bool]] = []
        for name, result in self.encoder_results.items():
            costs = result.get("costs", {})
            seconds = costs.get("extraction_seconds")
            if not seconds:
                seconds = costs.get("cached_creation_extraction_seconds")
            if seconds is None or not np.isfinite(seconds):
                continue
            points.append(
                (
                    name,
                    float(seconds),
                    float(result["validation_metrics"][primary_metric]),
                    name == self.selected_encoder,
                )
            )
        if not points:
            raise ValueError("No encoder extraction timing is available for a cost/performance plot.")
        figure = Figure(figsize=(6.8, 4.4))
        axes = figure.subplots()
        for name, seconds, loss, selected in points:
            axes.scatter(
                [seconds],
                [loss],
                color="#1d4ed8" if selected else "#9ca3af",
                s=70 if selected else 50,
            )
            axes.annotate(name, (seconds, loss), xytext=(5, 4), textcoords="offset points")
        axes.set_xlabel("Observed representation extraction time (seconds)")
        axes.set_ylabel(f"Validation {primary_metric.replace('_', ' ')} (lower is better)")
        axes.set_title("Representation cost/performance (this run or verified cache origin)")
        figure.tight_layout()
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=160, bbox_inches="tight")
        return axes

    def plot_split_stability(self, path: str | Path | None = None):
        """Plot empirical split-to-split variation, not confidence intervals."""

        if not self.repeated_split_results:
            raise ValueError("No repeated-split results are available.")
        from matplotlib.figure import Figure

        frame = pd.DataFrame(self.repeated_split_results["split_level_results"])
        names = list(dict.fromkeys(frame["method"].tolist()))
        values = [
            frame.loc[frame.method == name, "test_primary"].to_numpy()
            for name in names
        ]
        figure = Figure(figsize=(max(7.0, 1.2 * len(names)), 4.5))
        axes = figure.subplots()
        axes.boxplot(values, showmeans=True)
        axes.set_xticks(np.arange(1, len(names) + 1), names)
        axes.set_ylabel(self.repeated_split_results["performance_metric"].upper())
        axes.set_title("Empirical split-to-split variation (not confidence intervals)")
        axes.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=160, bbox_inches="tight")
        return axes

    def plot_stack_gains(self, path: str | Path | None = None):
        """Plot linear-stack gain across repeated splits."""

        gain = (self.repeated_split_results or {}).get("stack_gain_distribution")
        if not gain:
            raise ValueError("No repeated-split linear-stack gains are available.")
        from matplotlib.figure import Figure

        values = np.asarray(gain["values"], dtype=float)
        figure = Figure(figsize=(6.5, 4.0))
        axes = figure.subplots()
        axes.axhline(0.0, color="#555555", linewidth=1)
        axes.scatter(np.arange(1, len(values) + 1), values, color="#1f77b4")
        axes.set_xlabel("Repeated split")
        axes.set_ylabel(
            f"Stack - validation-selected single ({gain['performance_metric']})"
        )
        axes.set_title("Stack gain across splits (variation, not confidence intervals)")
        figure.tight_layout()
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=160, bbox_inches="tight")
        return axes

    def plot_x_i_xi(self, path: str | Path | None = None):
        """Plot locked-test X, I, and validation-selected X+I performance."""

        if not self.integration_results:
            raise ValueError("No covariate integration results are available.")
        from matplotlib.figure import Figure

        comparison = self.integration_results["x_i_x_plus_i"]
        metric = "r2" if self.task == "regression" else "accuracy"
        names = ["X structured", "I image", "X+I selected"]
        values = [
            comparison["X_structured_only"][metric],
            comparison["I_image_only"][metric],
            comparison["X_plus_I_selected"][metric],
        ]
        figure = Figure(figsize=(6.8, 4.2))
        axes = figure.subplots()
        axes.bar(names, values, color=["#6b7280", "#60a5fa", "#1d4ed8"])
        axes.set_ylabel(metric.upper())
        axes.set_title("Locked-test information sets (X+I method selected on validation)")
        axes.tick_params(axis="x", rotation=12)
        figure.tight_layout()
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=160, bbox_inches="tight")
        return axes

    def to_dict(self) -> dict[str, Any]:
        """Return the shareable configuration/results payload without source paths."""

        return _json_safe(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "package_version": __version__,
                "task": self.task,
                "selected_encoder": self.selected_encoder,
                "encoder_results": self.encoder_results,
                "metrics": self.metrics,
                "selected_hyperparameters": self.selected_hyperparameters,
                "configuration": self.configuration,
                "audit": _shareable_audit(self.audit),
                "qc": self.qc,
                "costs": self.costs,
                "classes": self.classes,
                "stack_results": self.stack_results,
                "covariate_results": self.covariate_results,
                "integration_results": self.integration_results,
                "repeated_split_results": self.repeated_split_results,
                "uncertainty": self.uncertainty,
                "cache_metadata": self.cache_metadata,
                "logme_results": self.logme_results,
            }
        )

    def save(
        self,
        directory: str | Path,
        *,
        include_predictions: bool = True,
        include_manifest: bool = True,
    ) -> dict[str, Path]:
        """Save documented JSON/CSV outputs with anonymized source references."""

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}
        results_path = output / "results.json"
        results_path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )
        files["results"] = results_path

        summary_path = output / "metrics.csv"
        self.summary().reset_index().to_csv(summary_path, index=False)
        files["metrics"] = summary_path

        comparison_path = output / "encoder_comparison.csv"
        self.comparison().reset_index().to_csv(comparison_path, index=False)
        files["encoder_comparison"] = comparison_path

        if self.stack_results:
            rows = []
            for name, result in self.stack_results["methods"].items():
                row = {"method": name, **result["metadata"]}
                row.update(
                    {
                        f"validation_{key}": value
                        for key, value in result["validation_metrics"].items()
                    }
                )
                row.update(
                    {
                        f"test_{key}": value
                        for key, value in result["test_metrics"].items()
                    }
                )
                rows.append(row)
            path = output / "combination_comparison.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            files["combination_comparison"] = path
            if include_predictions and self.method_predictions:
                prediction_rows: dict[str, Any] = {
                    "observation_id": self.manifest["observation_id"].to_numpy(),
                    "split": self.manifest["split"].to_numpy(),
                }
                for name, prediction in self.method_predictions.items():
                    array = np.asarray(prediction)
                    if array.ndim == 1:
                        prediction_rows[f"{name}__prediction"] = array
                    else:
                        for class_index in range(array.shape[1]):
                            prediction_rows[
                                f"{name}__probability_class_{class_index}"
                            ] = array[:, class_index]
                path = output / "combination_predictions.csv"
                pd.DataFrame(prediction_rows).to_csv(path, index=False)
                files["combination_predictions"] = path
        if self.repeated_split_results:
            level_path = output / "repeated_split_results.csv"
            pd.DataFrame(
                self.repeated_split_results["split_level_results"]
            ).drop(columns=["test_metrics"], errors="ignore").to_csv(
                level_path, index=False
            )
            files["repeated_split_results"] = level_path
            repeated_summary_path = output / "repeated_split_summary.csv"
            pd.DataFrame(self.repeated_split_results["summary"]).to_csv(
                repeated_summary_path, index=False
            )
            files["repeated_split_summary"] = repeated_summary_path
            assignment_path = output / "repeated_split_assignments.csv"
            pd.DataFrame(self.repeated_split_results["split_assignments"]).to_csv(
                assignment_path, index=False
            )
            files["repeated_split_assignments"] = assignment_path
            repeated_audit_path = output / "repeated_split_audit.json"
            repeated_audit_path.write_text(
                json.dumps(
                    _json_safe(self.repeated_split_results["split_audits"]),
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            files["repeated_split_audit"] = repeated_audit_path
            xi_repeated = self.repeated_split_results.get("x_i_x_plus_i")
            if xi_repeated:
                path = output / "repeated_xi_selected_results.csv"
                pd.DataFrame(xi_repeated["selected_split_results"]).to_csv(path, index=False)
                files["repeated_xi_selected_results"] = path
                path = output / "repeated_xi_method_results.csv"
                pd.DataFrame(xi_repeated["method_split_results"]).to_csv(path, index=False)
                files["repeated_xi_method_results"] = path
                path = output / "repeated_xi_summary.json"
                path.write_text(
                    json.dumps(
                        _json_safe(
                            {
                                "terminology": xi_repeated["split_percentiles_label"],
                                "selected_procedure_summary": xi_repeated[
                                    "selected_procedure_summary"
                                ],
                                "information_set_summary": xi_repeated[
                                    "information_set_summary"
                                ],
                                "information_set_rank_stability": xi_repeated[
                                    "information_set_rank_stability"
                                ],
                                "method_summary": xi_repeated["method_summary"],
                                "integration_method_rank_stability": xi_repeated[
                                    "integration_method_rank_stability"
                                ],
                            }
                        ),
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                files["repeated_xi_summary"] = path
                path = output / "repeated_xi_decision_audit.json"
                path.write_text(
                    json.dumps(
                        _json_safe(xi_repeated["decision_audits"]),
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                files["repeated_xi_decision_audit"] = path
        if self.uncertainty:
            rows = []
            for comparison, metrics in self.uncertainty[
                "difference_intervals"
            ].items():
                for metric, values in metrics.items():
                    rows.append(
                        {"comparison": comparison, "metric": metric, **values}
                    )
            path = output / "paired_bootstrap_intervals.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            files["paired_bootstrap_intervals"] = path
        if self.logme_results:
            path = output / "logme_scores.csv"
            pd.DataFrame(
                [
                    {"encoder": name, "logme_score": score}
                    for name, score in self.logme_results["scores"].items()
                ]
            ).to_csv(path, index=False)
            files["logme_scores"] = path
        if self.covariate_results:
            rows = []
            for name, result in self.covariate_results["models"].items():
                row = {
                    "model": name,
                    "validation_selected": result["validation_selected"],
                    **{
                        f"validation_{key}": value
                        for key, value in result["validation_metrics"].items()
                    },
                    **{
                        f"test_{key}": value
                        for key, value in result["test_metrics"].items()
                    },
                }
                rows.append(row)
            path = output / "structured_model_comparison.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            files["structured_model_comparison"] = path
            path = output / "structured_preprocessing.json"
            path.write_text(
                json.dumps(
                    _json_safe(self.covariate_results["preprocessing"]),
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            files["structured_preprocessing"] = path
        if self.integration_results:
            rows = []
            for name, result in self.integration_results["methods"].items():
                row = {
                    "method": name,
                    "validation_selected": result["validation_selected"],
                    **{
                        f"validation_{key}": value
                        for key, value in result["validation_metrics"].items()
                    },
                    **{
                        f"test_{key}": value
                        for key, value in result["test_metrics"].items()
                    },
                    **result["incremental_metrics_vs_structured"],
                }
                rows.append(row)
            path = output / "integration_comparison.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            files["integration_comparison"] = path
            if include_predictions and self.xi_predictions:
                prediction_rows: dict[str, Any] = {
                    "observation_id": self.manifest["observation_id"].to_numpy(),
                    "split": self.manifest["split"].to_numpy(),
                }
                for name, prediction in self.xi_predictions.items():
                    array = np.asarray(prediction)
                    if array.ndim == 1:
                        prediction_rows[f"{name}__prediction"] = array
                    else:
                        for class_index in range(array.shape[1]):
                            prediction_rows[
                                f"{name}__probability_class_{class_index}"
                            ] = array[:, class_index]
                path = output / "xi_predictions.csv"
                pd.DataFrame(prediction_rows).to_csv(path, index=False)
                files["xi_predictions"] = path
            path = output / "stage4_decision_audit.json"
            path.write_text(
                json.dumps(
                    _json_safe(
                        {
                            "qc": self.qc,
                            "preprocessing": self.covariate_results["preprocessing"],
                            "structured_selection": {
                                "selected_model": self.covariate_results["selected_model"],
                                "selection_partition": "validation",
                                "models": {
                                    name: {
                                        "selected_hyperparameters": result[
                                            "selected_hyperparameters"
                                        ],
                                        "validation_grid": result["validation_grid"],
                                        "decision_audit": result["decision_audit"],
                                        "validation_metrics": result["validation_metrics"],
                                    }
                                    for name, result in self.covariate_results["models"].items()
                                },
                            },
                            "integration_selection": {
                                "selected_method": self.integration_results["selected_method"],
                                "selection_partition": "validation",
                                "image_score_source": self.integration_results[
                                    "image_score_source"
                                ],
                                "methods": {
                                    name: {
                                        "metadata": result["metadata"],
                                        "validation_metrics": result["validation_metrics"],
                                    }
                                    for name, result in self.integration_results["methods"].items()
                                },
                            },
                            "test_outcomes_used_for_decisions": False,
                            "scientific_guardrails": self.integration_results[
                                "scientific_guardrails"
                            ],
                        }
                    ),
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            files["stage4_decision_audit"] = path
        if (
            self.stack_results
            or self.repeated_split_results
            or self.uncertainty
            or self.logme_results
        ):
            path = output / "stage3_decision_audit.json"
            path.write_text(
                json.dumps(
                    _json_safe(
                        {
                            "qc": self.qc,
                            "encoder_decisions": {
                                name: {
                                    "selected_hyperparameters": result[
                                        "selected_hyperparameters"
                                    ],
                                    "validation_grid": result["validation_grid"],
                                    "decision_audit": result.get("decision_audit"),
                                }
                                for name, result in self.encoder_results.items()
                            },
                            "validation_preferred_method": (
                                self.stack_results or {}
                            ).get("validation_preferred_method"),
                            "combination_decisions": (
                                {
                                    name: {
                                        "metadata": result["metadata"],
                                        "validation_audit": result[
                                            "validation_audit"
                                        ],
                                        "validation_metrics": result[
                                            "validation_metrics"
                                        ],
                                    }
                                    for name, result in (
                                        (self.stack_results or {}).get("methods", {})
                                    ).items()
                                }
                                if self.stack_results
                                else None
                            ),
                            "bootstrap": (
                                {
                                    key: self.uncertainty[key]
                                    for key in (
                                        "method",
                                        "repetitions",
                                        "resampling_unit",
                                        "same_sample_for_all_methods",
                                        "conditioning_statement",
                                    )
                                }
                                if self.uncertainty
                                else None
                            ),
                            "repeated_split_terminology": (
                                self.repeated_split_results or {}
                            ).get("split_percentiles_label"),
                            "logme": self.logme_results,
                            "test_outcomes_used_for_decisions": False,
                        }
                    ),
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            files["stage3_decision_audit"] = path

        if include_predictions:
            prediction_path = output / "predictions.csv"
            self.predictions.to_csv(prediction_path, index=False)
            files["predictions"] = prediction_path
        if include_manifest:
            manifest_path = output / "manifest.csv"
            shareable_manifest = self.manifest.assign(
                group_id_sha256=self.manifest["user_group_id"].map(_token_digest)
            )
            shareable_columns = [
                "observation_id",
                "source_position",
                "image_sha256",
                "group_id_sha256",
                "effective_group_id",
                "split",
                "is_exact_duplicate",
                "status",
            ]
            shareable_manifest[shareable_columns].to_csv(manifest_path, index=False)
            files["manifest"] = manifest_path
        return files

    def export(self, directory: str | Path, **kwargs: Any) -> dict[str, Path]:
        """Alias for :meth:`save`."""

        return self.save(directory, **kwargs)

    def __repr__(self) -> str:
        primary = "mse" if self.task == "regression" else "log_loss"
        value = self.metrics.get("test", {}).get(primary)
        return (
            f"LookAgainResults(task={self.task!r}, selected_encoder={self.selected_encoder!r}, "
            f"test_{primary}={value!r}, n={len(self.manifest)})"
        )
