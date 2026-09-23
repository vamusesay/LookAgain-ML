"""Typed advanced configuration for :func:`lookagain_ml.analyze`.

The ordinary keyword interface remains the primary entry point.  These small,
optional sections group advanced controls without introducing paper-specific
behavior into the general API.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from .exceptions import InputValidationError
from .heads import DEFAULT_LOGISTIC_CS, DEFAULT_RIDGE_ALPHAS
from .structured import DEFAULT_RF_CONFIGURATIONS, DEFAULT_RF_SEEDS

Precision = Literal["float32", "float64"]
ExecutionProfile = Literal["quick", "standard", "full", "paper"]
CompletedRunPolicy = Literal["reuse", "recompute"]


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputValidationError(f"{name} must be a positive integer.")


def _precision(name: str, value: str) -> None:
    if value not in {"float32", "float64"}:
        raise InputValidationError(f"{name} must be 'float32' or 'float64'.")


def _numeric_grid(name: str, values: tuple[float, ...]) -> None:
    if not values or any(
        isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0
        for value in values
    ):
        raise InputValidationError(f"{name} must contain finite positive numeric values.")


def _coerce_numeric_grid(name: str, values: Any) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise InputValidationError(
            f"{name} must contain finite positive numeric values."
        )
    try:
        raw = tuple(values)
    except TypeError as error:
        raise InputValidationError(
            f"{name} must contain finite positive numeric values."
        ) from error
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
        raise InputValidationError(
            f"{name} must contain finite positive numeric values."
        )
    result = tuple(float(value) for value in raw)
    _numeric_grid(name, result)
    return result


@dataclass(frozen=True)
class LinearHeadConfig:
    """Advanced linear-head controls; defaults match ordinary ``analyze``."""

    working_precision: Precision = "float64"
    pca_component_cap: int | None = None
    pca_iterated_power: int = 3
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    logistic_cs: tuple[float, ...] = DEFAULT_LOGISTIC_CS
    logistic_max_iter: int = 2000

    def __post_init__(self) -> None:
        object.__setattr__(self, "ridge_alphas", _coerce_numeric_grid("ridge_alphas", self.ridge_alphas))
        object.__setattr__(self, "logistic_cs", _coerce_numeric_grid("logistic_cs", self.logistic_cs))
        _precision("working_precision", self.working_precision)
        if self.pca_component_cap is not None:
            _positive("pca_component_cap", self.pca_component_cap)
        _positive("pca_iterated_power", self.pca_iterated_power)
        _positive("logistic_max_iter", self.logistic_max_iter)


@dataclass(frozen=True)
class RandomForestConfig:
    """Random-forest menus, fitting seeds, tree count, folds, and parallelism."""

    structured_configurations: tuple[dict[str, Any], ...] = field(
        default_factory=lambda: tuple(dict(value) for value in DEFAULT_RF_CONFIGURATIONS)
    )
    score_configurations: tuple[dict[str, Any], ...] = field(
        default_factory=lambda: tuple(dict(value) for value in DEFAULT_RF_CONFIGURATIONS)
    )
    seeds: tuple[int, ...] = DEFAULT_RF_SEEDS
    n_estimators: int = 500
    score_folds: int = 5
    n_jobs: int | None = -1

    def __post_init__(self) -> None:
        raw_seeds = tuple(self.seeds)
        if not raw_seeds or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_seeds
        ):
            raise InputValidationError(
                "Random-forest fitting seeds must contain at least one integer."
            )
        object.__setattr__(self, "structured_configurations", tuple(dict(v) for v in self.structured_configurations))
        object.__setattr__(self, "score_configurations", tuple(dict(v) for v in self.score_configurations))
        object.__setattr__(self, "seeds", tuple(int(v) for v in raw_seeds))
        if not self.structured_configurations or not self.score_configurations:
            raise InputValidationError("Random-forest configuration menus must be nonempty.")
        _positive("n_estimators", self.n_estimators)
        _positive("score_folds", self.score_folds)
        if self.n_jobs is not None and (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise InputValidationError("n_jobs must be None, -1, or a positive integer.")


@dataclass(frozen=True)
class RFScoreConfig:
    """Controls for leakage-safe RF-score image reduction and cross-fitting."""

    image_max_components: int = 32
    reduction_random_state: int | None = None
    group_fold_random_state: int | None = None
    standardize_reduced_scores: bool = False
    refit_image_reduction_on_train_validation: bool = True
    working_precision: Precision = "float64"

    def __post_init__(self) -> None:
        _positive("image_max_components", self.image_max_components)
        for name in ("reduction_random_state", "group_fold_random_state"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise InputValidationError(f"{name} must be an integer or None.")
        if not isinstance(self.standardize_reduced_scores, bool):
            raise InputValidationError("standardize_reduced_scores must be boolean.")
        if not isinstance(self.refit_image_reduction_on_train_validation, bool):
            raise InputValidationError(
                "refit_image_reduction_on_train_validation must be boolean."
            )
        _precision("working_precision", self.working_precision)


@dataclass(frozen=True)
class AnalysisRuntimeConfig:
    """Runtime and method-menu controls kept separate from scientific heads."""

    encoders: tuple[str, ...] | None = None
    structured_models: tuple[str, ...] | None = None
    integration_methods: tuple[str, ...] | None = None
    device: str = "auto"
    batch_size: int = 32

    def __post_init__(self) -> None:
        for name in ("encoders", "structured_models", "integration_methods"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, (str, bytes)):
                    raise InputValidationError(
                        f"{name} must be an iterable of names, not one string."
                    )
                normalized = tuple(str(item) for item in value)
                if not normalized:
                    raise InputValidationError(f"{name} must be nonempty when supplied.")
                object.__setattr__(self, name, normalized)
        if not isinstance(self.device, str) or not self.device.strip():
            raise InputValidationError("device must be a nonempty string.")
        device = self.device.strip().lower()
        if device not in {"auto", "cpu", "mps", "cuda"} and re.fullmatch(
            r"cuda:\d+", device
        ) is None:
            raise InputValidationError(
                "device must be 'auto', 'cpu', 'mps', 'cuda', or an indexed "
                "CUDA device such as 'cuda:0'."
            )
        object.__setattr__(self, "device", device)
        _positive("batch_size", self.batch_size)


@dataclass(frozen=True)
class ExecutionConfig:
    """Non-scientific progress, checkpoint, resume, and profile controls."""

    profile: ExecutionProfile = "standard"
    progress: Literal["auto"] | bool = "auto"
    progress_callback: Callable[[Any], None] | None = field(default=None, repr=False)
    heartbeat_interval_seconds: float = 30.0
    checkpoint_dir: str | Path | None = None
    resume: bool = True
    run_id: str | None = None
    protocol_identifier: str | None = None
    protocol_fingerprint: str | None = None
    completed_run_policy: CompletedRunPolicy = "reuse"
    persistent_event_interval_seconds: float = 30.0
    stale_lock_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.profile not in {"quick", "standard", "full", "paper"}:
            raise InputValidationError(
                "profile must be 'quick', 'standard', 'full', or 'paper'."
            )
        if self.progress != "auto" and not isinstance(self.progress, bool):
            raise InputValidationError("progress must be 'auto', True, or False.")
        if self.progress_callback is not None and not callable(self.progress_callback):
            raise InputValidationError("progress_callback must be callable or None.")
        if (
            isinstance(self.heartbeat_interval_seconds, bool)
            or not math.isfinite(float(self.heartbeat_interval_seconds))
            or float(self.heartbeat_interval_seconds) <= 0
        ):
            raise InputValidationError("heartbeat_interval_seconds must be finite and positive.")
        if not isinstance(self.resume, bool):
            raise InputValidationError("resume must be boolean.")
        if self.checkpoint_dir is not None and not str(self.checkpoint_dir).strip():
            raise InputValidationError("checkpoint_dir must be a nonempty path or None.")
        if self.run_id is not None and not str(self.run_id).strip():
            raise InputValidationError("run_id must be a nonempty string or None.")
        if self.profile == "paper" and (
            not self.protocol_identifier or not self.protocol_fingerprint
        ):
            raise InputValidationError(
                "The paper execution profile requires a protocol identifier and fingerprint."
            )
        if self.completed_run_policy not in {"reuse", "recompute"}:
            raise InputValidationError(
                "completed_run_policy must be 'reuse' or 'recompute'."
            )
        if (
            isinstance(self.persistent_event_interval_seconds, bool)
            or not math.isfinite(float(self.persistent_event_interval_seconds))
            or float(self.persistent_event_interval_seconds) <= 0
        ):
            raise InputValidationError(
                "persistent_event_interval_seconds must be finite and positive."
            )
        if self.stale_lock_timeout_seconds is not None and (
            isinstance(self.stale_lock_timeout_seconds, bool)
            or not math.isfinite(float(self.stale_lock_timeout_seconds))
            or float(self.stale_lock_timeout_seconds) <= 0
        ):
            raise InputValidationError(
                "stale_lock_timeout_seconds must be finite and positive or None."
            )

    @property
    def scientific_comparison_eligible(self) -> bool:
        """Only a versioned, fingerprinted paper protocol is comparison eligible."""

        return self.profile == "paper"

    def to_dict(self) -> dict[str, Any]:
        """Return a shareable record without callback objects or absolute paths."""

        return {
            "profile": self.profile,
            "progress": self.progress,
            "progress_callback_supplied": self.progress_callback is not None,
            "heartbeat_interval_seconds": float(self.heartbeat_interval_seconds),
            "checkpointing_enabled": self.checkpoint_dir is not None,
            "checkpoint_location": (
                "user-local; absolute path intentionally omitted"
                if self.checkpoint_dir is not None
                else None
            ),
            "resume": self.resume,
            "run_id_supplied": self.run_id is not None,
            "protocol_identifier": self.protocol_identifier,
            "protocol_fingerprint": self.protocol_fingerprint,
            "completed_run_policy": self.completed_run_policy,
            "persistent_event_interval_seconds": float(
                self.persistent_event_interval_seconds
            ),
            "stale_lock_timeout_seconds": self.stale_lock_timeout_seconds,
            "scientific_comparison_eligible": self.scientific_comparison_eligible,
            "diagnostic_only": self.profile == "quick",
        }


@dataclass(frozen=True)
class AdvancedConfig:
    """Optional typed sections that override corresponding advanced keywords.

    Sections not supplied leave the ordinary ``analyze`` keyword values intact.
    If a section is supplied, that typed section is authoritative for its fields.
    """

    linear_head: LinearHeadConfig | None = None
    random_forest: RandomForestConfig | None = None
    rf_score: RFScoreConfig | None = None
    runtime: AnalysisRuntimeConfig | None = None
    execution: ExecutionConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration record."""

        record = asdict(self)
        if self.execution is not None:
            record["execution"] = self.execution.to_dict()
        return record
