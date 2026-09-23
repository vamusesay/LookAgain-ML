"""High-level leakage-aware representation analysis workflow."""

from __future__ import annotations

import functools
import gc
import hashlib
import json
import logging
import os
import platform
import time
import uuid
import warnings
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from ._version import DISTRIBUTION_NAME, IMPORT_NAME, __version__
from .cache import EmbeddingCache, build_cache_identity
from .checkpointing import CheckpointStore
from .combination import (
    fit_feature_concatenation,
    fit_validation_combinations,
    normalize_combination_methods,
)
from .configuration import AdvancedConfig, ExecutionConfig
from .covariate_analysis import (
    fit_covariate_analysis,
    joint_image_encoder_scope,
    take_covariate_rows,
)
from .encoders import create_encoder, get_registration
from .encoders.base import EncoderMetadata
from .exceptions import InputValidationError
from .hashing import digest_strings, sha256_file
from .heads import (
    DEFAULT_LOGISTIC_CS,
    DEFAULT_RIDGE_ALPHAS,
    fit_linear_head,
    fitted_linear_head_audit,
)
from .images import build_image_manifest
from .integration import incremental_metrics, normalize_integration_methods
from .logme import logme_score
from .metrics import metric_function
from .progress import ProgressReporter, encoder_batch_callback, run_with_heartbeat
from .protocols import configuration_sha256
from .result_checkpoint import (
    load_result_checkpoint,
    result_checkpoint_reference,
    verify_reference_metadata,
    write_result_checkpoint,
)
from .results import LookAgainResults
from .splitting import make_split, validate_partition_safety
from .stability import repeated_split_analysis
from .stability_xi import repeated_xi_analysis
from .structured import (
    DEFAULT_RF_CONFIGURATIONS,
    DEFAULT_RF_SEEDS,
    normalize_structured_models,
    subset_covariates,
)
from .uncertainty import paired_locked_test_bootstrap

LOGGER = logging.getLogger(__name__)
_CHECKPOINT_SCOPE: ContextVar[list[CheckpointStore] | None] = ContextVar(
    "lookagain_checkpoint_scope", default=None
)


def _feature_values_sha256(features: np.ndarray) -> str:
    """Digest float32 feature values in C row order without a whole-array copy."""

    matrix = np.asanyarray(features)
    digest = hashlib.sha256()
    if matrix.flags.c_contiguous:
        digest.update(memoryview(matrix).cast("B"))
    else:
        for row in matrix:
            digest.update(memoryview(np.ascontiguousarray(row)).cast("B"))
    return digest.hexdigest()


def _positive_integer(name: str, value: Any, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise InputValidationError(f"{name} must be a {qualifier} integer.")


def _finite_number(name: str, value: Any, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise InputValidationError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise InputValidationError(f"{name} must be at least {minimum}.")


def _validate_public_options(**options: Any) -> None:
    """Fail early with actionable errors before images or models are touched."""

    for name in (
        "pretrained",
        "cache",
        "strict",
        "verify_images",
        "logme",
        "covariates_preprocessed",
        "numeric_missing_indicators",
    ):
        if not isinstance(options[name], bool):
            raise InputValidationError(f"{name} must be True or False.")
    for name in (
        "batch_size",
        "repeated_splits",
        "logme_max_components",
        "feature_concatenation_max_components",
        "joint_image_max_components",
        "nn_max_iter",
        "rf_n_estimators",
        "rf_score_folds",
        "rf_score_image_max_components",
        "head_pca_iterated_power",
        "logistic_max_iter",
    ):
        _positive_integer(name, options[name])
    if options["head_max_components"] is not None:
        _positive_integer("head_max_components", options["head_max_components"])
    if options["head_working_dtype"] not in {"float32", "float64"}:
        raise InputValidationError("head_working_dtype must be 'float32' or 'float64'.")
    if options["rf_score_working_dtype"] not in {"float32", "float64"}:
        raise InputValidationError("rf_score_working_dtype must be 'float32' or 'float64'.")
    if options["rf_n_jobs"] is not None and (
        isinstance(options["rf_n_jobs"], bool)
        or not isinstance(options["rf_n_jobs"], Integral)
        or options["rf_n_jobs"] == 0
        or options["rf_n_jobs"] < -1
    ):
        raise InputValidationError("rf_n_jobs must be None, -1, or a positive integer.")
    for name in ("rf_score_reduction_random_state", "rf_score_fold_random_state"):
        if options[name] is not None and (
            isinstance(options[name], bool) or not isinstance(options[name], Integral)
        ):
            raise InputValidationError(f"{name} must be an integer or None.")
    for name in ("rf_score_standardize_reduced_scores", "rf_score_refit_image_reduction"):
        if not isinstance(options[name], bool):
            raise InputValidationError(f"{name} must be True or False.")
    bootstrap_repetitions = options["bootstrap_repetitions"]
    if (
        isinstance(bootstrap_repetitions, bool)
        or not isinstance(bootstrap_repetitions, Integral)
        or (bootstrap_repetitions != 0 and bootstrap_repetitions < 100)
    ):
        raise InputValidationError(
            "bootstrap_repetitions must be the integer 0 or an integer of at least 100."
        )
    if isinstance(options["random_state"], bool) or not isinstance(
        options["random_state"], Integral
    ):
        raise InputValidationError("random_state must be an integer.")
    if not isinstance(options["device"], str) or not options["device"].strip():
        raise InputValidationError(
            "device must be 'auto', 'cpu', 'mps', 'cuda', or a CUDA device such as 'cuda:0'."
        )
    for name in ("stack_alpha", "stack_c"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise InputValidationError(f"{name} must be a positive finite number.")
    _finite_number("score_integration_alpha", options["score_integration_alpha"], minimum=0.0)
    _finite_number("score_integration_c", options["score_integration_c"], minimum=0.0)
    if options["score_integration_c"] == 0:
        raise InputValidationError("score_integration_c must be greater than 0.")


def _infer_task(values: np.ndarray, requested: str) -> str:
    if not isinstance(requested, str):
        raise InputValidationError(
            "task must be 'auto', 'regression', or 'classification'."
        )
    task = requested.strip().lower()
    if task not in {"auto", "regression", "classification"}:
        raise InputValidationError("task must be 'auto', 'regression', or 'classification'.")
    if task != "auto":
        return task
    target_type = type_of_target(values)
    if target_type == "continuous":
        return "regression"
    if target_type in {"binary", "multiclass"}:
        return "classification"
    raise InputValidationError(
        f"Could not infer a supported task from target type {target_type!r}. "
        "Supply one continuous outcome or one binary/multiclass label per image."
    )


def _prepare_outcome(values: np.ndarray, task: str) -> tuple[np.ndarray, list[Any] | None]:
    if task == "regression":
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise InputValidationError("Regression outcomes must be numeric.") from error
        if not np.isfinite(numeric).all():
            raise InputValidationError("Regression outcomes must be finite.")
        return numeric, None
    try:
        encoder = LabelEncoder().fit(values)
        encoded = encoder.transform(values).astype(int)
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            "Classification outcomes must be one-dimensional scalar labels with "
            "a consistent, comparable type (for example, all strings or all numbers)."
        ) from error
    if len(encoder.classes_) < 2:
        raise InputValidationError("Classification requires at least two outcome classes.")
    return encoded, encoder.classes_.tolist()


def _dependency_versions() -> dict[str, str | None]:
    installed: dict[str, str | None] = {}
    for distribution in (
        "numpy", "pandas", "scipy", "scikit-learn", "torch", "torchvision",
        "Pillow", "transformers", "huggingface-hub",
    ):
        try:
            installed[distribution] = version(distribution)
        except PackageNotFoundError:
            installed[distribution] = None
    return installed


def _resource_record(rf_n_jobs: int | None) -> dict[str, Any]:
    """Record effective CPU controls without changing estimator behavior."""

    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    try:
        from threadpoolctl import threadpool_info

        libraries = [
            {
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "num_threads": item.get("num_threads"),
            }
            for item in threadpool_info()
        ]
    except ImportError:
        libraries = []
    return {
        "logical_cpu_count": os.cpu_count(),
        "rf_n_jobs": rf_n_jobs,
        "thread_environment": thread_variables,
        "threadpool_libraries": libraries,
        "gpu_primary_role": "frozen image encoding",
        "cpu_bound_stages": "scikit-learn preprocessing and estimator fitting",
        "oversubscription_guidance": (
            "Set RF n_jobs and BLAS/OpenMP thread limits conservatively for available RAM."
        ),
    }


def _resolve_encoders(encoder: str, encoders: Iterable[str] | None) -> list[str]:
    if not isinstance(encoder, str):
        raise InputValidationError("encoder must be one registered encoder name.")
    if isinstance(encoders, str):
        raise InputValidationError(
            "encoders must be an iterable of names, such as ['resnet50', 'dinov2_vitb14']; "
            "a single string is ambiguous."
        )
    if encoders is None:
        requested = [encoder]
    else:
        if encoder.strip().lower() != "resnet50":
            raise InputValidationError(
                "Use either encoder= for one representation or encoders= for a comparison, not both."
            )
        requested = list(encoders)
    if not requested:
        raise InputValidationError("encoders must contain at least one registered encoder name.")
    normalized: list[str] = []
    for name in requested:
        if not isinstance(name, str):
            raise InputValidationError("Every encoder name must be a string.")
        registered = get_registration(name)
        if registered.name in normalized:
            raise InputValidationError(f"Encoder {registered.name!r} was requested more than once.")
        normalized.append(registered.name)
    return normalized


def _cache_runtime_dependencies(metadata: EncoderMetadata) -> dict[str, str | None]:
    """Versions that can change an encoder's numerical representation."""

    installed = _dependency_versions()
    names = ["torch"]
    if metadata.backend == "torchvision":
        names.append("torchvision")
    elif metadata.backend == "transformers":
        names.extend(["transformers", "huggingface-hub"])
    return {name: installed[name] for name in names if installed[name] is not None}


def _verify_image_hashes(
    paths: list[str],
    expected_hashes: list[str],
    *,
    phase: str,
    progress: ProgressReporter | None = None,
) -> None:
    current_hashes = []
    if progress is not None:
        progress.emit(
            "STARTED", "input_integrity", f"verifying image contents {phase}", total=len(paths)
        )
    for index, path in enumerate(paths, start=1):
        current_hashes.append(sha256_file(path))
        if progress is not None:
            progress.emit(
                "VERIFYING",
                "input_integrity",
                "image content-integrity verification in progress",
                current=index,
                total=len(paths),
                rows_processed=index,
            )
    if current_hashes != expected_hashes:
        raise InputValidationError(
            f"At least one image changed {phase}. No representation created during that "
            "phase was cached; restart with stable input files."
        )
    if progress is not None:
        progress.emit(
            "COMPLETED",
            "input_integrity",
            f"image contents verified {phase}",
            current=len(paths),
            total=len(paths),
            rows_processed=len(paths),
        )


def _select_encoder(
    encoder_names: list[str], encoder_results: dict[str, Any], primary_metric: str
) -> str:
    """Freeze the validation-only encoder decision before test evaluation."""

    return min(
        encoder_names,
        key=lambda name: encoder_results[name]["validation_metrics"][primary_metric],
    )


def _declared_metadata(name: str, pretrained: bool) -> EncoderMetadata:
    metadata = get_registration(name).metadata
    if pretrained:
        return metadata
    return EncoderMetadata(
        **{
            **metadata.to_dict(),
            "checkpoint": "untrained weights (testing/infrastructure checks only)",
            "checkpoint_revision": None,
        }
    )


def _validate_precomputed_representation(
    name: str,
    *,
    features_by_name: Mapping[str, Any],
    audits_by_name: Mapping[str, Mapping[str, Any]],
    expected_metadata: EncoderMetadata,
    row_count: int,
    observation_digest: str,
    image_digest: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    audit_record = dict(audits_by_name[name])
    required_audit = {
        "ordered_observation_id_sha256",
        "ordered_image_hash_sha256",
        "feature_values_sha256",
        "encoder_metadata",
        "source_array_sha256",
        "source_metadata_sha256",
    }
    missing_audit = sorted(required_audit - set(audit_record))
    if missing_audit:
        raise InputValidationError(
            f"Precomputed audit for {name!r} is missing fields: {missing_audit}."
        )
    if audit_record["ordered_observation_id_sha256"] != observation_digest:
        raise InputValidationError(
            f"Precomputed features for {name!r} do not match the current ordered observation IDs."
        )
    if audit_record["ordered_image_hash_sha256"] != image_digest:
        raise InputValidationError(
            f"Precomputed features for {name!r} do not match the current ordered image hashes."
        )
    if audit_record["encoder_metadata"] != expected_metadata.to_dict():
        raise InputValidationError(
            f"Precomputed features for {name!r} do not match the registered checkpoint and feature definition."
        )
    features = np.asanyarray(features_by_name[name])
    expected_shape = (row_count, expected_metadata.feature_dimension)
    if features.shape != expected_shape or features.dtype != np.float32:
        raise InputValidationError(
            f"Precomputed features for {name!r} have {features.shape}/{features.dtype}; "
            f"expected {expected_shape}/float32."
        )
    value_digest = _feature_values_sha256(features)
    if value_digest != audit_record["feature_values_sha256"]:
        raise InputValidationError(
            f"Precomputed feature-value digest failed for {name!r}."
        )
    if not np.isfinite(features).all():
        raise InputValidationError(
            f"Precomputed features for {name!r} contain non-finite values."
        )
    creation_costs = dict(audit_record.get("creation_costs", {}))
    cache_record = {
        "status": "provided-verified",
        "cache_key": None,
        "cache_schema_version": None,
        "array_sha256": audit_record["source_array_sha256"],
        "source_metadata_sha256": audit_record["source_metadata_sha256"],
        "cache_bytes": int(features.nbytes),
        "verified_fields": sorted(required_audit),
        "creation_costs": creation_costs,
        "interpretation": (
            "Externally generated frozen features with verified row/image/checkpoint metadata; "
            "not fresh LookAgain-ML inference and not a LookAgain-ML-created cache."
        ),
    }
    costs = {
        "device": "provided-precomputed-no-inference",
        "parameter_count": None,
        "extraction_seconds": 0.0,
        "images_per_second": None,
        "feature_dimension": expected_metadata.feature_dimension,
        "peak_gpu_memory_bytes": None,
        "cached_creation_device": creation_costs.get("device"),
        "cached_creation_parameter_count": creation_costs.get("parameter_count"),
        "cached_creation_extraction_seconds": creation_costs.get("extraction_seconds"),
        "cached_creation_images_per_second": creation_costs.get("images_per_second"),
        "cached_creation_peak_gpu_memory_bytes": creation_costs.get("peak_gpu_memory_bytes"),
    }
    return features, cache_record, costs


def _prediction_frame(
    manifest: pd.DataFrame,
    raw_target: np.ndarray,
    prediction: np.ndarray,
    task: str,
    classes: list[Any] | None,
) -> pd.DataFrame:
    rows: dict[str, Any] = {
        "observation_id": manifest["observation_id"].to_numpy(),
        "source_position": manifest["source_position"].to_numpy(),
        "effective_group_id": manifest["effective_group_id"].to_numpy(),
        "split": manifest["split"].to_numpy(),
        "y_true": raw_target,
    }
    if task == "regression":
        rows["y_pred"] = prediction
    else:
        assert classes is not None
        predicted_index = prediction.argmax(axis=1)
        class_array = np.asarray(classes, dtype=object)
        rows["y_pred"] = class_array[predicted_index]
        for class_index in range(prediction.shape[1]):
            rows[f"probability_class_{class_index}"] = prediction[:, class_index]
    return pd.DataFrame(rows)


def _analyze_impl(
    images: Iterable[str],
    y: Iterable[Any],
    groups: Iterable[Any] | None = None,
    *,
    observation_ids: Iterable[Any] | None = None,
    covariates: Any | None = None,
    split_labels: Iterable[Any] | None = None,
    task: str = "auto",
    encoder: str = "resnet50",
    encoders: Iterable[str] | None = None,
    device: str = "auto",
    batch_size: int = 32,
    pretrained: bool = True,
    cache: bool = True,
    cache_dir: str | Path = ".lookagain_cache",
    model_cache_dir: str | Path | None = None,
    strict: bool = True,
    verify_images: bool = True,
    random_state: int = 20260818,
    ridge_alphas: Iterable[float] = DEFAULT_RIDGE_ALPHAS,
    logistic_cs: Iterable[float] = DEFAULT_LOGISTIC_CS,
    head_max_components: int | None = None,
    head_pca_iterated_power: int = 3,
    logistic_max_iter: int = 2000,
    head_working_dtype: str = "float64",
    combinations: Iterable[str] | None = None,
    repeated_splits: int = 1,
    bootstrap_repetitions: int = 0,
    bootstrap_metrics: Iterable[str] | None = None,
    logme: bool = False,
    logme_max_components: int = 128,
    stack_alpha: float = 10.0,
    stack_c: float = 0.1,
    feature_concatenation_max_components: int = 128,
    structured_models: Iterable[str] | None = None,
    integration_methods: Iterable[str] | None = None,
    rf_score_encoder: str | None = None,
    rf_structured_configurations: Iterable[dict[str, Any]] | None = None,
    rf_score_configurations: Iterable[dict[str, Any]] | None = None,
    rf_seeds: Iterable[int] = DEFAULT_RF_SEEDS,
    rf_n_estimators: int = 500,
    rf_score_folds: int = 5,
    rf_score_image_max_components: int = 32,
    covariates_preprocessed: bool = False,
    numeric_missing_indicators: bool = True,
    score_integration_alpha: float = 0.0,
    score_integration_c: float = 0.1,
    joint_image_max_components: int = 128,
    nn_max_iter: int = 300,
    precomputed_features: Mapping[str, Any] | None = None,
    precomputed_feature_audits: Mapping[str, Mapping[str, Any]] | None = None,
    advanced_config: AdvancedConfig | None = None,
) -> LookAgainResults:
    """Analyze image prediction safely with optional structured covariates.

    Parameters
    ----------
    images:
        One local image path per observation. LookAgain-ML validates and hashes
        every requested image and never silently drops failures.
    y:
        One regression outcome or classification label per image.
    groups:
        Optional group identifier per image. Rows in the same group are kept in
        one partition. Exact duplicate image hashes are grouped automatically.
    observation_ids:
        Optional stable unique row identifiers. Supply these when validating
        externally generated frozen features against their original row order.
    covariates:
        Optional pandas DataFrame or numeric matrix containing structured
        covariates X. DataFrame preprocessing is fitted on training rows only.
    split_labels:
        Optional explicit ``train``, ``validation``, and ``test`` labels. A
        supplied test partition is validated but never repartitioned.
    task:
        ``"auto"``, ``"regression"``, or ``"classification"``. Specify
        regression when an integer-valued outcome is conceptually continuous.
    encoder, encoders:
        Use ``encoder=`` for one registered representation or ``encoders=`` for
        a validation comparison. Do not supply both.
    device:
        ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or a CUDA device such
        as ``"cuda:0"``. Availability is checked when the encoder is created.
    cache, cache_dir, model_cache_dir:
        Control verified embedding reuse and public checkpoint storage. Cache
        identity includes ordered rows, image hashes, checkpoint,
        preprocessing, feature dimension, runtime versions, and package schema.
    strict:
        When true (default), any missing/corrupt image fails the analysis. When
        false, failures are warned about and retained in ``results.audit``.
    combinations:
        Optional image-encoder combination methods: ``equal_average``,
        ``linear_stack``, ``greedy_average``, or ``feature_concatenation``.
    repeated_splits:
        Number of complete group-safe refits. Values greater than one require
        generated splits rather than explicit ``split_labels``.
    bootstrap_repetitions:
        Paired locked-test bootstrap repetitions. Use zero to disable or at
        least 100 when enabled.
    logme:
        Enable the optional training-only LogME screening diagnostic.
    structured_models:
        With ``covariates=``, choose ``linear`` and optionally ``nn`` or ``rf``.
    integration_methods:
        With ``covariates=``, choose from ``linear_score``, ``nn_score``,
        ``rf_score`` (regression only), ``joint_linear``, and ``joint_nn``.
        Linear score is the default.
    rf_score_encoder:
        Single frozen representation used to form the leakage-safe out-of-fold
        image score for ``rf_score``. Defaults to the validation-selected single
        encoder and must name one of ``encoders=``.
    precomputed_features, precomputed_feature_audits:
        Optional externally generated frozen feature matrices and their strict
        provenance audits. Both mappings must contain exactly the requested
        encoder names. Every audit must match the current ordered observation
        IDs, current image hashes, registered encoder metadata, feature-value
        digest, row count, and dimension. Accepted matrices are labeled
        ``provided-verified`` and are never described as fresh package
        inference or a package-created representation cache.

    Returns
    -------
    LookAgainResults
        A structured result containing validation decisions, locked-test
        metrics, costs, QC, optional uncertainty/repeated-split outputs, and
        save/plot helpers.

    Notes
    -----
    All preprocessing, head tuning, encoder selection, image-combination
    weights, structured-model choice, and X+I integration decisions are frozen
    from training/validation information before test outcomes are evaluated.
    See the API reference for the complete advanced-option table.
    """

    rf_n_jobs: int | None = -1
    rf_score_reduction_random_state: int | None = None
    rf_score_fold_random_state: int | None = None
    rf_score_standardize_reduced_scores = False
    rf_score_refit_image_reduction = True
    rf_score_working_dtype = "float64"
    execution_config = ExecutionConfig()
    if advanced_config is not None:
        if not isinstance(advanced_config, AdvancedConfig):
            raise InputValidationError("advanced_config must be an AdvancedConfig instance.")
        if advanced_config.linear_head is not None:
            section = advanced_config.linear_head
            ridge_alphas = section.ridge_alphas
            logistic_cs = section.logistic_cs
            head_max_components = section.pca_component_cap
            head_pca_iterated_power = section.pca_iterated_power
            logistic_max_iter = section.logistic_max_iter
            head_working_dtype = section.working_precision
        if advanced_config.random_forest is not None:
            section = advanced_config.random_forest
            rf_structured_configurations = section.structured_configurations
            rf_score_configurations = section.score_configurations
            rf_seeds = section.seeds
            rf_n_estimators = section.n_estimators
            rf_score_folds = section.score_folds
            rf_n_jobs = section.n_jobs
        if advanced_config.rf_score is not None:
            section = advanced_config.rf_score
            rf_score_image_max_components = section.image_max_components
            rf_score_reduction_random_state = section.reduction_random_state
            rf_score_fold_random_state = section.group_fold_random_state
            rf_score_standardize_reduced_scores = section.standardize_reduced_scores
            rf_score_refit_image_reduction = (
                section.refit_image_reduction_on_train_validation
            )
            rf_score_working_dtype = section.working_precision
        if advanced_config.runtime is not None:
            section = advanced_config.runtime
            if section.encoders is not None:
                encoder = "resnet50"
                encoders = section.encoders
            if section.structured_models is not None:
                structured_models = section.structured_models
            if section.integration_methods is not None:
                integration_methods = section.integration_methods
            device = section.device
            batch_size = section.batch_size
        if advanced_config.execution is not None:
            execution_config = advanced_config.execution

    _validate_public_options(
        device=device,
        batch_size=batch_size,
        pretrained=pretrained,
        cache=cache,
        strict=strict,
        verify_images=verify_images,
        random_state=random_state,
        repeated_splits=repeated_splits,
        bootstrap_repetitions=bootstrap_repetitions,
        logme=logme,
        logme_max_components=logme_max_components,
        stack_alpha=stack_alpha,
        stack_c=stack_c,
        feature_concatenation_max_components=feature_concatenation_max_components,
        covariates_preprocessed=covariates_preprocessed,
        numeric_missing_indicators=numeric_missing_indicators,
        score_integration_alpha=score_integration_alpha,
        score_integration_c=score_integration_c,
        joint_image_max_components=joint_image_max_components,
        nn_max_iter=nn_max_iter,
        rf_n_estimators=rf_n_estimators,
        rf_score_folds=rf_score_folds,
        rf_score_image_max_components=rf_score_image_max_components,
        head_max_components=head_max_components,
        head_pca_iterated_power=head_pca_iterated_power,
        logistic_max_iter=logistic_max_iter,
        head_working_dtype=head_working_dtype,
        rf_n_jobs=rf_n_jobs,
        rf_score_reduction_random_state=rf_score_reduction_random_state,
        rf_score_fold_random_state=rf_score_fold_random_state,
        rf_score_standardize_reduced_scores=rf_score_standardize_reduced_scores,
        rf_score_refit_image_reduction=rf_score_refit_image_reduction,
        rf_score_working_dtype=rf_score_working_dtype,
    )
    encoder_names = _resolve_encoders(encoder, encoders)
    provided_features = precomputed_features is not None or precomputed_feature_audits is not None
    if (precomputed_features is None) != (precomputed_feature_audits is None):
        raise InputValidationError(
            "precomputed_features and precomputed_feature_audits must be supplied together."
        )
    if provided_features:
        if not isinstance(precomputed_features, Mapping) or not isinstance(
            precomputed_feature_audits, Mapping
        ):
            raise InputValidationError(
                "precomputed_features and precomputed_feature_audits must be mappings."
            )
        expected_names = set(encoder_names)
        if set(precomputed_features) != expected_names or set(precomputed_feature_audits) != expected_names:
            raise InputValidationError(
                "Precomputed feature/audit keys must exactly match requested encoders: "
                f"{sorted(expected_names)}."
            )
        if not pretrained:
            raise InputValidationError(
                "precomputed_features require pretrained=True because their audits identify frozen checkpoints."
            )
    raw_rf_seeds = tuple(rf_seeds)
    if not raw_rf_seeds or any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_rf_seeds
    ):
        raise InputValidationError("rf_seeds must contain at least one integer seed.")
    rf_seed_values = tuple(int(value) for value in raw_rf_seeds)
    structured_rf_values = tuple(
        dict(value) for value in (
            DEFAULT_RF_CONFIGURATIONS
            if rf_structured_configurations is None
            else rf_structured_configurations
        )
    )
    score_rf_values = tuple(
        dict(value) for value in (
            DEFAULT_RF_CONFIGURATIONS
            if rf_score_configurations is None
            else rf_score_configurations
        )
    )
    if not structured_rf_values or not score_rf_values:
        raise InputValidationError("RF configuration grids must not be empty.")
    if rf_score_encoder is not None:
        rf_score_encoder = str(rf_score_encoder).strip().lower()
        if rf_score_encoder not in encoder_names:
            raise InputValidationError(
                "rf_score_encoder must name one of the requested encoders."
            )
    combination_methods = normalize_combination_methods(combinations)
    has_covariates = covariates is not None
    if has_covariates:
        structured_model_names = normalize_structured_models(structured_models)
        integration_method_names = normalize_integration_methods(integration_methods)
    else:
        if structured_models is not None or integration_methods is not None:
            raise InputValidationError(
                "structured_models and integration_methods require covariates=."
            )
        structured_model_names = []
        integration_method_names = []
    if isinstance(bootstrap_metrics, str):
        raise InputValidationError(
            "bootstrap_metrics must be an iterable of metric names, such as ['r2', 'mse']."
        )
    if repeated_splits > 1 and split_labels is not None:
        raise InputValidationError(
            "repeated_splits cannot repartition user-supplied split_labels. "
            "Run the supplied locked split once, or omit split_labels for repeated group-safe splits."
        )
    cache_requested = cache
    if cache and not pretrained:
        warnings.warn(
            "Caching was disabled because untrained random weights do not have a verifiable "
            "checkpoint identity. pretrained=False is for infrastructure checks only.",
            UserWarning,
            stacklevel=2,
        )
        cache = False
    requested_splits = list(split_labels) if split_labels is not None else None
    ridge_alpha_values = tuple(ridge_alphas)
    logistic_c_values = tuple(logistic_cs)
    initial_run_id = execution_config.run_id or uuid.uuid4().hex
    if execution_config.checkpoint_dir is not None:
        prior_manifest_path = (
            Path(execution_config.checkpoint_dir).expanduser().resolve()
            / "run_manifest.json"
        )
        if prior_manifest_path.is_file():
            try:
                initial_run_id = str(
                    json.loads(prior_manifest_path.read_text(encoding="utf-8"))["run_id"]
                )
            except (KeyError, TypeError, ValueError, OSError):
                pass
    progress = ProgressReporter(
        execution_config.progress,
        callback=execution_config.progress_callback,
        run_id=initial_run_id,
        heartbeat_interval_seconds=execution_config.heartbeat_interval_seconds,
    )
    progress.emit(
        "STARTED",
        "input_validation",
        "validating and content-hashing input images",
    )
    try:
        manifest_result = build_image_manifest(
            images,
            y,
            groups,
            observation_ids=observation_ids,
            strict=strict,
            verify_images=verify_images,
            progress_callback=lambda current, total: progress.emit(
                "VERIFYING",
                "input_validation",
                "input image verification in progress",
                current=current,
                total=total,
                rows_processed=current,
            ),
        )
    except BaseException as error:
        progress.emit(
            "FAILED",
            "input_validation",
            "input validation failed",
            details={"exception_type": type(error).__name__},
        )
        raise
    progress.emit(
        "COMPLETED",
        "input_validation",
        "input validation and content hashing completed",
        current=manifest_result.audit.requested_n,
        total=manifest_result.audit.requested_n,
        rows_processed=manifest_result.audit.successful_n,
    )
    manifest = manifest_result.manifest.copy()
    aligned_covariates = (
        subset_covariates(
            covariates,
            manifest["source_position"].to_numpy(dtype=int),
            manifest_result.audit.requested_n,
        )
        if has_covariates
        else None
    )
    if manifest_result.audit.failed_n:
        warnings.warn(
            f"LookAgain omitted {manifest_result.audit.failed_n} failed observations. "
            "Inspect results.audit for every row and reason.",
            UserWarning,
            stacklevel=2,
        )
    raw_target = manifest["target"].to_numpy()
    inferred_task = _infer_task(raw_target, task)
    outcome, classes = _prepare_outcome(raw_target, inferred_task)

    filtered_splits = None
    if requested_splits is not None:
        if len(requested_splits) != manifest_result.audit.requested_n:
            raise InputValidationError(
                "split_labels must have one value per requested image before validation; "
                f"got {len(requested_splits)} and {manifest_result.audit.requested_n}."
            )
        filtered_splits = [requested_splits[int(position)] for position in manifest.source_position]
    split = make_split(
        outcome,
        manifest["user_group_id"],
        manifest["image_sha256"],
        task=inferred_task,
        random_state=random_state,
        split_labels=filtered_splits,
    )
    manifest["effective_group_id"] = split.effective_groups
    manifest["split"] = split.labels
    validate_partition_safety(
        manifest["split"], manifest["user_group_id"], manifest["image_sha256"]
    )
    masks = {
        name: manifest["split"].to_numpy() == name
        for name in ("train", "validation", "test")
    }
    if inferred_task == "classification":
        train_classes = set(outcome[masks["train"]].tolist())
        all_classes = set(outcome.tolist())
        if train_classes != all_classes:
            missing = sorted(all_classes - train_classes)
            raise InputValidationError(
                "The leakage-safe training partition does not contain every class "
                f"(missing encoded classes: {missing}). Supply more independent groups "
                "or explicit safe splits."
            )

    observation_digest = digest_strings(manifest["observation_id"].astype(str).tolist())
    image_digest = digest_strings(manifest["image_sha256"].astype(str).tolist())
    outcome_digest = digest_strings(manifest["target"].astype(str).tolist())
    group_digest = digest_strings(manifest["user_group_id"].astype(str).tolist())
    split_assignment_digest = digest_strings(
        (manifest["observation_id"].astype(str) + "|" + manifest["split"].astype(str)).tolist()
    )
    if aligned_covariates is None:
        covariate_digest = None
    else:
        covariate_frame = (
            aligned_covariates
            if isinstance(aligned_covariates, pd.DataFrame)
            else pd.DataFrame(np.asarray(aligned_covariates))
        )
        covariate_digest = hashlib.sha256(
            pd.util.hash_pandas_object(covariate_frame, index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        ).hexdigest()
    resource_record = _resource_record(rf_n_jobs)
    scientific_settings = {
        "task": inferred_task,
        "encoders": encoder_names,
        "random_state": random_state,
        "ridge_alphas": ridge_alpha_values,
        "logistic_cs": logistic_c_values,
        "head_max_components": head_max_components,
        "head_pca_iterated_power": head_pca_iterated_power,
        "logistic_max_iter": logistic_max_iter,
        "head_working_dtype": head_working_dtype,
        "combinations": combination_methods,
        "repeated_splits": repeated_splits,
        "bootstrap_repetitions": bootstrap_repetitions,
        "structured_models": structured_model_names,
        "integration_methods": integration_method_names,
        "rf_score_encoder": rf_score_encoder,
        "rf_structured_configurations": structured_rf_values,
        "rf_score_configurations": score_rf_values,
        "rf_seeds": rf_seed_values,
        "rf_n_estimators": rf_n_estimators,
        "rf_score_folds": rf_score_folds,
        "rf_score_image_max_components": rf_score_image_max_components,
        "rf_n_jobs": rf_n_jobs,
        "rf_score_reduction_random_state": rf_score_reduction_random_state,
        "rf_score_fold_random_state": rf_score_fold_random_state,
        "rf_score_standardize_reduced_scores": rf_score_standardize_reduced_scores,
        "rf_score_refit_image_reduction": rf_score_refit_image_reduction,
        "rf_score_working_dtype": rf_score_working_dtype,
    }
    analysis_configuration_sha256 = configuration_sha256(scientific_settings)
    planned_units = [f"encoder:{name}" for name in encoder_names]
    if repeated_splits > 1:
        planned_units.extend(
            f"repeated_image:split_{index:04d}" for index in range(repeated_splits)
        )
        if has_covariates:
            planned_units.extend(
                f"repeated_xi:split_{index:04d}" for index in range(repeated_splits)
            )
    run_identity = {
        "package_version": __version__,
        "protocol_identifier": execution_config.protocol_identifier,
        "protocol_fingerprint": execution_config.protocol_fingerprint,
        "configuration_sha256": analysis_configuration_sha256,
        "profile": execution_config.profile,
        "scientific_comparison_eligible": execution_config.scientific_comparison_eligible,
        "ordered_observation_id_sha256": observation_digest,
        "ordered_image_hash_sha256": image_digest,
        "ordered_outcome_sha256": outcome_digest,
        "ordered_group_sha256": group_digest,
        "split_assignment_sha256": split_assignment_digest,
        "ordered_covariate_sha256": covariate_digest,
        "row_count": len(manifest),
        "device": device,
        "batch_size": batch_size,
        "encoder_identities": {
            name: _declared_metadata(name, pretrained).to_dict() for name in encoder_names
        },
        "scientific_settings": scientific_settings,
        "environment": {
            "python": platform.python_version(),
            "operating_system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "dependencies": _dependency_versions(),
            "resources": resource_record,
        },
        "planned_units": planned_units,
    }
    checkpoint_store = (
        CheckpointStore(
            execution_config.checkpoint_dir,
            identity=run_identity,
            resume=execution_config.resume,
            run_id=execution_config.run_id,
            completed_run_policy=execution_config.completed_run_policy,
            stale_lock_timeout_seconds=execution_config.stale_lock_timeout_seconds,
            persistent_event_interval_seconds=(
                execution_config.persistent_event_interval_seconds
            ),
        )
        if execution_config.checkpoint_dir is not None
        else None
    )
    if checkpoint_store is not None:
        scope = _CHECKPOINT_SCOPE.get()
        if scope is not None:
            scope.append(checkpoint_store)
        progress.run_id = checkpoint_store.run_id
        progress.log_callback = checkpoint_store.record_event
    progress.emit(
        "STARTED",
        "analysis",
        "LookAgain-ML analysis started",
        total=len(planned_units),
        device=device,
        details={
            "profile": execution_config.profile,
            "scientific_comparison_eligible": (
                execution_config.scientific_comparison_eligible
            ),
            "configuration_sha256": analysis_configuration_sha256,
        },
    )
    cache_root = Path(cache_dir).expanduser().resolve()
    effective_embedding_cache_root = (
        checkpoint_store.root / "encoder_cache"
        if checkpoint_store is not None and not cache
        else cache_root
    )
    model_root = (
        Path(model_cache_dir).expanduser().resolve()
        if model_cache_dir is not None
        else cache_root / "models"
    )
    if provided_features and cache:
        warnings.warn(
            "cache=True does not copy externally supplied features into a LookAgain-ML cache. "
            "They remain labeled provided-verified and are revalidated on every call.",
            UserWarning,
            stacklevel=2,
        )
    representation_cache = (
        EmbeddingCache(effective_embedding_cache_root)
        if (cache or checkpoint_store is not None) and not provided_features
        else None
    )
    if (
        checkpoint_store is not None
        and execution_config.completed_run_policy == "reuse"
        and checkpoint_store.completed_result_candidates()
    ):
        progress.emit(
            "STARTED",
            "completed_result_compatibility",
            "verifying completed-run identity, caches, and result components",
            total=len(encoder_names),
        )
        restored_features: dict[str, np.ndarray] = {}
        restored_cache_records: dict[str, dict[str, Any]] = {}
        for position, name in enumerate(encoder_names, start=1):
            expected_metadata = _declared_metadata(name, pretrained)
            if provided_features:
                assert precomputed_features is not None
                assert precomputed_feature_audits is not None
                features, cache_record, _ = _validate_precomputed_representation(
                    name,
                    features_by_name=precomputed_features,
                    audits_by_name=precomputed_feature_audits,
                    expected_metadata=expected_metadata,
                    row_count=len(manifest),
                    observation_digest=observation_digest,
                    image_digest=image_digest,
                )
            else:
                assert representation_cache is not None
                cache_identity = build_cache_identity(
                    expected_metadata,
                    row_count=len(manifest),
                    ordered_observation_id_sha256=observation_digest,
                    ordered_image_hash_sha256=image_digest,
                    runtime_dependencies=_cache_runtime_dependencies(expected_metadata),
                )
                cached = representation_cache.load(cache_identity)
                if cached is None:
                    raise InputValidationError(
                        f"Completed result cannot be returned because the verified {name!r} "
                        "representation cache is missing. Use a new run identifier for recomputation."
                    )
                features, cache_record = cached.features, cached.metadata
            unit = checkpoint_store.load_unit(f"encoder:{name}", record_reuse=False)
            if unit is None or unit.get("cache_sha256") != cache_record.get("array_sha256"):
                raise InputValidationError(
                    f"Completed result cannot be returned because encoder checkpoint {name!r} "
                    "is missing or does not match its verified representation cache."
                )
            restored_features[name] = features
            restored_cache_records[name] = cache_record
            progress.emit(
                "VERIFYING",
                "completed_result_compatibility",
                "completed encoder provenance verified",
                encoder=name,
                current=position,
                total=len(encoder_names),
                cache_status="reused",
                details={"cache_sha256": cache_record.get("array_sha256")},
            )
        candidate_path, stored_reference = checkpoint_store.completed_result_candidates()[0]
        reference = stored_reference or result_checkpoint_reference(
            candidate_path, relative_to=checkpoint_store.root
        )
        if stored_reference is not None:
            verify_reference_metadata(candidate_path, stored_reference)
        progress.emit(
            "STARTED",
            "result_loading",
            "loading verified completed scientific result",
        )
        restored = load_result_checkpoint(
            candidate_path,
            identity_sha256=checkpoint_store.identity_sha256,
            protocol_identifier=execution_config.protocol_identifier,
            protocol_fingerprint=execution_config.protocol_fingerprint,
            configuration_sha256=analysis_configuration_sha256,
            row_count=len(manifest),
            representation_features=restored_features,
        )
        for name in encoder_names:
            recorded = (restored.cache_metadata or {}).get(name, {})
            if recorded.get("array_sha256") != restored_cache_records[name].get(
                "array_sha256"
            ):
                raise InputValidationError(
                    f"Completed result cache provenance for {name!r} is incompatible."
                )
        progress.emit(
            "REUSED",
            "result_loading",
            "verified completed scientific result reconstructed",
            details={"result_checkpoint_schema": reference["checkpoint_schema_version"]},
        )
        progress.emit(
            "COMPLETED",
            "analysis",
            "LookAgain-ML completed result returned without encoding or fitting",
            current=len(planned_units),
            total=len(planned_units),
            details={"result_disposition": "returned_verified_completed_run"},
        )
        checkpoint_store.complete_reuse(reference)
        return restored
    compute_metrics = metric_function(inferred_task)
    image_paths = manifest["image_path"].astype(str).tolist()
    expected_hashes = manifest["image_sha256"].astype(str).tolist()
    encoder_results: dict[str, Any] = {}
    predictions_by_encoder: dict[str, np.ndarray] = {}
    heads_by_encoder: dict[str, Any] = {}
    cache_by_encoder: dict[str, dict[str, Any]] = {}
    cost_by_encoder: dict[str, dict[str, Any]] = {}
    features_by_encoder: dict[str, np.ndarray] = {}

    for name in encoder_names:
        encoder_started = time.perf_counter()
        expected_metadata = _declared_metadata(name, pretrained)
        progress.emit(
            "STARTED",
            "encoder",
            "encoder processing started",
            encoder=name,
            checkpoint=expected_metadata.checkpoint,
            checkpoint_revision=expected_metadata.checkpoint_revision,
            device=device,
            total=len(image_paths),
        )
        identity = build_cache_identity(
            expected_metadata,
            row_count=len(manifest),
            ordered_observation_id_sha256=observation_digest,
            ordered_image_hash_sha256=image_digest,
            runtime_dependencies=_cache_runtime_dependencies(expected_metadata),
        )
        cached = representation_cache.load(identity) if representation_cache else None
        if provided_features:
            assert precomputed_features is not None and precomputed_feature_audits is not None
            actual_metadata = expected_metadata
            features, cache_record, encoder_costs = _validate_precomputed_representation(
                name,
                features_by_name=precomputed_features,
                audits_by_name=precomputed_feature_audits,
                expected_metadata=expected_metadata,
                row_count=len(manifest),
                observation_digest=observation_digest,
                image_digest=image_digest,
            )
            progress.emit(
                "REUSED",
                "image_encoding",
                "verified supplied representation reused",
                encoder=name,
                checkpoint=actual_metadata.checkpoint,
                checkpoint_revision=actual_metadata.checkpoint_revision,
                device="provided-precomputed-no-inference",
                rows_processed=len(manifest),
                feature_dimension=actual_metadata.feature_dimension,
                cache_status="provided-verified",
                representation_source="provided-verified",
                details={"cache_sha256": cache_record["array_sha256"]},
            )
        elif cached is not None:
            LOGGER.info(
                "Reused verified representation cache for %s (key %s).",
                name,
                cached.metadata["cache_key"],
            )
            features = cached.features
            cache_record = cached.metadata
            creation_costs = cache_record.get("creation_costs", {})
            encoder_costs: dict[str, Any] = {
                "device": "cache-hit-no-inference",
                "parameter_count": None,
                "extraction_seconds": 0.0,
                "images_per_second": None,
                "feature_dimension": expected_metadata.feature_dimension,
                "peak_gpu_memory_bytes": None,
                "cached_creation_device": creation_costs.get("device"),
                "cached_creation_parameter_count": creation_costs.get("parameter_count"),
                "cached_creation_parameter_count_scope": creation_costs.get("parameter_count_scope"),
                "cached_creation_loaded_parameter_count": creation_costs.get("loaded_parameter_count"),
                "cached_creation_extraction_seconds": creation_costs.get("extraction_seconds"),
                "cached_creation_images_per_second": creation_costs.get("images_per_second"),
                "cached_creation_peak_gpu_memory_bytes": creation_costs.get("peak_gpu_memory_bytes"),
            }
            actual_metadata = expected_metadata
            progress.emit(
                "REUSED",
                "image_encoding",
                "verified representation cache reused",
                encoder=name,
                checkpoint=actual_metadata.checkpoint,
                checkpoint_revision=actual_metadata.checkpoint_revision,
                device="cache-hit-no-inference",
                rows_processed=len(manifest),
                feature_dimension=actual_metadata.feature_dimension,
                cache_status="reused",
                location=str(effective_embedding_cache_root),
                representation_source="verified-cache",
                details={"cache_sha256": cache_record["array_sha256"]},
            )
        else:
            encoder_kwargs: dict[str, Any] = {"pretrained": pretrained, "device": device}
            if get_registration(name).metadata.backend in {"torchvision", "transformers"}:
                encoder_kwargs["model_cache_dir"] = model_root
            progress.emit(
                "STARTED",
                "checkpoint_loading",
                "downloading the pinned checkpoint if absent and loading the encoder",
                encoder=name,
                checkpoint=expected_metadata.checkpoint,
                checkpoint_revision=expected_metadata.checkpoint_revision,
                device=device,
            )
            try:
                image_encoder = create_encoder(name, **encoder_kwargs)
            except BaseException as error:
                if checkpoint_store is not None:
                    checkpoint_store.fail_unit(f"encoder:{name}", error)
                progress.emit(
                    "FAILED",
                    "encoder_creation",
                    "encoder creation failed",
                    encoder=name,
                    checkpoint=expected_metadata.checkpoint,
                    checkpoint_revision=expected_metadata.checkpoint_revision,
                    device=device,
                    details={
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    },
                )
                raise
            actual_metadata = image_encoder.metadata
            progress.emit(
                "CREATED",
                "encoder_creation",
                "encoder loaded and ready",
                encoder=name,
                checkpoint=actual_metadata.checkpoint,
                checkpoint_revision=actual_metadata.checkpoint_revision,
                device=str(getattr(image_encoder, "costs", {}).get("device", device)),
            )
            if actual_metadata.to_dict() != expected_metadata.to_dict():
                raise RuntimeError(
                    f"Encoder {name!r} runtime metadata differs from its registered cache identity; "
                    "refusing to create an unverifiable representation cache."
                )
            if hasattr(image_encoder, "set_batch_progress_callback"):
                image_encoder.set_batch_progress_callback(
                    encoder_batch_callback(progress, encoder=name)
                )
            progress.emit(
                "ENCODING",
                "image_encoding",
                "frozen image encoding started",
                encoder=name,
                checkpoint=actual_metadata.checkpoint,
                checkpoint_revision=actual_metadata.checkpoint_revision,
                device=str(getattr(image_encoder, "costs", {}).get("device", device)),
                current=0,
                total=len(image_paths),
                rows_processed=0,
            )
            try:
                features = image_encoder.encode(image_paths, batch_size=batch_size)
            except BaseException as error:
                if checkpoint_store is not None:
                    checkpoint_store.fail_unit(f"encoder:{name}", error)
                progress.emit(
                    "FAILED",
                    "image_encoding",
                    "frozen image encoding failed",
                    encoder=name,
                    checkpoint=actual_metadata.checkpoint,
                    checkpoint_revision=actual_metadata.checkpoint_revision,
                    device=str(getattr(image_encoder, "costs", {}).get("device", device)),
                    elapsed_seconds=time.perf_counter() - encoder_started,
                    details={
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    },
                )
                raise
            encoder_costs = dict(getattr(image_encoder, "costs", {}))
            del image_encoder
            gc.collect()
            if str(encoder_costs.get("device", "")).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
            _verify_image_hashes(
                image_paths,
                expected_hashes,
                phase=f"while {name} was encoded",
                progress=progress,
            )
            if representation_cache:
                saved = representation_cache.save(
                    identity, features, creation_costs=encoder_costs
                )
                features, cache_record = saved.features, saved.metadata
                LOGGER.info(
                    "Created verified representation cache for %s (key %s).",
                    name,
                    cache_record["cache_key"],
                )
                progress.emit(
                    "CHECKPOINTED",
                    "image_encoding",
                    "verified representation cache committed atomically",
                    encoder=name,
                    checkpoint=actual_metadata.checkpoint,
                    checkpoint_revision=actual_metadata.checkpoint_revision,
                    device=str(encoder_costs.get("device", device)),
                    rows_processed=len(manifest),
                    feature_dimension=actual_metadata.feature_dimension,
                    cache_status="created",
                    location=str(effective_embedding_cache_root),
                    representation_source="fresh-inference",
                    details={"cache_sha256": cache_record["array_sha256"]},
                )
            else:
                cache_record = {
                    "status": "disabled",
                    "cache_key": None,
                    "cache_schema_version": None,
                    "array_sha256": None,
                    "cache_bytes": 0,
                    "verified_fields": [],
                }
        expected_shape = (len(manifest), actual_metadata.feature_dimension)
        if features.shape != expected_shape or not np.isfinite(features).all():
            raise RuntimeError(
                f"Encoder {name!r} output does not match finite expected shape {expected_shape}."
            )
        if checkpoint_store is not None:
            checkpoint_path = checkpoint_store.save_unit(
                f"encoder:{name}",
                {
                    "encoder": name,
                    "metadata": actual_metadata.to_dict(),
                    "rows_processed": len(manifest),
                    "feature_dimension": actual_metadata.feature_dimension,
                    "cache_sha256": cache_record.get("array_sha256"),
                },
            )
            progress.emit(
                "CHECKPOINTED",
                "encoder",
                "completed encoder representation recorded atomically",
                encoder=name,
                checkpoint=actual_metadata.checkpoint,
                checkpoint_revision=actual_metadata.checkpoint_revision,
                rows_processed=len(manifest),
                feature_dimension=actual_metadata.feature_dimension,
                cache_status=cache_record["status"],
                location=str(checkpoint_path),
                representation_source=(
                    "newly-created"
                    if cache_record["status"] == "created"
                    else "safely-reused"
                    if cache_record["status"] in {"reused", "provided-verified"}
                    else "uncached"
                ),
                details={"cache_sha256": cache_record.get("array_sha256")},
            )
        progress.emit(
            "COMPLETED",
            "encoder",
            "encoder representation completed",
            encoder=name,
            checkpoint=actual_metadata.checkpoint,
            checkpoint_revision=actual_metadata.checkpoint_revision,
            device=str(encoder_costs.get("device", device)),
            rows_processed=len(manifest),
            feature_dimension=actual_metadata.feature_dimension,
            cache_status=cache_record["status"],
            elapsed_seconds=time.perf_counter() - encoder_started,
            location=(
                str(effective_embedding_cache_root)
                if cache_record.get("cache_key")
                else None
            ),
            representation_source=(
                "newly-created"
                if cache_record["status"] == "created"
                else "safely-reused"
                if cache_record["status"] in {"reused", "provided-verified"}
                else "uncached"
            ),
            details={"cache_sha256": cache_record.get("array_sha256")},
        )

        fit_started = time.perf_counter()
        try:
            with progress.heartbeat(
                "linear_head",
                "fitting validation-tuned downstream head",
                encoder=name,
            ):
                head = fit_linear_head(
                    inferred_task,
                    features[masks["train"]],
                    outcome[masks["train"]],
                    features[masks["validation"]],
                    outcome[masks["validation"]],
                    ridge_alphas=ridge_alpha_values,
                    logistic_cs=logistic_c_values,
                    random_state=random_state,
                    max_components=head_max_components,
                    pca_iterated_power=head_pca_iterated_power,
                    logistic_max_iter=logistic_max_iter,
                    working_dtype=head_working_dtype,
                )
                prediction = head.predict(features)
        except BaseException as error:
            if checkpoint_store is not None:
                checkpoint_store.fail_unit(f"linear_head:{name}", error)
            raise
        progress.emit(
            "COMPLETED",
            "linear_head",
            "validation-tuned downstream head completed",
            encoder=name,
            elapsed_seconds=time.perf_counter() - fit_started,
        )
        encoder_costs["downstream_fit_and_prediction_seconds"] = time.perf_counter() - fit_started
        validation_metrics = compute_metrics(
            outcome[masks["validation"]], prediction[masks["validation"]]
        )
        encoder_results[name] = {
            "metadata": actual_metadata.to_dict(),
            "validation_metrics": validation_metrics,
            "selected_hyperparameters": head.selected_hyperparameter,
            "selection_metric": head.selection_metric,
            "validation_grid": head.validation_grid,
            "decision_audit": fitted_linear_head_audit(head),
            "costs": encoder_costs,
            "cache": cache_record,
        }
        predictions_by_encoder[name] = prediction
        features_by_encoder[name] = features
        heads_by_encoder[name] = head
        cache_by_encoder[name] = cache_record
        cost_by_encoder[name] = encoder_costs

    primary_metric = "mse" if inferred_task == "regression" else "log_loss"
    selected_encoder = _select_encoder(encoder_names, encoder_results, primary_metric)
    validation_predictions = {
        name: prediction[masks["validation"]]
        for name, prediction in predictions_by_encoder.items()
    }
    prediction_combination_methods = [
        name for name in combination_methods if name != "feature_concatenation"
    ]
    with progress.heartbeat(
        "image_combination",
        "fitting validation-only image combination methods",
    ):
        fitted_combinations, combination_audit = fit_validation_combinations(
            inferred_task,
            validation_predictions,
            outcome[masks["validation"]],
            encoder_names,
            prediction_combination_methods,
            random_state=random_state,
            stack_alpha=stack_alpha,
            stack_c=stack_c,
        )
    combination_predictions: dict[str, np.ndarray] = {}
    combination_results: dict[str, Any] = {}
    for name, fitted in fitted_combinations.items():
        prediction = fitted.predict(predictions_by_encoder)
        combination_predictions[name] = prediction
        combination_results[name] = {
            "metadata": fitted.metadata(),
            "validation_metrics": compute_metrics(
                outcome[masks["validation"]], prediction[masks["validation"]]
            ),
            "validation_audit": combination_audit[name],
        }
    if "feature_concatenation" in combination_methods:
        fitted_union, union_audit = run_with_heartbeat(
            progress,
            "feature_concatenation",
            "fitting train-only feature concatenation",
            fit_feature_concatenation,
            inferred_task,
            {
                name: features[masks["train"]]
                for name, features in features_by_encoder.items()
            },
            outcome[masks["train"]],
            {
                name: features[masks["validation"]]
                for name, features in features_by_encoder.items()
            },
            outcome[masks["validation"]],
            encoder_names,
            max_components_per_encoder=feature_concatenation_max_components,
            ridge_alphas=ridge_alpha_values,
            logistic_cs=logistic_c_values,
            random_state=random_state,
        )
        union_prediction = fitted_union.predict(features_by_encoder)
        combination_predictions["feature_concatenation"] = union_prediction
        combination_results["feature_concatenation"] = {
            "metadata": fitted_union.metadata(),
            "validation_metrics": compute_metrics(
                outcome[masks["validation"]], union_prediction[masks["validation"]]
            ),
            "validation_audit": union_audit,
        }
    progress.emit(
        "COMPLETED",
        "image_combination",
        "validation-only image combination fitting completed",
    )
    decision_order = [selected_encoder, *combination_methods]
    validation_losses = {
        selected_encoder: encoder_results[selected_encoder]["validation_metrics"][
            primary_metric
        ],
        **{
            name: combination_results[name]["validation_metrics"][primary_metric]
            for name in combination_methods
        },
    }
    validation_preferred_method = min(
        decision_order, key=lambda name: validation_losses[name]
    )

    covariate_fit = None
    if has_covariates:
        image_source_prediction = (
            predictions_by_encoder[selected_encoder]
            if validation_preferred_method == selected_encoder
            else combination_predictions[validation_preferred_method]
        )
        joint_feature_encoders = joint_image_encoder_scope(
            validation_preferred_method,
            selected_encoder,
            combination_results,
        )
        with progress.heartbeat(
            "structured_and_integration",
            "fitting structured and X/I/X+I candidates",
        ):
            covariate_fit = fit_covariate_analysis(
                take_covariate_rows(aligned_covariates, masks["train"]),
                take_covariate_rows(aligned_covariates, masks["validation"]),
                aligned_covariates,
                outcome[masks["train"]],
                outcome[masks["validation"]],
                manifest.loc[masks["train"], "effective_group_id"].to_numpy(object),
                manifest.loc[masks["validation"], "effective_group_id"].to_numpy(object),
                manifest["split"].to_numpy(str),
                image_source_prediction[masks["validation"]],
                image_source_prediction,
                {
                    name: features[masks["train"]]
                    for name, features in features_by_encoder.items()
                },
                {
                    name: features[masks["validation"]]
                    for name, features in features_by_encoder.items()
                },
                features_by_encoder,
                task=inferred_task,
                structured_model_names=structured_model_names,
                integration_method_names=integration_method_names,
                image_source_method=validation_preferred_method,
                image_feature_encoders=joint_feature_encoders,
                rf_score_encoder=rf_score_encoder or selected_encoder,
                missing_indicators=numeric_missing_indicators,
                preprocessed_numeric_matrix=covariates_preprocessed,
                ridge_alphas=ridge_alpha_values,
                logistic_cs=logistic_c_values,
                random_state=random_state,
                score_alpha=score_integration_alpha,
                score_c=score_integration_c,
                joint_image_max_components=joint_image_max_components,
                nn_max_iter=nn_max_iter,
                rf_structured_configurations=structured_rf_values,
                rf_score_configurations=score_rf_values,
                rf_seeds=rf_seed_values,
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
        progress.emit(
            "COMPLETED",
            "structured_and_integration",
            "structured and X/I/X+I candidate fitting completed",
        )

    # No test outcome is read by a fitting or decision function above this line.
    progress.emit(
        "STARTED",
        "locked_test_evaluation",
        "evaluating the locked test set after freezing every fitted choice",
    )
    for name in encoder_names:
        encoder_results[name]["test_metrics"] = compute_metrics(
            outcome[masks["test"]], predictions_by_encoder[name][masks["test"]]
        )
    for name in combination_methods:
        combination_results[name]["test_metrics"] = compute_metrics(
            outcome[masks["test"]], combination_predictions[name][masks["test"]]
        )
        combination_results[name]["validation_preferred"] = (
            name == validation_preferred_method
        )
    covariate_results = None
    integration_results = None
    xi_predictions = None
    if covariate_fit is not None:
        selected_x_prediction = covariate_fit.structured_predictions[
            covariate_fit.selected_structured_model
        ]
        image_source_prediction = (
            predictions_by_encoder[selected_encoder]
            if covariate_fit.image_source_method == selected_encoder
            else combination_predictions[covariate_fit.image_source_method]
        )
        for name, record in covariate_fit.structured_records.items():
            record["test_metrics"] = compute_metrics(
                outcome[masks["test"]],
                covariate_fit.structured_predictions[name][masks["test"]],
            )
            record["validation_selected"] = name == covariate_fit.selected_structured_model
        x_test_metrics = covariate_fit.structured_records[
            covariate_fit.selected_structured_model
        ]["test_metrics"]
        image_test_metrics = compute_metrics(
            outcome[masks["test"]], image_source_prediction[masks["test"]]
        )
        for name, record in covariate_fit.integration_records.items():
            combined_metrics = compute_metrics(
                outcome[masks["test"]],
                covariate_fit.integration_predictions[name][masks["test"]],
            )
            record["test_metrics"] = combined_metrics
            record["incremental_metrics_vs_structured"] = incremental_metrics(
                inferred_task, x_test_metrics, combined_metrics
            )
            record["validation_selected"] = name == covariate_fit.selected_integration_method
        selected_xi_metrics = covariate_fit.integration_records[
            covariate_fit.selected_integration_method
        ]["test_metrics"]
        covariate_results = {
            "preprocessing": covariate_fit.preprocessor.metadata(),
            "models": covariate_fit.structured_records,
            "selected_model": covariate_fit.selected_structured_model,
            "selection_partition": "validation",
            "test_outcomes_used_for_preprocessing_or_selection": False,
            "interpretation": "X is the supplied structured information set.",
        }
        integration_results = {
            "methods": covariate_fit.integration_records,
            "selected_method": covariate_fit.selected_integration_method,
            "default_method": "linear_score",
            "selection_partition": "validation",
            "image_score_source": covariate_fit.image_source_method,
            "joint_image_feature_encoders": covariate_fit.image_feature_encoders,
            "joint_image_feature_scope": (
                "Matched to the encoder membership of the validation-selected "
                "image-only procedure."
            ),
            "x_i_x_plus_i": {
                "X_structured_only": x_test_metrics,
                "I_image_only": image_test_metrics,
                "X_plus_I_selected": selected_xi_metrics,
                "selected_incremental_metrics": incremental_metrics(
                    inferred_task, x_test_metrics, selected_xi_metrics
                ),
            },
            "test_outcomes_used_for_weights_or_selection": False,
            "model_flexibility_note": (
                "Neural X+I integration was explicitly requested while the structured "
                "baseline menu remained linear-only; interpret any gain with that "
                "model-flexibility difference visible."
                if any(name in {"nn_score", "joint_nn"} for name in integration_method_names)
                and "nn" not in structured_model_names
                else "Structured and integration model menus are recorded explicitly."
            ),
            "scientific_guardrails": [
                "Image-only prediction does not establish incremental information beyond X.",
                "A negative fitted incremental R2 does not imply negative population information.",
                "Generated image scores are predictive quantities, not causal variables.",
                "Optional neural or joint integration is not guaranteed to improve performance.",
            ],
        }
        xi_predictions = {
            "structured_only": selected_x_prediction,
            "image_only": image_source_prediction,
            **covariate_fit.integration_predictions,
        }
    progress.emit(
        "COMPLETED",
        "locked_test_evaluation",
        "locked-test evaluation completed without tuning or selection",
    )
    selected_entry = encoder_results[selected_encoder]
    selected_head = heads_by_encoder[selected_encoder]
    selected_metrics = {
        "validation": selected_entry["validation_metrics"],
        "test": selected_entry["test_metrics"],
    }
    predictions = _prediction_frame(
        manifest, raw_target, predictions_by_encoder[selected_encoder], inferred_task, classes
    )

    logme_results = None
    if logme:
        scores = {
            name: logme_score(
                features_by_encoder[name][masks["train"]],
                outcome[masks["train"]],
                task=inferred_task,
                max_components=logme_max_components,
                random_state=random_state,
            )
            for name in encoder_names
        }
        ranked = sorted(encoder_names, key=lambda name: scores[name], reverse=True)
        logme_results = {
            "partition": "train",
            "uses_validation_outcomes": False,
            "uses_test_outcomes": False,
            "scores": scores,
            "preprocessing": (
                "training-only standardization and PCA when needed, capped at "
                f"{logme_max_components} components"
            ),
            "ranking": ranked,
            "top_encoder": ranked[0],
            "matches_validation_selected_encoder": ranked[0] == selected_encoder,
            "interpretation": (
                "Optional training-only screening diagnostic; LogME does not replace "
                "task-specific validation selection."
            ),
        }

    repeated_results = None
    if repeated_splits > 1:
        repeated_results = run_with_heartbeat(
            progress,
            "repeated_splits",
            "running repeated image-only split analyses",
            repeated_split_analysis,
            features_by_encoder,
            outcome,
            manifest["user_group_id"].to_numpy(object),
            manifest["image_sha256"].to_numpy(object),
            manifest["observation_id"].to_numpy(object),
            task=inferred_task,
            repetitions=repeated_splits,
            encoder_order=encoder_names,
            combination_methods=combination_methods,
            random_state=random_state,
            ridge_alphas=ridge_alpha_values,
            logistic_cs=logistic_c_values,
            stack_alpha=stack_alpha,
            stack_c=stack_c,
            feature_concatenation_max_components=(
                feature_concatenation_max_components
            ),
            head_max_components=head_max_components,
            head_pca_iterated_power=head_pca_iterated_power,
            logistic_max_iter=logistic_max_iter,
            head_working_dtype=head_working_dtype,
            checkpoint_store=checkpoint_store,
            progress_reporter=progress,
        )
        if has_covariates:
            repeated_results["x_i_x_plus_i"] = run_with_heartbeat(
                progress,
                "repeated_xi_splits",
                "running repeated X/I/X+I split analyses",
                repeated_xi_analysis,
                aligned_covariates,
                features_by_encoder,
                outcome,
                manifest["user_group_id"].to_numpy(object),
                manifest["image_sha256"].to_numpy(object),
                manifest["observation_id"].to_numpy(object),
                task=inferred_task,
                repetitions=repeated_splits,
                encoder_order=encoder_names,
                combination_methods=combination_methods,
                structured_model_names=structured_model_names,
                integration_method_names=integration_method_names,
                missing_indicators=numeric_missing_indicators,
                preprocessed_numeric_matrix=covariates_preprocessed,
                random_state=random_state,
                ridge_alphas=ridge_alpha_values,
                logistic_cs=logistic_c_values,
                stack_alpha=stack_alpha,
                stack_c=stack_c,
                feature_concatenation_max_components=feature_concatenation_max_components,
                score_alpha=score_integration_alpha,
                score_c=score_integration_c,
                joint_image_max_components=joint_image_max_components,
                nn_max_iter=nn_max_iter,
                rf_score_encoder=rf_score_encoder,
                rf_structured_configurations=structured_rf_values,
                rf_score_configurations=score_rf_values,
                rf_seeds=rf_seed_values,
                rf_n_estimators=rf_n_estimators,
                rf_score_folds=rf_score_folds,
                rf_score_image_max_components=rf_score_image_max_components,
                rf_n_jobs=rf_n_jobs,
                rf_score_reduction_random_state=rf_score_reduction_random_state,
                rf_score_fold_random_state=rf_score_fold_random_state,
                rf_score_standardize_reduced_scores=rf_score_standardize_reduced_scores,
                rf_score_refit_image_reduction=rf_score_refit_image_reduction,
                rf_score_working_dtype=rf_score_working_dtype,
                head_max_components=head_max_components,
                head_pca_iterated_power=head_pca_iterated_power,
                logistic_max_iter=logistic_max_iter,
                head_working_dtype=head_working_dtype,
                checkpoint_store=checkpoint_store,
                progress_reporter=progress,
            )

    uncertainty = None
    if bootstrap_repetitions:
        test_predictions = {
            selected_encoder: predictions_by_encoder[selected_encoder][masks["test"]],
            **{
                name: combination_predictions[name][masks["test"]]
                for name in combination_methods
            },
        }
        if not combination_methods:
            test_predictions = {
                name: predictions_by_encoder[name][masks["test"]]
                for name in encoder_names
            }
        if covariate_fit is not None:
            test_predictions.update(
                {
                    "structured_only": xi_predictions["structured_only"][masks["test"]],
                    **{
                        name: prediction[masks["test"]]
                        for name, prediction in covariate_fit.integration_predictions.items()
                    },
                }
            )
            comparisons = [
                (name, "structured_only") for name in integration_method_names
            ]
        else:
            comparisons = [
                (name, selected_encoder)
                for name in test_predictions
                if name != selected_encoder
            ]
        metric_names = (
            list(bootstrap_metrics)
            if bootstrap_metrics is not None
            else (
                ["r2", "mse", "mae"]
                if inferred_task == "regression"
                else ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]
            )
        )
        uncertainty = run_with_heartbeat(
            progress,
            "locked_test_uncertainty",
            "computing paired locked-test uncertainty",
            paired_locked_test_bootstrap,
            outcome[masks["test"]],
            test_predictions,
            manifest.loc[masks["test"], "effective_group_id"].to_numpy(object),
            task=inferred_task,
            comparisons=comparisons,
            metrics=metric_names,
            repetitions=bootstrap_repetitions,
            random_state=random_state + 40,
        )

    audit = manifest_result.audit.to_dict()
    audit.update(
        {
            "exact_duplicate_rows": int(manifest["is_exact_duplicate"].sum()),
            "exact_duplicate_hashes": int((manifest["image_sha256"].value_counts() > 1).sum()),
            "split_counts": {
                key: int(value) for key, value in manifest["split"].value_counts().items()
            },
            "ordered_observation_id_sha256": observation_digest,
            "ordered_image_hash_sha256": image_digest,
            "split_assignment_sha256": digest_strings(
                (manifest["observation_id"].astype(str) + "|" + manifest["split"].astype(str)).tolist()
            ),
        }
    )
    qc = {
        "status": "PASS",
        "group_split_overlap": 0,
        "hash_split_overlap": 0,
        "all_features_finite": True,
        "manifest_feature_rows_aligned": True,
        "scaling_fit_train_only": True,
        "hyperparameters_selected_validation_only": True,
        "encoder_selected_validation_only": True,
        "encoder_selection_metric": primary_metric,
        "encoder_tie_break": "first requested encoder",
        "test_outcomes_used_for_selection": False,
        "encoder_selection_frozen_before_test_evaluation": True,
        "combination_weights_selected_validation_only": (
            True if combination_methods else None
        ),
        "combination_decisions_frozen_before_test_evaluation": (
            True if combination_methods else None
        ),
        "repeated_split_decisions_validation_only": (
            True if repeated_splits > 1 else None
        ),
        "logme_training_only": True if logme else None,
        "structured_preprocessing_fit_train_only": True if has_covariates else None,
        "structured_model_selected_validation_only": True if has_covariates else None,
        "integration_selected_validation_only": True if has_covariates else None,
        "integration_decisions_frozen_before_test_evaluation": True if has_covariates else None,
        "split_labels_user_supplied": requested_splits is not None,
    }
    costs = {**cost_by_encoder[selected_encoder], "by_encoder": cost_by_encoder}
    configuration = {
        "package": {
            "name": IMPORT_NAME,
            "distribution_name": DISTRIBUTION_NAME,
            "import_name": IMPORT_NAME,
            "version": __version__,
        },
        "runtime": {
            "python": platform.python_version(),
            "operating_system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "dependencies": _dependency_versions(),
            "resources": resource_record,
        },
        "analysis": {
            "task_requested": task.strip().lower(),
            "task_resolved": inferred_task,
            "encoders": encoder_names,
            "selected_encoder": selected_encoder,
            "selection_partition": "validation",
            "selection_metric": primary_metric,
            "device_requested": device,
            "batch_size": batch_size,
            "pretrained": pretrained,
            "cache_requested": cache_requested,
            "cache_enabled": (cache or checkpoint_store is not None) and not provided_features,
            "representation_source": (
                "provided-verified frozen features" if provided_features else "encoder inference/cache"
            ),
            "cache_location": "user-local; absolute path intentionally omitted",
            "model_cache_location": "user-local; absolute path intentionally omitted",
            "strict": strict,
            "verify_images": verify_images,
            "split_labels_user_supplied": requested_splits is not None,
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "ridge_alphas": ridge_alpha_values,
            "logistic_cs": logistic_c_values,
            "head_max_components": head_max_components,
            "head_pca_iterated_power": head_pca_iterated_power,
            "logistic_max_iter": logistic_max_iter,
            "head_working_dtype": head_working_dtype,
            "combinations": combination_methods,
            "validation_preferred_method": validation_preferred_method,
            "repeated_splits": repeated_splits,
            "bootstrap_repetitions": bootstrap_repetitions,
            "logme": logme,
            "logme_max_components": logme_max_components,
            "stack_alpha": stack_alpha,
            "stack_c": stack_c,
            "feature_concatenation_max_components": (
                feature_concatenation_max_components
            ),
            "covariates_supplied": has_covariates,
            "covariates_preprocessed": covariates_preprocessed if has_covariates else None,
            "numeric_missing_indicators": numeric_missing_indicators if has_covariates else None,
            "structured_models": structured_model_names,
            "integration_methods": integration_method_names,
            "rf_score_encoder": (
                rf_score_encoder or selected_encoder
                if "rf_score" in integration_method_names
                else None
            ),
            "rf_structured_configurations": structured_rf_values,
            "rf_score_configurations": score_rf_values,
            "rf_seeds": rf_seed_values,
            "rf_n_estimators": rf_n_estimators,
            "rf_score_folds": rf_score_folds,
            "rf_score_image_max_components": rf_score_image_max_components,
            "rf_n_jobs": rf_n_jobs,
            "rf_score_reduction_random_state": rf_score_reduction_random_state,
            "rf_score_fold_random_state": rf_score_fold_random_state,
            "rf_score_standardize_reduced_scores": rf_score_standardize_reduced_scores,
            "rf_score_refit_image_reduction": rf_score_refit_image_reduction,
            "rf_score_working_dtype": rf_score_working_dtype,
            "advanced_config": advanced_config.to_dict() if advanced_config else None,
            "execution_profile": execution_config.profile,
            "scientific_comparison_eligible": (
                execution_config.scientific_comparison_eligible
            ),
            "diagnostic_only": execution_config.profile == "quick",
            "protocol_identifier": execution_config.protocol_identifier,
            "protocol_fingerprint": execution_config.protocol_fingerprint,
            "analysis_configuration_sha256": analysis_configuration_sha256,
            "progress_mode": execution_config.progress,
            "checkpointing_enabled": checkpoint_store is not None,
            "resume_enabled": execution_config.resume,
            "selected_structured_model": (
                covariate_fit.selected_structured_model if covariate_fit else None
            ),
            "selected_integration_method": (
                covariate_fit.selected_integration_method if covariate_fit else None
            ),
            "score_integration_alpha": score_integration_alpha,
            "score_integration_c": score_integration_c,
            "joint_image_max_components": joint_image_max_components,
            "nn_max_iter": nn_max_iter,
        },
        "random_seeds": {"split_and_downstream_head": random_state},
    }
    result = LookAgainResults(
        task=inferred_task,
        selected_encoder=selected_encoder,
        encoder_results=encoder_results,
        metrics=selected_metrics,
        selected_hyperparameters=selected_head.selected_hyperparameter,
        manifest=manifest,
        predictions=predictions,
        audit=audit,
        qc=qc,
        costs=costs,
        configuration=configuration,
        classes=classes,
        fitted_head=selected_head,
        cache_metadata=cache_by_encoder,
        stack_results=(
            {
                "methods": combination_results,
                "validation_preferred_method": validation_preferred_method,
                "selection_partition": "validation",
                "test_outcomes_used_for_weights_or_selection": False,
            }
            if combination_methods
            else None
        ),
        repeated_split_results=repeated_results,
        uncertainty=uncertainty,
        logme_results=logme_results,
        method_predictions={**combination_predictions},
        covariate_results=covariate_results,
        integration_results=integration_results,
        fitted_covariate_preprocessor=(
            covariate_fit.preprocessor if covariate_fit else None
        ),
        xi_predictions=xi_predictions,
        representation_features=features_by_encoder,
    )
    result_reference = None
    if checkpoint_store is not None:
        progress.emit(
            "STARTED",
            "result_checkpoint",
            "serializing completed scientific result transactionally",
        )
        reference = write_result_checkpoint(
            checkpoint_store.result_checkpoint_path(),
            result,
            identity_sha256=checkpoint_store.identity_sha256,
            protocol_identifier=execution_config.protocol_identifier,
            protocol_fingerprint=execution_config.protocol_fingerprint,
            configuration_sha256=analysis_configuration_sha256,
        )
        result_reference = checkpoint_store.link_result_checkpoint(reference)
        progress.emit(
            "CHECKPOINTED",
            "result_checkpoint",
            "completed scientific result verified and promoted atomically",
            location=str(checkpoint_store.result_checkpoint_path()),
            details={
                "result_checkpoint_schema": result_reference[
                    "checkpoint_schema_version"
                ],
                "metadata_sha256": result_reference["metadata_sha256"],
            },
        )
    progress.emit(
        "COMPLETED",
        "analysis",
        "LookAgain-ML analysis completed",
        current=len(planned_units),
        total=len(planned_units),
        device=str(cost_by_encoder[selected_encoder].get("device", device)),
        details={
            "selected_encoder": selected_encoder,
            "test_outcomes_used_for_selection": False,
        },
    )
    if checkpoint_store is not None:
        assert result_reference is not None
        checkpoint_store.complete(
            output_locations=[
                checkpoint_store.manifest_path,
                checkpoint_store.log_path,
                checkpoint_store.result_checkpoint_path(),
            ],
            result_reference=result_reference,
        )
    return result


@functools.wraps(_analyze_impl)
def analyze(*args: Any, **kwargs: Any) -> LookAgainResults:
    """Run :func:`_analyze_impl` and release every checkpoint writer lease."""

    stores: list[CheckpointStore] = []
    token = _CHECKPOINT_SCOPE.set(stores)
    error: BaseException | None = None
    try:
        return _analyze_impl(*args, **kwargs)
    except BaseException as caught:
        error = caught
        raise
    finally:
        for store in reversed(stores):
            store.close(error)
        _CHECKPOINT_SCOPE.reset(token)
