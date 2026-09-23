"""Safe transactional checkpoints for completed :class:`LookAgainResults`."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._version import RESULT_SCHEMA_VERSION, __version__
from .exceptions import CheckpointValidationError
from .heads import FittedLinearHead
from .results import LookAgainResults
from .structured import FittedStructuredPreprocessor

RESULT_CHECKPOINT_SCHEMA_VERSION = "1"


def _public_payload(result: LookAgainResults) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "package_version": __version__,
        "task": result.task,
        "selected_encoder": result.selected_encoder,
        "encoder_results": result.encoder_results,
        "metrics": result.metrics,
        "selected_hyperparameters": result.selected_hyperparameters,
        "audit": result.audit,
        "qc": result.qc,
        "costs": result.costs,
        "configuration": result.configuration,
        "classes": result.classes,
        "stack_results": result.stack_results,
        "covariate_results": result.covariate_results,
        "integration_results": result.integration_results,
        "repeated_split_results": result.repeated_split_results,
        "uncertainty": result.uncertainty,
        "cache_metadata": result.cache_metadata,
        "logme_results": result.logme_results,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise CheckpointValidationError(
            "Completed results contain a non-finite value and cannot be checkpointed."
        )
    if value is pd.NA or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CheckpointValidationError(
            f"Rejected completed result: {label} is unreadable or partial ({error})."
        ) from error


def _write_array(path: Path, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise CheckpointValidationError(
            f"Completed result array {path.name} must have a non-object numeric dtype."
        )
    if not np.isfinite(array).all():
        raise CheckpointValidationError(
            f"Completed result array {path.name} contains non-finite values."
        )
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _read_array(path: Path, label: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as error:
        raise CheckpointValidationError(
            f"Rejected completed result: {label} array is unreadable ({error})."
        ) from error
    if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number):
        raise CheckpointValidationError(
            f"Rejected completed result: {label} has an unsupported array dtype."
        )
    if not np.isfinite(value).all():
        raise CheckpointValidationError(
            f"Rejected completed result: {label} contains non-finite values."
        )
    return value


def _dataframe_payload(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise CheckpointValidationError(
            "Completed result DataFrame contains non-finite numeric values."
        )
    return {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "data": [[_json_value(value) for value in row] for row in frame.itertuples(index=False, name=None)],
    }


def _dataframe_from_payload(value: Any, label: str) -> pd.DataFrame:
    if not isinstance(value, dict) or set(value) != {"columns", "dtypes", "data"}:
        raise CheckpointValidationError(
            f"Rejected completed result: {label} table schema is invalid."
        )
    columns = value["columns"]
    dtypes = value["dtypes"]
    data = value["data"]
    if not isinstance(columns, list) or not isinstance(dtypes, list) or len(columns) != len(dtypes):
        raise CheckpointValidationError(
            f"Rejected completed result: {label} table columns/dtypes are invalid."
        )
    frame = pd.DataFrame(data, columns=columns)
    for column, dtype in zip(columns, dtypes):
        try:
            frame[column] = frame[column].astype(dtype)
        except (TypeError, ValueError) as error:
            raise CheckpointValidationError(
                f"Rejected completed result: {label}.{column} cannot restore dtype {dtype}."
            ) from error
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise CheckpointValidationError(
            f"Rejected completed result: {label} contains non-finite numeric values."
        )
    return frame


def _write_array_mapping(root: Path, stem: str, value: dict[str, np.ndarray] | None) -> None:
    index: dict[str, Any] = {"present": value is not None, "items": []}
    if value is not None:
        for position, (name, array) in enumerate(value.items()):
            filename = f"{stem}_{position:04d}.npy"
            _write_array(root / filename, np.asarray(array))
            index["items"].append({"name": str(name), "filename": filename})
    _write_json(root / f"{stem}.json", index)


def _read_array_mapping(root: Path, stem: str) -> dict[str, np.ndarray] | None:
    index = _read_json(root / f"{stem}.json", stem)
    if not isinstance(index, dict) or set(index) != {"present", "items"}:
        raise CheckpointValidationError(
            f"Rejected completed result: {stem} index is invalid."
        )
    if not index["present"]:
        if index["items"]:
            raise CheckpointValidationError(
                f"Rejected completed result: absent {stem} contains array entries."
            )
        return None
    result: dict[str, np.ndarray] = {}
    for item in index["items"]:
        if not isinstance(item, dict) or set(item) != {"name", "filename"}:
            raise CheckpointValidationError(
                f"Rejected completed result: {stem} item is invalid."
            )
        filename = str(item["filename"])
        if Path(filename).name != filename or not filename.startswith(f"{stem}_"):
            raise CheckpointValidationError(
                f"Rejected completed result: {stem} contains an unsafe filename."
            )
        name = str(item["name"])
        if name in result:
            raise CheckpointValidationError(
                f"Rejected completed result: duplicate {stem} key {name!r}."
            )
        result[name] = _read_array(root / filename, f"{stem}.{name}")
    return result


def _write_fitted_head(root: Path, head: Any) -> None:
    if head is None:
        _write_json(root / "fitted_head.json", {"present": False})
        return
    if not isinstance(head, FittedLinearHead):
        raise CheckpointValidationError(
            "The completed result contains an unsupported fitted-head type."
        )
    scaler = head.pipeline.named_steps["scale"]
    pca = head.pipeline.named_steps.get("pca")
    model_name = "ridge" if head.task == "regression" else "logistic"
    model = head.pipeline.named_steps[model_name]
    arrays = {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_var": scaler.var_,
        "model_coef": model.coef_,
        "model_intercept": np.atleast_1d(model.intercept_),
    }
    if pca is not None:
        arrays.update(
            {
                "pca_components": pca.components_,
                "pca_mean": pca.mean_,
                "pca_explained_variance": pca.explained_variance_,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_,
                "pca_singular_values": pca.singular_values_,
            }
        )
    if head.task == "classification":
        arrays["model_classes"] = model.classes_
        arrays["model_n_iter"] = model.n_iter_
    for name, array in arrays.items():
        _write_array(root / f"fitted_head_{name}.npy", np.asarray(array))
    _write_json(
        root / "fitted_head.json",
        {
            "present": True,
            "task": head.task,
            "selected_hyperparameter": head.selected_hyperparameter,
            "validation_grid": head.validation_grid,
            "selection_metric": head.selection_metric,
            "working_dtype": head.working_dtype,
            "scaler_n_features_in": int(scaler.n_features_in_),
            "scaler_n_samples_seen": int(np.asarray(scaler.n_samples_seen_).max()),
            "pca": (
                {
                    "present": True,
                    "n_components": int(pca.n_components_),
                    "n_features_in": int(pca.n_features_in_),
                    "n_samples": int(pca.n_samples_),
                    "noise_variance": float(pca.noise_variance_),
                    "svd_solver": str(pca.svd_solver),
                    "iterated_power": int(pca.iterated_power),
                    "random_state": pca.random_state,
                }
                if pca is not None
                else {"present": False}
            ),
            "model_n_features_in": int(model.n_features_in_),
        },
    )


def _read_fitted_head(root: Path) -> FittedLinearHead | None:
    value = _read_json(root / "fitted_head.json", "fitted_head")
    if not isinstance(value, dict) or "present" not in value:
        raise CheckpointValidationError("Rejected completed result: fitted-head schema is invalid.")
    if not value["present"]:
        return None
    scaler = StandardScaler()
    scaler.mean_ = _read_array(root / "fitted_head_scaler_mean.npy", "scaler_mean")
    scaler.scale_ = _read_array(root / "fitted_head_scaler_scale.npy", "scaler_scale")
    scaler.var_ = _read_array(root / "fitted_head_scaler_var.npy", "scaler_var")
    scaler.n_features_in_ = int(value["scaler_n_features_in"])
    scaler.n_samples_seen_ = int(value["scaler_n_samples_seen"])
    steps: list[tuple[str, Any]] = [("scale", scaler)]
    pca_value = value["pca"]
    if pca_value.get("present"):
        pca = PCA(
            n_components=int(pca_value["n_components"]),
            svd_solver=str(pca_value["svd_solver"]),
            iterated_power=int(pca_value["iterated_power"]),
            random_state=pca_value["random_state"],
        )
        pca.components_ = _read_array(root / "fitted_head_pca_components.npy", "pca_components")
        pca.mean_ = _read_array(root / "fitted_head_pca_mean.npy", "pca_mean")
        pca.explained_variance_ = _read_array(root / "fitted_head_pca_explained_variance.npy", "pca_explained_variance")
        pca.explained_variance_ratio_ = _read_array(root / "fitted_head_pca_explained_variance_ratio.npy", "pca_explained_variance_ratio")
        pca.singular_values_ = _read_array(root / "fitted_head_pca_singular_values.npy", "pca_singular_values")
        pca.n_components_ = int(pca_value["n_components"])
        pca.n_features_in_ = int(pca_value["n_features_in"])
        pca.n_samples_ = int(pca_value["n_samples"])
        pca.noise_variance_ = float(pca_value["noise_variance"])
        steps.append(("pca", pca))
    if value["task"] == "regression":
        model: Any = Ridge(alpha=float(value["selected_hyperparameter"]["alpha"]))
        model.coef_ = _read_array(root / "fitted_head_model_coef.npy", "model_coef")
        model.intercept_ = float(_read_array(root / "fitted_head_model_intercept.npy", "model_intercept")[0])
        model_name = "ridge"
    elif value["task"] == "classification":
        model = LogisticRegression(C=float(value["selected_hyperparameter"]["C"]))
        model.coef_ = _read_array(root / "fitted_head_model_coef.npy", "model_coef")
        model.intercept_ = _read_array(root / "fitted_head_model_intercept.npy", "model_intercept")
        model.classes_ = _read_array(root / "fitted_head_model_classes.npy", "model_classes")
        model.n_iter_ = _read_array(root / "fitted_head_model_n_iter.npy", "model_n_iter")
        model_name = "logistic"
    else:
        raise CheckpointValidationError("Rejected completed result: unsupported fitted-head task.")
    model.n_features_in_ = int(value["model_n_features_in"])
    steps.append((model_name, model))
    return FittedLinearHead(
        task=str(value["task"]),
        pipeline=Pipeline(steps),
        selected_hyperparameter=dict(value["selected_hyperparameter"]),
        validation_grid=list(value["validation_grid"]),
        selection_metric=str(value["selection_metric"]),
        working_dtype=str(value["working_dtype"]),
    )


def _write_preprocessor(root: Path, value: Any) -> None:
    if value is None:
        payload = {"present": False}
    elif isinstance(value, FittedStructuredPreprocessor):
        payload = {"present": True, "state": asdict(value)}
    else:
        raise CheckpointValidationError(
            "The completed result contains an unsupported structured preprocessor."
        )
    _write_json(root / "fitted_covariate_preprocessor.json", payload)


def _read_preprocessor(root: Path) -> FittedStructuredPreprocessor | None:
    value = _read_json(
        root / "fitted_covariate_preprocessor.json", "fitted_covariate_preprocessor"
    )
    if not isinstance(value, dict) or "present" not in value:
        raise CheckpointValidationError(
            "Rejected completed result: structured-preprocessor schema is invalid."
        )
    return FittedStructuredPreprocessor(**value["state"]) if value["present"] else None


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "result_checkpoint.json"
    }


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(12):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            transient = getattr(error, "winerror", None) in {5, 32, 33} or error.errno in {
                errno.EACCES,
                errno.EBUSY,
            }
            if not transient or attempt == 11:
                raise
            time.sleep(min(0.1 * (attempt + 1), 1.0))


def _verify_inventory(root: Path, metadata: dict[str, Any]) -> None:
    inventory = metadata.get("components")
    if not isinstance(inventory, dict) or not inventory:
        raise CheckpointValidationError(
            "Rejected completed result: component inventory is missing."
        )
    actual = {path.name for path in root.iterdir() if path.is_file()} - {"result_checkpoint.json"}
    if actual != set(inventory):
        raise CheckpointValidationError(
            "Rejected completed result: component inventory is incomplete or contains extras."
        )
    for filename, record in inventory.items():
        if Path(filename).name != filename or not isinstance(record, dict):
            raise CheckpointValidationError(
                "Rejected completed result: component inventory contains an unsafe entry."
            )
        path = root / filename
        if path.stat().st_size != record.get("size_bytes"):
            raise CheckpointValidationError(
                f"Rejected completed result: {filename} size changed."
            )
        if _sha256_file(path) != record.get("sha256"):
            raise CheckpointValidationError(
                f"Rejected completed result: {filename} checksum failed."
            )


def load_result_checkpoint(
    directory: str | Path,
    *,
    identity_sha256: str,
    protocol_identifier: str | None,
    protocol_fingerprint: str | None,
    configuration_sha256: str,
    row_count: int,
    representation_features: dict[str, np.ndarray] | None,
) -> LookAgainResults:
    """Verify all files and reconstruct a completed result without executable data."""

    root = Path(directory).expanduser().resolve()
    metadata_path = root / "result_checkpoint.json"
    if not root.is_dir() or not metadata_path.is_file():
        raise CheckpointValidationError(
            f"Completed result checkpoint is missing or incomplete at {root}."
        )
    metadata = _read_json(metadata_path, "result checkpoint metadata")
    expected = {
        "checkpoint_schema_version": RESULT_CHECKPOINT_SCHEMA_VERSION,
        "public_result_schema_version": RESULT_SCHEMA_VERSION,
        "package_version": __version__,
        "identity_sha256": identity_sha256,
        "protocol_identifier": protocol_identifier,
        "protocol_fingerprint": protocol_fingerprint,
        "configuration_sha256": configuration_sha256,
        "row_count": int(row_count),
        "status": "COMPLETED",
    }
    mismatches = [key for key, expected_value in expected.items() if metadata.get(key) != expected_value]
    if mismatches:
        raise CheckpointValidationError(
            "Rejected completed result: incompatible " + ", ".join(mismatches) + "."
        )
    if metadata.get("contains_frozen_embedding_arrays") is not False:
        raise CheckpointValidationError(
            "Rejected completed result: frozen embeddings must not be stored in result checkpoints."
        )
    _verify_inventory(root, metadata)
    public = _read_json(root / "public_payload.json", "public payload")
    if not isinstance(public, dict):
        raise CheckpointValidationError("Rejected completed result: public payload is invalid.")
    if public.pop("schema_version", None) != RESULT_SCHEMA_VERSION or public.pop(
        "package_version", None
    ) != __version__:
        raise CheckpointValidationError(
            "Rejected completed result: public result schema/package version is incompatible."
        )
    manifest = _dataframe_from_payload(
        _read_json(root / "manifest.json", "manifest"), "manifest"
    )
    predictions = _dataframe_from_payload(
        _read_json(root / "predictions.json", "predictions"), "predictions"
    )
    if len(manifest) != row_count or len(predictions) != row_count:
        raise CheckpointValidationError(
            "Rejected completed result: manifest or prediction row count is invalid."
        )
    required_manifest = {"observation_id", "image_sha256", "split"}
    required_predictions = {"observation_id", "split", "y_true", "y_pred"}
    if not required_manifest.issubset(manifest) or not required_predictions.issubset(predictions):
        raise CheckpointValidationError(
            "Rejected completed result: required manifest/prediction columns are missing."
        )
    if manifest["observation_id"].astype(str).tolist() != predictions[
        "observation_id"
    ].astype(str).tolist():
        raise CheckpointValidationError(
            "Rejected completed result: prediction rows are misaligned."
        )
    return LookAgainResults(
        **public,
        manifest=manifest,
        predictions=predictions,
        method_predictions=_read_array_mapping(root, "method_predictions"),
        xi_predictions=_read_array_mapping(root, "xi_predictions"),
        fitted_head=_read_fitted_head(root),
        fitted_covariate_preprocessor=_read_preprocessor(root),
        representation_features=representation_features,
    )


def write_result_checkpoint(
    directory: str | Path,
    result: LookAgainResults,
    *,
    identity_sha256: str,
    protocol_identifier: str | None,
    protocol_fingerprint: str | None,
    configuration_sha256: str,
) -> dict[str, Any]:
    """Write to a private temporary directory, verify, then atomically promote."""

    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise CheckpointValidationError(
            f"Refusing to overwrite completed result checkpoint {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex[:12]}.partial"
    temporary.mkdir()
    preserve_failed_partial = False
    try:
        _write_json(temporary / "public_payload.json", _public_payload(result))
        _write_json(temporary / "manifest.json", _dataframe_payload(result.manifest))
        _write_json(temporary / "predictions.json", _dataframe_payload(result.predictions))
        _write_array_mapping(temporary, "method_predictions", result.method_predictions)
        _write_array_mapping(temporary, "xi_predictions", result.xi_predictions)
        _write_fitted_head(temporary, result.fitted_head)
        _write_preprocessor(temporary, result.fitted_covariate_preprocessor)
        metadata = {
            "checkpoint_schema_version": RESULT_CHECKPOINT_SCHEMA_VERSION,
            "public_result_schema_version": RESULT_SCHEMA_VERSION,
            "package_version": __version__,
            "status": "COMPLETED",
            "identity_sha256": identity_sha256,
            "protocol_identifier": protocol_identifier,
            "protocol_fingerprint": protocol_fingerprint,
            "configuration_sha256": configuration_sha256,
            "row_count": len(result.manifest),
            "created_at_utc": _utc_now(),
            "contains_frozen_embedding_arrays": False,
            "components": _inventory(temporary),
        }
        _write_json(temporary / "result_checkpoint.json", metadata)
        restored = load_result_checkpoint(
            temporary,
            identity_sha256=identity_sha256,
            protocol_identifier=protocol_identifier,
            protocol_fingerprint=protocol_fingerprint,
            configuration_sha256=configuration_sha256,
            row_count=len(result.manifest),
            representation_features=result.representation_features,
        )
        if restored.to_dict() != result.to_dict():
            raise CheckpointValidationError(
                "Completed result checkpoint did not reproduce the public result payload."
            )
        try:
            _replace_with_retry(temporary, destination)
        except BaseException:
            preserve_failed_partial = True
            raise
        final_metadata = destination / "result_checkpoint.json"
        return {
            "checkpoint_schema_version": RESULT_CHECKPOINT_SCHEMA_VERSION,
            "path": destination.name,
            "metadata_sha256": _sha256_file(final_metadata),
            "components": metadata["components"],
            "row_count": len(result.manifest),
            "contains_frozen_embedding_arrays": False,
        }
    finally:
        if (
            not preserve_failed_partial
            and temporary.exists()
            and temporary.parent == destination.parent
            and temporary.name.endswith(".partial")
        ):
            shutil.rmtree(temporary)


def verify_reference_metadata(directory: str | Path, reference: dict[str, Any]) -> None:
    root = Path(directory).expanduser().resolve()
    metadata = root / "result_checkpoint.json"
    if not metadata.is_file() or _sha256_file(metadata) != reference.get("metadata_sha256"):
        raise CheckpointValidationError(
            "Completed result metadata is missing or its manifest-linked checksum failed."
        )
    payload = _read_json(metadata, "result checkpoint metadata")
    for key in (
        "checkpoint_schema_version",
        "components",
        "row_count",
        "contains_frozen_embedding_arrays",
    ):
        if reference.get(key) != payload.get(key):
            raise CheckpointValidationError(
                f"Completed result manifest link has incompatible {key}."
            )


def result_checkpoint_reference(
    directory: str | Path, *, relative_to: str | Path
) -> dict[str, Any]:
    """Build a verified portable reference for an already promoted checkpoint."""

    root = Path(directory).expanduser().resolve()
    base = Path(relative_to).expanduser().resolve()
    try:
        relative = root.relative_to(base).as_posix()
    except ValueError as error:
        raise CheckpointValidationError(
            "Completed result checkpoint is outside its run directory."
        ) from error
    metadata = _read_json(root / "result_checkpoint.json", "result checkpoint metadata")
    _verify_inventory(root, metadata)
    return {
        "checkpoint_schema_version": metadata.get("checkpoint_schema_version"),
        "path": relative,
        "metadata_sha256": _sha256_file(root / "result_checkpoint.json"),
        "components": metadata.get("components"),
        "row_count": metadata.get("row_count"),
        "contains_frozen_embedding_arrays": metadata.get(
            "contains_frozen_embedding_arrays"
        ),
    }
