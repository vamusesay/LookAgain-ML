"""Content-addressed, auditable caches for frozen image representations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._version import DISTRIBUTION_NAME, IMPORT_NAME, __version__
from .encoders.base import EncoderMetadata
from .exceptions import CacheValidationError

CACHE_SCHEMA_VERSION = "4"
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _canonical(value: Any) -> Any:
    """Normalize tuples and NumPy scalars through the JSON data model."""

    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _open_atomic_partial(path: Path, mode: str):
    """Open one new partial file, tolerating one transient directory loss."""

    for attempt in range(2):
        try:
            if "b" in mode:
                return path.open(mode)
            return path.open(mode, encoding="utf-8", newline="\n")
        except FileNotFoundError as error:
            if attempt:
                raise CacheValidationError(
                    f"Could not create atomic cache partial file in {path.parent}. "
                    "Check that the cache directory is writable and stable. On Windows, "
                    "choose a shorter cache_dir if long-path support is disabled."
                ) from error
            path.parent.mkdir(parents=True, exist_ok=True)
    raise AssertionError("unreachable")


def build_cache_identity(
    metadata: EncoderMetadata,
    *,
    row_count: int,
    ordered_observation_id_sha256: str,
    ordered_image_hash_sha256: str,
    runtime_dependencies: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Build the complete identity that must match before features are reused."""

    return _canonical(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "distribution_name": DISTRIBUTION_NAME,
            "import_name": IMPORT_NAME,
            "package_version": __version__,
            "encoder": metadata.to_dict(),
            "runtime_dependencies": runtime_dependencies or {},
            "row_count": int(row_count),
            "ordered_observation_id_sha256": ordered_observation_id_sha256,
            "ordered_image_hash_sha256": ordered_image_hash_sha256,
            "feature_dimension": int(metadata.feature_dimension),
            "dtype": "float32",
        }
    )


def _identity_key(identity: dict[str, Any]) -> str:
    serialized = json.dumps(_canonical(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheOutcome:
    features: np.ndarray
    metadata: dict[str, Any]


class EmbeddingCache:
    """Verify and atomically persist one exact representation matrix per identity."""

    def __init__(self, directory: str | Path) -> None:
        self.root = Path(directory).expanduser().resolve()

    def _paths(self, identity: dict[str, Any]) -> tuple[str, Path, Path]:
        encoder_name = str(identity["encoder"]["name"])
        if not _SAFE_NAME.fullmatch(encoder_name):
            raise CacheValidationError(
                f"Encoder name {encoder_name!r} is unsafe for a cache directory."
            )
        key = _identity_key(identity)
        folder = self.root / "embeddings" / encoder_name
        return key, folder / f"{key}.npy", folder / f"{key}.json"

    def load(self, identity: dict[str, Any]) -> CacheOutcome | None:
        """Return a verified cache hit, a clean miss, or raise on partial/corrupt state."""

        expected = _canonical(identity)
        key, array_path, metadata_path = self._paths(expected)
        array_exists = array_path.is_file()
        metadata_exists = metadata_path.is_file()
        if not array_exists and not metadata_exists:
            return None
        if array_exists != metadata_exists:
            raise CacheValidationError(
                f"Rejected representation cache {key}: its array/metadata pair is incomplete. "
                "Remove that cache entry and rerun extraction."
            )
        try:
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise CacheValidationError(
                f"Rejected representation cache {key}: metadata is unreadable ({error})."
            ) from error
        if stored.get("identity") != expected:
            raise CacheValidationError(
                f"Rejected representation cache {key}: stored identity does not match the "
                "current image order, hashes, encoder, checkpoint, preprocessing, dimension, "
                "schema, and package version."
            )
        try:
            features = np.load(array_path, allow_pickle=False)
        except Exception as error:
            raise CacheValidationError(
                f"Rejected representation cache {key}: feature array is unreadable ({error})."
            ) from error
        expected_shape = (expected["row_count"], expected["feature_dimension"])
        if features.shape != expected_shape or features.dtype != np.float32:
            raise CacheValidationError(
                f"Rejected representation cache {key}: found shape/dtype "
                f"{features.shape}/{features.dtype}; expected {expected_shape}/float32."
            )
        if not np.isfinite(features).all():
            raise CacheValidationError(
                f"Rejected representation cache {key}: features contain non-finite values."
            )
        observed_sha256 = _sha256_file(array_path)
        if observed_sha256 != stored.get("array_sha256"):
            raise CacheValidationError(
                f"Rejected representation cache {key}: the feature-array checksum failed."
            )
        return CacheOutcome(
            features=np.asarray(features, dtype=np.float32),
            metadata={
                "status": "reused",
                "cache_key": key,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "array_sha256": observed_sha256,
                "cache_bytes": int(array_path.stat().st_size + metadata_path.stat().st_size),
                "verified_fields": sorted(expected),
                "creation_costs": stored.get("creation_costs", {}),
            },
        )

    def save(
        self,
        identity: dict[str, Any],
        features: np.ndarray,
        *,
        creation_costs: dict[str, Any] | None = None,
    ) -> CacheOutcome:
        """Atomically save a finite float32 matrix and its verification metadata."""

        normalized = _canonical(identity)
        key, array_path, metadata_path = self._paths(normalized)
        matrix = np.asarray(features, dtype=np.float32)
        expected_shape = (normalized["row_count"], normalized["feature_dimension"])
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise CacheValidationError(
                f"Refused to cache shape {matrix.shape}; expected finite {expected_shape}."
            )
        array_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep partial names short enough for Windows installations where the
        # final content-addressed path is valid but legacy MAX_PATH handling is
        # still active. The final filenames retain the complete identity key.
        nonce = f"{os.getpid():x}-{uuid.uuid4().hex[:12]}"
        temporary_array = array_path.parent / f".tmp-{nonce}.partial.npy"
        temporary_metadata = metadata_path.parent / f".tmp-{nonce}.partial.json"
        try:
            with _open_atomic_partial(temporary_array, "wb") as stream:
                np.save(stream, matrix, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            array_sha256 = _sha256_file(temporary_array)
            payload = {
                "identity": normalized,
                "array_sha256": array_sha256,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "creation_costs": _canonical(creation_costs or {}),
            }
            with _open_atomic_partial(temporary_metadata, "w") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_array, array_path)
            os.replace(temporary_metadata, metadata_path)
        finally:
            for temporary in (temporary_array, temporary_metadata):
                if temporary.exists():
                    temporary.unlink()
        return CacheOutcome(
            features=matrix,
            metadata={
                "status": "created",
                "cache_key": key,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "array_sha256": array_sha256,
                "cache_bytes": int(array_path.stat().st_size + metadata_path.stat().st_size),
                "verified_fields": sorted(normalized),
                "creation_costs": _canonical(creation_costs or {}),
            },
        )

    def get_or_compute(
        self,
        identity: dict[str, Any],
        compute: Callable[[], np.ndarray],
    ) -> CacheOutcome:
        hit = self.load(identity)
        if hit is not None:
            return hit
        return self.save(identity, compute())
