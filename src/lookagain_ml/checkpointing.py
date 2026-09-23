"""Atomic, identity-bound execution manifests and resumable unit checkpoints."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import socket
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .exceptions import CheckpointValidationError
from .progress import ProgressEvent, portable_path
from .protocols import canonical_configuration_json, configuration_sha256

_SAFE_UNIT = re.compile(r"[^a-zA-Z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transient_replace_error(error: OSError) -> bool:
    """Recognize bounded retry cases caused by file scanners/sync locks."""

    return getattr(error, "winerror", None) in {5, 32, 33} or error.errno in {
        errno.EACCES,
        errno.EBUSY,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _compatibility_identity(value: dict[str, Any]) -> dict[str, Any]:
    """Exclude per-attempt resource observations from stable run identity."""

    normalized = _json_safe(value)
    environment = normalized.get("environment")
    if isinstance(environment, dict):
        environment = dict(environment)
        environment.pop("resources", None)
        normalized = dict(normalized)
        normalized["environment"] = environment
    return normalized


def _validated_prior_identity_sha256(prior: dict[str, Any]) -> str:
    identity = prior.get("identity")
    stored = prior.get("identity_sha256")
    if not isinstance(identity, dict) or not isinstance(stored, str):
        raise CheckpointValidationError("Checkpoint run identity is missing or invalid.")
    accepted = {
        configuration_sha256(_json_safe(identity)),
        configuration_sha256(_compatibility_identity(identity)),
    }
    if stored not in accepted:
        raise CheckpointValidationError("Checkpoint run-identity checksum failed.")
    return stored


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                break
            except OSError as error:
                if not _transient_replace_error(error) or attempt == 11:
                    raise
                time.sleep(min(0.1 * (attempt + 1), 1.0))
    finally:
        if temporary.exists():
            temporary.unlink()


class _RunLock:
    """One conservative cross-platform exclusive writer lease."""

    def __init__(
        self,
        path: Path,
        *,
        attempt_id: str,
        stale_timeout_seconds: float | None,
    ) -> None:
        self.path = path
        self.attempt_id = attempt_id
        self.stale_timeout_seconds = stale_timeout_seconds
        self.acquired = False

    @staticmethod
    def _process_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True
        return True

    def _archive_stale_lock(self) -> bool:
        if self.stale_timeout_seconds is None:
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            age = time.time() - self.path.stat().st_mtime
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False
        if age < self.stale_timeout_seconds:
            return False
        if payload.get("hostname") != socket.gethostname():
            return False
        try:
            pid = int(payload["pid"])
        except (KeyError, TypeError, ValueError):
            return False
        if self._process_alive(pid):
            return False
        archive = self.path.parent / "stale_locks"
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / f"run-{uuid.uuid4().hex}.lock.json"
        os.replace(self.path, destination)
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "attempt_id": self.attempt_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at_utc": _utc_now(),
        }
        for attempt in range(2):
            try:
                with self.path.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                self.acquired = True
                return
            except FileExistsError as error:
                if attempt == 0 and self._archive_stale_lock():
                    continue
                raise CheckpointValidationError(
                    "Another process owns this run checkpoint directory. Wait for it to "
                    "finish or, for a demonstrably abandoned same-host lock, configure a "
                    "conservative stale_lock_timeout_seconds value."
                ) from error

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("attempt_id") == self.attempt_id:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.acquired = False


class CheckpointStore:
    """Persist stable run identity and immutable execution-attempt history."""

    def __init__(
        self,
        directory: str | Path,
        *,
        identity: dict[str, Any],
        resume: bool = True,
        run_id: str | None = None,
        completed_run_policy: str = "reuse",
        stale_lock_timeout_seconds: float | None = None,
        persistent_event_interval_seconds: float = 30.0,
    ) -> None:
        self.root = Path(directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = self.root / "checkpoints"
        self.manifest_path = self.root / "run_manifest.json"
        self.index_path = self.root / "run_index.json"
        self.log_path = self.root / "execution_events.jsonl"
        self.identity = _json_safe(identity)
        compatibility_identity = _compatibility_identity(self.identity)
        self.identity_sha256 = configuration_sha256(compatibility_identity)
        self.started_monotonic = time.perf_counter()
        self.attempt_id = uuid.uuid4().hex[:16]
        self.attempt_root = self.root / "attempts" / self.attempt_id
        self.attempt_manifest_path = self.attempt_root / "attempt_manifest.json"
        self._terminal = False
        self._closed = False
        self._last_persisted_batch: dict[tuple[str, str | None], float] = {}
        self.persistent_event_interval_seconds = float(persistent_event_interval_seconds)
        self.completed_run_policy = completed_run_policy
        if self.manifest_path.exists():
            preliminary_prior = self._read_json(self.manifest_path, "run manifest")
            prior_identity = preliminary_prior.get("identity")
            if (
                not isinstance(prior_identity, dict)
                or _compatibility_identity(prior_identity) != compatibility_identity
            ):
                raise CheckpointValidationError(
                    "Checkpoint configuration is incompatible with this run. The package "
                    "version, protocol, inputs/order, encoder identities, or relevant settings "
                    "changed; use a new checkpoint directory."
                )
            self.identity_sha256 = _validated_prior_identity_sha256(preliminary_prior)
        self.lock = _RunLock(
            self.root / "run.lock",
            attempt_id=self.attempt_id,
            stale_timeout_seconds=stale_lock_timeout_seconds,
        )
        self.lock.acquire()
        try:
            prior: dict[str, Any] | None = None
            if self.manifest_path.exists():
                if not resume:
                    raise CheckpointValidationError(
                        f"Run manifest already exists at {self.manifest_path}; enable resume "
                        "or select a new checkpoint directory."
                    )
                prior = self._read_json(self.manifest_path, "run manifest")
                prior_identity = prior.get("identity")
                if (
                    not isinstance(prior_identity, dict)
                    or _compatibility_identity(prior_identity) != compatibility_identity
                ):
                    raise CheckpointValidationError(
                        "Checkpoint configuration is incompatible with this run. The package "
                        "version, protocol, inputs/order, encoder identities, or relevant settings "
                        "changed; use a new checkpoint directory."
                    )
                if _validated_prior_identity_sha256(prior) != self.identity_sha256:
                    raise CheckpointValidationError("Checkpoint run identity changed while locking.")
                self.run_id = str(prior["run_id"])
                if run_id is not None and str(run_id) != self.run_id:
                    raise CheckpointValidationError(
                        "The requested run_id differs from the checkpoint's stable run identifier."
                    )
                self._preserve_legacy_attempt(prior)
            else:
                self.run_id = run_id or uuid.uuid4().hex
            self._protect_completed_root = bool(
                prior and prior.get("status") == "COMPLETED"
            )
            self._owns_root_manifest = not self._protect_completed_root
            self.manifest = self._new_attempt_manifest(prior)
            self.attempt_root.mkdir(parents=True, exist_ok=False)
            self._write_manifest()
            self._write_index(status="STARTED")
        except BaseException:
            self.lock.release()
            raise

    def _new_attempt_manifest(self, prior: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "status": "STARTED",
            "result_disposition": "pending",
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "package_version": __version__,
            "protocol_identifier": self.identity.get("protocol_identifier"),
            "protocol_fingerprint": self.identity.get("protocol_fingerprint"),
            "configuration_sha256": self.identity.get("configuration_sha256"),
            "profile": self.identity.get("profile", "standard"),
            "scientific_comparison_eligible": self.identity.get(
                "scientific_comparison_eligible", False
            ),
            "environment": self.identity.get("environment", {}),
            "device": self.identity.get("device"),
            "current_stage": "initialization",
            "last_completed_stage": None,
            "completed_units": [],
            "reused_units": [],
            "skipped_units": [],
            "failed_units": {},
            "pending_units": list(self.identity.get("planned_units", [])),
            "relevant_cache_hashes": {},
            "cache_units": {},
            "start_time_utc": _utc_now(),
            "completion_time_utc": None,
            "update_time_utc": _utc_now(),
            "elapsed_seconds": 0.0,
            "resumed": prior is not None,
            "prior_status": prior.get("status") if prior else None,
            "final_output_locations": [],
            "result_checkpoint": None,
        }

    def _preserve_legacy_attempt(self, prior: dict[str, Any]) -> None:
        prior_attempt = prior.get("attempt_id")
        if isinstance(prior_attempt, str):
            existing = self.root / "attempts" / prior_attempt / "attempt_manifest.json"
            if existing.is_file():
                return
        digest = configuration_sha256(prior)
        destination = self.root / "attempts" / f"legacy-{digest[:16]}" / "attempt_manifest.json"
        if not destination.exists():
            _atomic_json(destination, prior)

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "schema_version": 1,
                "run_id": self.run_id,
                "identity_sha256": self.identity_sha256,
                "attempts": [],
                "completed_result": None,
            }
        index = self._read_json(self.index_path, "run index")
        if index.get("identity_sha256") != self.identity_sha256:
            raise CheckpointValidationError("Run index identity is incompatible.")
        return index

    def _write_index(
        self,
        *,
        status: str,
        result_reference: dict[str, Any] | None = None,
    ) -> None:
        index = self._read_index()
        attempts = [
            item for item in index.setdefault("attempts", [])
            if item.get("attempt_id") != self.attempt_id
        ]
        attempts.append(
            {
                "attempt_id": self.attempt_id,
                "status": status,
                "result_disposition": self.manifest.get("result_disposition"),
                "manifest": self.attempt_manifest_path.relative_to(self.root).as_posix(),
                "start_time_utc": self.manifest.get("start_time_utc"),
                "completion_time_utc": self.manifest.get("completion_time_utc"),
            }
        )
        index["attempts"] = attempts
        index["updated_at_utc"] = _utc_now()
        if result_reference is not None:
            index["completed_result"] = result_reference
        _atomic_json(self.index_path, index)

    @staticmethod
    def _read_json(path: Path, label: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise CheckpointValidationError(
                f"Rejected {label} {path}: unreadable or partial JSON ({error})."
            ) from error
        if not isinstance(payload, dict):
            raise CheckpointValidationError(f"Rejected {label} {path}: expected a JSON object.")
        return payload

    def _unit_path(self, unit: str) -> Path:
        safe = _SAFE_UNIT.sub("_", unit).strip("._") or "unit"
        suffix = hashlib.sha256(unit.encode("utf-8")).hexdigest()[:12]
        return self.checkpoint_root / f"{safe}-{suffix}.json"

    def save_unit(self, unit: str, payload: dict[str, Any]) -> Path:
        normalized = _json_safe(payload)
        payload_sha256 = hashlib.sha256(
            canonical_configuration_json(normalized).encode("utf-8")
        ).hexdigest()
        envelope = {
            "schema_version": 1,
            "status": "COMPLETED",
            "unit": unit,
            "identity_sha256": self.identity_sha256,
            "package_version": __version__,
            "protocol_identifier": self.identity.get("protocol_identifier"),
            "protocol_fingerprint": self.identity.get("protocol_fingerprint"),
            "payload_sha256": payload_sha256,
            "completed_at_utc": _utc_now(),
            "payload": normalized,
        }
        path = self._unit_path(unit)
        if path.exists():
            existing = self.load_unit(unit)
            if existing != normalized:
                raise CheckpointValidationError(
                    f"Compatible completed checkpoint {unit!r} already exists with different "
                    "contents; refusing to overwrite it. Use a new checkpoint directory."
                )
            self._record_unit(unit, "completed")
            return path
        _atomic_json(path, envelope)
        self._record_unit(unit, "completed")
        return path

    def load_unit(self, unit: str, *, record_reuse: bool = True) -> dict[str, Any] | None:
        path = self._unit_path(unit)
        if not path.exists():
            return None
        envelope = self._read_json(path, f"checkpoint {unit!r}")
        if envelope.get("status") != "COMPLETED":
            raise CheckpointValidationError(
                f"Rejected checkpoint {unit!r}: status is not COMPLETED."
            )
        expected_fields = {
            "unit": unit,
            "identity_sha256": self.identity_sha256,
            "package_version": __version__,
            "protocol_identifier": self.identity.get("protocol_identifier"),
            "protocol_fingerprint": self.identity.get("protocol_fingerprint"),
        }
        mismatches = [
            name for name, expected in expected_fields.items() if envelope.get(name) != expected
        ]
        if mismatches:
            raise CheckpointValidationError(
                f"Rejected checkpoint {unit!r}: incompatible {', '.join(mismatches)}."
            )
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise CheckpointValidationError(
                f"Rejected checkpoint {unit!r}: payload is missing or invalid."
            )
        observed = hashlib.sha256(
            canonical_configuration_json(payload).encode("utf-8")
        ).hexdigest()
        if observed != envelope.get("payload_sha256"):
            raise CheckpointValidationError(
                f"Rejected checkpoint {unit!r}: payload checksum failed."
            )
        if record_reuse:
            self._record_unit(unit, "reused")
        return payload

    def completed_result_candidates(self) -> list[tuple[Path, dict[str, Any] | None]]:
        """Return manifest-linked then orphan-promoted compatible result candidates."""

        candidates: list[tuple[Path, dict[str, Any] | None]] = []
        seen: set[Path] = set()
        index = self._read_index()
        reference = index.get("completed_result")
        if isinstance(reference, dict) and isinstance(reference.get("path"), str):
            path = (self.root / reference["path"]).resolve()
            if path.is_relative_to(self.root):
                candidates.append((path, reference))
                seen.add(path)
        if self.manifest_path.exists():
            prior = self._read_json(self.manifest_path, "run manifest")
            reference = prior.get("result_checkpoint")
            if isinstance(reference, dict) and isinstance(reference.get("path"), str):
                path = (self.root / reference["path"]).resolve()
                if path.is_relative_to(self.root) and path not in seen:
                    candidates.append((path, reference))
                    seen.add(path)
        for path in sorted((self.root / "attempts").glob("*/result_checkpoint")):
            resolved = path.resolve()
            if resolved not in seen:
                candidates.append((resolved, None))
                seen.add(resolved)
        return candidates

    def result_checkpoint_path(self) -> Path:
        return self.attempt_root / "result_checkpoint"

    def link_result_checkpoint(self, reference: dict[str, Any]) -> dict[str, Any]:
        linked = dict(reference)
        linked["path"] = self.result_checkpoint_path().relative_to(self.root).as_posix()
        self.manifest["result_checkpoint"] = linked
        self.manifest["result_disposition"] = "newly_created"
        self._write_manifest()
        return linked

    def record_event(self, event: ProgressEvent) -> None:
        now = time.monotonic()
        if event.event in {"ENCODING", "VERIFYING"} and event.current is not None:
            key = (event.stage, event.encoder)
            last = self._last_persisted_batch.get(key)
            boundary = event.current in {0, 1, event.total}
            if not boundary and last is not None and now - last < self.persistent_event_interval_seconds:
                return
            self._last_persisted_batch[key] = now
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_json_safe(event.to_dict()), sort_keys=True, allow_nan=False)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
        self.manifest["current_stage"] = event.stage
        self.manifest["update_time_utc"] = event.timestamp_utc
        if event.event in {"COMPLETED", "CHECKPOINTED"}:
            self.manifest["last_completed_stage"] = event.stage
        if event.event == "FAILED":
            self.manifest["status"] = "FAILED"
            self.manifest["exception_type"] = event.details.get("exception_type")
            self.manifest["exception_message"] = event.details.get("exception_message")
        unit = event.details.get("unit")
        if unit and event.event == "SKIPPED":
            values = self.manifest.setdefault("skipped_units", [])
            if unit not in values:
                values.append(unit)
        if event.encoder and event.cache_status:
            self.manifest["cache_units"][event.encoder] = {
                "status": event.cache_status,
                "representation_source": event.representation_source,
                "location": portable_path(event.location, root=self.root)
                if event.location
                else None,
            }
            if event.details.get("cache_sha256"):
                self.manifest["relevant_cache_hashes"][event.encoder] = event.details[
                    "cache_sha256"
                ]
        self._write_manifest()

    def fail_unit(self, unit: str, error: BaseException) -> None:
        self.manifest["failed_units"][unit] = {
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "updated_at_utc": _utc_now(),
        }
        self.manifest["status"] = "FAILED"
        self._write_manifest()

    def complete(
        self,
        *,
        output_locations: list[str | Path] | None = None,
        result_reference: dict[str, Any] | None = None,
        result_disposition: str = "newly_created",
    ) -> None:
        if result_reference is None:
            raise CheckpointValidationError(
                "A run cannot be marked COMPLETED without a verified result checkpoint."
            )
        self.manifest["status"] = "COMPLETED"
        self.manifest["result_disposition"] = result_disposition
        self.manifest["current_stage"] = "completed"
        self.manifest["last_completed_stage"] = "completed"
        self.manifest["completion_time_utc"] = _utc_now()
        self.manifest["result_checkpoint"] = result_reference
        self.manifest["final_output_locations"] = [
            portable_path(path, root=self.root) for path in (output_locations or [])
        ]
        self._write_manifest()
        self._write_index(status="COMPLETED", result_reference=result_reference)
        self._terminal = True

    def complete_reuse(self, reference: dict[str, Any], *, reconstructed: bool = True) -> None:
        self.complete(
            output_locations=[self.log_path],
            result_reference=reference,
            result_disposition=(
                "returned_verified_completed_run" if reconstructed else "reused"
            ),
        )

    def _record_unit(self, unit: str, category: str) -> None:
        key = f"{category}_units"
        values = self.manifest.setdefault(key, [])
        if unit not in values:
            values.append(unit)
        if category in {"completed", "reused"}:
            pending = self.manifest.setdefault("pending_units", [])
            if unit in pending:
                pending.remove(unit)
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.manifest["update_time_utc"] = _utc_now()
        self.manifest["elapsed_seconds"] = time.perf_counter() - self.started_monotonic
        _atomic_json(self.attempt_manifest_path, self.manifest)
        if self._owns_root_manifest:
            _atomic_json(self.manifest_path, self.manifest)

    def close(self, error: BaseException | None = None) -> None:
        if self._closed:
            return
        try:
            if not self._terminal:
                if self.manifest.get("status") != "FAILED":
                    if isinstance(error, KeyboardInterrupt):
                        self.manifest["status"] = "INTERRUPTED"
                    elif error is not None:
                        self.manifest["status"] = "FAILED"
                    else:
                        self.manifest["status"] = "INCOMPLETE"
                if error is not None:
                    self.manifest["exception_type"] = type(error).__name__
                    self.manifest["exception_message"] = str(error)
                self.manifest["completion_time_utc"] = _utc_now()
                self._write_manifest()
                self._write_index(status=self.manifest["status"])
                self._terminal = True
        finally:
            self.lock.release()
            self._closed = True

    def __del__(self) -> None:
        with suppress(OSError, CheckpointValidationError, AttributeError):
            self.close()

    def portable_manifest(self) -> dict[str, Any]:
        """Return a copy with absolute path values redacted for export."""

        def redact(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, str) and Path(value).is_absolute():
                return f"<redacted>/{Path(value).name}"
            return value

        return redact(self.manifest)
