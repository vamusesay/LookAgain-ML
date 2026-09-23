"""Replication-facing compatibility wrapper over package checkpoint support."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from lookagain_ml.checkpointing import CheckpointStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def completed_output_manifest(root: Path) -> dict[str, dict[str, Any]]:
    """Hash completed adapter outputs, excluding mutable execution state."""

    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(
            part in {"adapter_state", "run_state"} for part in relative.parts
        ):
            continue
        records[relative.as_posix()] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    return records


def validate_completed_outputs(
    root: Path, expected: dict[str, dict[str, Any]]
) -> None:
    """Reject missing or changed files before adapter-level reuse."""

    for relative, record in expected.items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise RuntimeError(
                f"Completed adapter output {relative!r} is missing or changed; use a "
                "new output directory rather than overwriting completed evidence."
            )


class ProgressLedger(CheckpointStore):
    """Retain the Prompt 3 ledger API using the shared atomic checkpoint system."""

    def __init__(
        self,
        output_root: Path,
        *,
        configuration_sha256: str,
        protocol_identifier: str | None = None,
        protocol_fingerprint: str | None = None,
        profile: str = "full",
    ) -> None:
        self.configuration_sha256 = str(configuration_sha256)
        super().__init__(
            output_root,
            identity={
                "configuration_sha256": self.configuration_sha256,
                "protocol_identifier": protocol_identifier,
                "protocol_fingerprint": protocol_fingerprint,
                "profile": profile,
                "scientific_comparison_eligible": profile == "paper",
                "planned_units": [],
            },
            resume=True,
        )

    def is_complete(self, unit: str, *, protocol_fingerprint: str) -> bool:
        payload = self.completed_payload(unit, protocol_fingerprint=protocol_fingerprint)
        return payload is not None

    def completed_payload(
        self, unit: str, *, protocol_fingerprint: str
    ) -> dict[str, Any] | None:
        """Return a verified compatible completion payload, if present."""

        payload = self.load_unit(unit)
        if (
            payload is not None
            and payload.get("status") == "PASS"
            and payload.get("protocol_fingerprint") == protocol_fingerprint
        ):
            return payload
        if payload is not None and payload.get("status") == "PASS":
            raise RuntimeError(
                "Completed adapter checkpoint has a different protocol fingerprint; "
                "use a new output directory."
            )
        return None

    def update(
        self,
        unit: str,
        *,
        status: str,
        protocol_fingerprint: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "status": str(status),
            "protocol_fingerprint": str(protocol_fingerprint),
            "details": details or {},
        }
        if status == "PASS":
            self.save_unit(unit, payload)
        else:
            error = RuntimeError(str((details or {}).get("error", status)))
            self.fail_unit(unit, error)

    def complete(
        self,
        *,
        output_locations: list[str | Path] | None = None,
        result_reference: dict[str, Any] | None = None,
        result_disposition: str = "newly_created",
    ) -> None:
        """Complete an adapter run with a verified manifest checkpoint.

        Adapter runs emit ordinary files rather than a serialized AnalysisResult.  The
        shared checkpoint store nevertheless requires every completed run to point to
        a durable, verified result checkpoint.  Persist the output-file identities in
        the attempt checkpoint directory instead of weakening that shared invariant.
        """

        if result_reference is None:
            components: dict[str, dict[str, Any]] = {}
            output_root = self.root.parent
            for value in output_locations or []:
                path = Path(value).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Adapter completion output does not exist: {path}"
                    )
                try:
                    relative = path.relative_to(output_root).as_posix()
                except ValueError:
                    relative = path.name
                components[relative] = {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            checkpoint = self.result_checkpoint_path()
            checkpoint.mkdir(parents=True, exist_ok=False)
            payload = {
                "checkpoint_schema_version": "adapter-output-manifest-v1",
                "components": components,
            }
            manifest = checkpoint / "manifest.json"
            temporary = checkpoint / "manifest.json.partial"
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, manifest)
            result_reference = {
                **payload,
                "path": checkpoint.relative_to(self.root).as_posix(),
                "metadata_sha256": _sha256(manifest),
            }
        super().complete(
            output_locations=output_locations,
            result_reference=result_reference,
            result_disposition=result_disposition,
        )
