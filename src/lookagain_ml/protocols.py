"""Protocol provenance, fingerprints, and guarded scientific comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np

from .exceptions import InputValidationError


def canonical_configuration_json(value: Any) -> str:
    """Serialize configuration deterministically for provenance hashing."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def configuration_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical configuration serialization."""

    return hashlib.sha256(canonical_configuration_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProtocolSpec:
    """Versioned scientific comparison contract, independent of test results."""

    protocol_id: str
    protocol_version: str
    scientific_definitions: Mapping[str, Any]
    required_settings: Mapping[str, Any]
    reference_provenance: Mapping[str, Any]
    comparison_semantics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.protocol_id.strip() or not self.protocol_version.strip():
            raise InputValidationError("Protocol ID and version must be nonempty.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["configuration_sha256"] = configuration_sha256(payload)
        return payload

    @property
    def fingerprint(self) -> str:
        return configuration_sha256(asdict(self))


@dataclass(frozen=True)
class ProtocolCompatibility:
    status: str
    reasons: tuple[str, ...]
    actual_fingerprint: str
    reference_fingerprint: str

    @property
    def comparable(self) -> bool:
        return self.status == "COMPATIBLE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_protocol_compatibility(
    actual: ProtocolSpec, reference: ProtocolSpec
) -> ProtocolCompatibility:
    """Require exact agreement on definitions, settings, provenance, and semantics."""

    reasons: list[str] = []
    for field_name in (
        "protocol_id",
        "protocol_version",
        "scientific_definitions",
        "required_settings",
        "reference_provenance",
        "comparison_semantics",
    ):
        if canonical_configuration_json(getattr(actual, field_name)) != canonical_configuration_json(
            getattr(reference, field_name)
        ):
            reasons.append(f"{field_name} differs")
    return ProtocolCompatibility(
        status="COMPATIBLE" if not reasons else "NOT COMPARABLE",
        reasons=tuple(reasons),
        actual_fingerprint=actual.fingerprint,
        reference_fingerprint=reference.fingerprint,
    )


def compare_protocol_results(
    actual_values: Mapping[str, Real],
    reference_values: Mapping[str, Real],
    *,
    actual_protocol: ProtocolSpec,
    reference_protocol: ProtocolSpec,
    tolerance: float = 0.0001,
    package_execution_status: str = "PASS",
) -> dict[str, Any]:
    """Compare only compatible quantities and keep execution status separate."""

    if not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0:
        raise InputValidationError("tolerance must be a finite nonnegative number.")
    compatibility = assess_protocol_compatibility(actual_protocol, reference_protocol)
    base = {
        "package_execution_status": str(package_execution_status),
        "protocol_compatibility": compatibility.to_dict(),
        "precommitted_absolute_tolerance": float(tolerance),
    }
    if not compatibility.comparable:
        return {
            **base,
            "scientific_agreement_status": "NOT COMPARABLE",
            "comparisons": [],
            "reason": "; ".join(compatibility.reasons),
        }
    if set(actual_values) != set(reference_values):
        missing_actual = sorted(set(reference_values) - set(actual_values))
        missing_reference = sorted(set(actual_values) - set(reference_values))
        return {
            **base,
            "scientific_agreement_status": "NOT COMPARABLE",
            "comparisons": [],
            "reason": (
                f"metric keys differ; missing from actual={missing_actual}, "
                f"missing from reference={missing_reference}"
            ),
        }
    rows: list[dict[str, Any]] = []
    for name in sorted(actual_values):
        actual = float(actual_values[name])
        reference = float(reference_values[name])
        if not np.isfinite(actual) or not np.isfinite(reference):
            raise InputValidationError("Scientific comparison values must be finite.")
        difference = abs(actual - reference)
        rows.append(
            {
                "metric": name,
                "package_value": actual,
                "reference_value": reference,
                "absolute_difference": difference,
                "tolerance_precommitted": float(tolerance),
                "status": "PASS" if difference <= tolerance else "FAIL",
            }
        )
    return {
        **base,
        "scientific_agreement_status": (
            "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL"
        ),
        "comparisons": rows,
    }
