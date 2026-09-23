"""Image manifest construction and input auditing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .exceptions import InputValidationError
from .hashing import sha256_file

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


@dataclass(frozen=True)
class ImageFailure:
    """One rejected observation and its human-readable reason."""

    source_position: int
    image: str
    reason: str


@dataclass(frozen=True)
class ImageAudit:
    """Counts and reasons from image-manifest validation."""

    requested_n: int
    successful_n: int
    failed_n: int
    failures: tuple[ImageFailure, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["failures"] = [asdict(item) for item in self.failures]
        return value


@dataclass(frozen=True)
class ManifestBuildResult:
    """Validated observations plus the complete audit."""

    manifest: pd.DataFrame
    audit: ImageAudit


def _as_list(values: Iterable[Any], name: str) -> list[Any]:
    if isinstance(values, (str, bytes, Path)):
        raise InputValidationError(
            f"{name} must contain one value per observation; a single string/path was provided."
        )
    try:
        return list(values)
    except TypeError as error:
        raise InputValidationError(f"{name} must be an iterable with one value per image.") from error


def _group_token(value: Any, position: int) -> str:
    if not pd.api.types.is_scalar(value):
        raise InputValidationError(
            f"groups row {position} is not a scalar identifier. "
            "Supply one string or numeric group ID per image."
        )
    if pd.isna(value):
        raise InputValidationError(
            f"groups contains a missing value at position {position}. "
            "Supply a valid group ID or omit groups entirely."
        )
    return f"{type(value).__name__}:{value}"


def build_image_manifest(
    images: Iterable[str | Path],
    y: Iterable[Any],
    groups: Iterable[Any] | None = None,
    *,
    observation_ids: Iterable[Any] | None = None,
    strict: bool = True,
    verify_images: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ManifestBuildResult:
    """Validate paths, outcomes, groups, image readability, and exact hashes.

    In strict mode, any failed observation raises with the complete audit. In
    non-strict mode, failed rows are omitted only after a warning can be issued
    by the caller; the audit always records every omission.
    """

    image_values = _as_list(images, "images")
    outcome_values = _as_list(y, "y")
    if len(image_values) != len(outcome_values):
        raise InputValidationError(
            f"images and y must have the same length; got {len(image_values)} and {len(outcome_values)}."
        )
    if not image_values:
        raise InputValidationError("At least one image and outcome are required.")

    if observation_ids is None:
        id_values = [f"obs-{position:08d}" for position in range(len(image_values))]
    else:
        raw_ids = _as_list(observation_ids, "observation_ids")
        if len(raw_ids) != len(image_values):
            raise InputValidationError(
                "observation_ids must match images; got "
                f"{len(raw_ids)} and {len(image_values)} rows."
            )
        id_values = []
        for position, value in enumerate(raw_ids):
            if pd.isna(value) or not str(value).strip():
                raise InputValidationError(
                    f"observation_ids contains a missing/blank value at position {position}."
                )
            id_values.append(str(value))
        if len(set(id_values)) != len(id_values):
            raise InputValidationError("observation_ids must be unique.")

    if groups is None:
        group_values: Sequence[Any] = [f"row:{i}" for i in range(len(image_values))]
        groups_supplied = False
    else:
        group_values = _as_list(groups, "groups")
        groups_supplied = True
        if len(group_values) != len(image_values):
            raise InputValidationError(
                f"groups must match images; got {len(group_values)} and {len(image_values)} rows."
            )

    rows: list[dict[str, Any]] = []
    failures: list[ImageFailure] = []
    for position, (raw_path, target, raw_group) in enumerate(
        zip(image_values, outcome_values, group_values)
    ):
        display_path = "" if raw_path is None else str(raw_path)
        reason: str | None = None
        resolved: Path | None = None
        digest: str | None = None
        image_format: str | None = None
        width: int | None = None
        height: int | None = None

        if not pd.api.types.is_scalar(target):
            raise InputValidationError(
                f"y row {position} is not a scalar outcome. Supply one continuous "
                "number or one class label per image; multilabel and multi-output "
                "targets are not supported."
            )
        if pd.isna(target):
            reason = "outcome is missing"
        elif raw_path is None or not display_path.strip():
            reason = "image path is missing"
        else:
            try:
                resolved = Path(display_path).expanduser().resolve(strict=False)
                if resolved.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
                    reason = f"unsupported image extension {resolved.suffix!r}; supported: {supported}"
                elif not resolved.exists():
                    reason = "file does not exist"
                elif not resolved.is_file():
                    reason = "path is not a file"
                else:
                    digest = sha256_file(resolved)
                    if verify_images:
                        with Image.open(resolved) as image:
                            image_format = image.format
                            width, height = image.size
                            image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as error:
                reason = f"image could not be read: {error}"

        user_group = _group_token(raw_group, position)

        if reason is not None:
            failures.append(ImageFailure(position, display_path, reason))
            if progress_callback is not None:
                progress_callback(position + 1, len(image_values))
            continue

        assert resolved is not None and digest is not None
        rows.append(
            {
                "observation_id": id_values[position],
                "source_position": position,
                "image_path": str(resolved),
                "image_name": resolved.name,
                "target": target,
                "user_group_id": user_group,
                "groups_supplied": groups_supplied,
                "image_sha256": digest,
                "image_format": image_format,
                "width": width,
                "height": height,
                "status": "ok",
            }
        )
        if progress_callback is not None:
            progress_callback(position + 1, len(image_values))

    audit = ImageAudit(
        requested_n=len(image_values),
        successful_n=len(rows),
        failed_n=len(failures),
        failures=tuple(failures),
    )
    if failures and strict:
        preview = "; ".join(
            f"row {item.source_position}: {item.reason}" for item in failures[:5]
        )
        extra = "" if len(failures) <= 5 else f"; plus {len(failures) - 5} more"
        raise InputValidationError(
            f"{len(failures)} of {len(image_values)} observations failed validation: {preview}{extra}. "
            "Fix the inputs or use strict=False and inspect results.audit before proceeding.",
            audit=audit.to_dict(),
        )
    if not rows:
        raise InputValidationError(
            "No usable observations remain after image validation.", audit=audit.to_dict()
        )

    manifest = pd.DataFrame(rows)
    counts = manifest["image_sha256"].value_counts()
    manifest["is_exact_duplicate"] = manifest["image_sha256"].map(counts).gt(1)
    return ManifestBuildResult(manifest=manifest, audit=audit)
