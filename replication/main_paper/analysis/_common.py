"""Shared, non-scientific plumbing for the replication entry points."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a JSON object.")
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            missing = sorted({name for name in pattern.findall(value) if name not in os.environ})
            if missing:
                raise ValueError(
                    f"Configuration {path} requires environment variables: {missing}"
                )
            return pattern.sub(lambda match: os.environ[match.group(1)], value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return expand(config)


def installed_package_evidence(*, require_installed: bool = True) -> dict[str, Any]:
    import lookagain_ml

    module_path = Path(lookagain_ml.__file__).resolve()
    installed = any(part.lower() in {"site-packages", "dist-packages"} for part in module_path.parts)
    evidence = {
        "distribution": "lookagain-ml",
        "version": lookagain_ml.__version__,
        "module_path": str(module_path),
        "installed_location_verified": installed,
        "distribution_version": importlib.metadata.version("lookagain-ml"),
    }
    if require_installed and not installed:
        raise RuntimeError(
            "lookagain_ml imported outside site-packages/dist-packages. Install the built wheel "
            "in a clean environment and run from outside the source tree."
        )
    return evidence


def environment_record(output_root: Path, *, device: str, batch_size: int) -> dict[str, Any]:
    import torch

    disk = shutil.disk_usage(output_root.resolve().anchor)
    record: dict[str, Any] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "requested_device": device,
        "batch_size": batch_size,
        "free_disk_bytes": disk.free,
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "environment": {
            key: os.environ.get(key)
            for key in ("COLAB_RELEASE_TAG", "COLAB_GPU", "CUDA_VISIBLE_DEVICES")
            if os.environ.get(key) is not None
        },
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        record["physical_memory_total_bytes"] = int(memory.total)
        record["physical_memory_available_bytes"] = int(memory.available)
    except ImportError:
        record["physical_memory_total_bytes"] = None
        record["physical_memory_available_bytes"] = None
        record["memory_note"] = "psutil is not installed; RAM availability was not measured."
    if device.lower().startswith("cuda") and not record["cuda_available"]:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return record


def load_manifest(path: Path, *, data_root: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"observation_id", "image_path", "outcome", "group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {missing}")
    if frame["observation_id"].duplicated().any():
        raise ValueError(f"Manifest {path} has duplicate observation_id values.")
    root = data_root if data_root is not None else path.parent
    frame["image_path"] = frame["image_path"].map(
        lambda value: str((root / str(value)).resolve()) if not Path(str(value)).is_absolute() else str(Path(str(value)).resolve())
    )
    missing_images = [value for value in frame["image_path"] if not Path(value).is_file()]
    if missing_images:
        preview = missing_images[:3]
        raise FileNotFoundError(f"{len(missing_images)} image files are missing; first paths: {preview}")
    return frame


def validate_split_column(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        raise ValueError(f"Split column {column!r} is absent.")
    labels = frame[column].astype(str).str.lower().tolist()
    observed = set(labels)
    if observed != {"train", "validation", "test"}:
        raise ValueError(f"{column!r} must contain exactly train/validation/test; found {sorted(observed)}")
    for partition in ("train", "validation", "test"):
        if labels.count(partition) < 2:
            raise ValueError(f"{column!r} has fewer than two {partition} rows.")
    group_partitions = pd.DataFrame({"group": frame["group_id"].astype(str), "split": labels})
    if (group_partitions.groupby("group")["split"].nunique() > 1).any():
        raise ValueError(f"{column!r} assigns at least one group to multiple partitions.")
    return labels




def final_status(records: list[dict[str, Any]]) -> str:
    statuses = {str(record.get("status")) for record in records}
    if statuses <= {"PASS"}:
        return "PASS"
    if "FAIL" in statuses:
        return "FAIL"
    if "PASS" in statuses:
        return "PARTIAL"
    return "NOT RUN"
