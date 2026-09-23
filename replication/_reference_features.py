"""Strict, read-only adapters for authoritative frozen research embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _common import sha256

ALL_PAPER_ENCODERS = [
    "resnet50",
    "vgg16",
    "inception_v3",
    "mobilenet_v2",
    "coco_deeplab",
    "ade20k_segformer",
    "dinov2_vitb14",
    "siglip2_b16",
    "convnext_b",
    "vit_b16",
]


def known_reference_specifications(
    layout: str,
    research_root: str | Path,
    *,
    dataset: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve documented research layouts without embedding private paths in notebooks."""

    root = Path(research_root).expanduser().resolve()
    if layout == "representation_risk":
        if not dataset:
            raise ValueError("representation_risk reference layout requires dataset")
        base = root / "representation_risk_analysis"
        legacy = base / "datasets" / dataset / "embeddings_legacy"
        modern = base / "embeddings_modern" / dataset
        source_manifest = base / "datasets" / dataset / "manifest.csv"
        locations = {
            "resnet50": legacy,
            "vgg16": legacy,
            "inception_v3": legacy,
            "mobilenet_v2": legacy,
            "coco_deeplab": legacy,
            "ade20k_segformer": legacy,
            "dinov2_vitb14": modern,
            "siglip2_b16": modern,
            "convnext_b": modern,
            "vit_b16": modern,
        }
        return {
            name: {
                "array": str(locations[name] / f"{name}.npy"),
                "metadata": str(locations[name] / f"{name}.json"),
                "source_manifest": str(source_manifest),
                "source_id_column": "observation_id",
                "source_hash_column": "image_sha256",
                "source_encoder_aliases": [name],
            }
            for name in ALL_PAPER_ENCODERS
        }
    if layout == "obs_horses":
        base = root / "horse/obs_horse_dataset/analysis"
        aliases = {
            "resnet50": "resnet50",
            "vgg16": "vgg16_gap",
            "inception_v3": "inception_v3_gap",
            "mobilenet_v2": "mobilenet_v2_gap",
            "coco_deeplab": "coco_semantic",
            "ade20k_segformer": "ade20k_scene",
            "dinov2_vitb14": "dinov2_vitb14",
            "siglip2_b16": "siglip2_b16",
            "convnext_b": "convnext_b",
            "vit_b16": "vit_b16",
        }
        return {
            name: {
                "array": str(base / "embeddings" / source_name / "embeddings.npy"),
                "metadata": str(base / "embeddings" / source_name / "manifest.json"),
                "source_manifest": str(base / "results/master_photo_index.csv"),
                "source_id_column": "horse_sale_id",
                "source_hash_column": "photo_sha256",
                "source_encoder_aliases": [source_name],
            }
            for name, source_name in aliases.items()
        }

    raise ValueError(f"Unknown reference embedding layout {layout!r}")


def _digest_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _feature_digest(matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    if matrix.flags.c_contiguous:
        digest.update(memoryview(matrix).cast("B"))
    else:
        for row in matrix:
            digest.update(memoryview(np.ascontiguousarray(row)).cast("B"))
    return digest.hexdigest()


def _metadata_value(metadata: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def load_reference_features(
    frame: pd.DataFrame,
    specifications: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any]]:
    """Load only embeddings whose source metadata proves exact row/image alignment."""

    from lookagain_ml.encoders import encoder_metadata

    required = {"observation_id", "image_sha256", "reference_row"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reference-feature manifest is missing columns: {missing}")
    rows = frame["reference_row"].to_numpy(dtype=int)
    current_ids = frame["observation_id"].astype(str).tolist()
    current_hashes = frame["image_sha256"].astype(str).str.lower().tolist()
    current_id_digest = _digest_strings(current_ids)
    current_image_digest = _digest_strings(current_hashes)
    features: dict[str, np.ndarray] = {}
    audits: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {}

    for name, spec in specifications.items():
        array_path = Path(spec["array"]).expanduser().resolve()
        metadata_path = Path(spec["metadata"]).expanduser().resolve()
        source_manifest_path = Path(spec["source_manifest"]).expanduser().resolve()
        if not array_path.is_file() or not metadata_path.is_file() or not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"{name}: reference array, metadata, or source manifest is unreadable"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = pd.read_csv(source_manifest_path, low_memory=False)
        id_column = str(spec["source_id_column"])
        hash_column = str(spec["source_hash_column"])
        if id_column not in source or hash_column not in source:
            raise ValueError(f"{name}: source manifest lacks {id_column!r}/{hash_column!r}")
        source_ids = source[id_column].astype(str).tolist()
        source_hashes = source[hash_column].astype(str).str.lower().tolist()
        source_id_digest = _digest_strings(source_ids)
        source_image_digest = _digest_strings(source_hashes)
        expected_rows = int(_metadata_value(metadata, "rows", "row_count"))
        if expected_rows != len(source):
            raise ValueError(f"{name}: metadata/source row-count mismatch")
        if metadata.get("row_id_sha256") != source_id_digest:
            raise ValueError(f"{name}: source row-ID digest mismatch")
        stored_image_digest = _metadata_value(
            metadata, "image_hash_sha256", "photo_hash_sha256", "image_sha256_digest"
        )
        if stored_image_digest != source_image_digest:
            raise ValueError(f"{name}: source image-hash digest mismatch")
        if rows.min(initial=0) < 0 or rows.max(initial=-1) >= len(source):
            raise ValueError(f"{name}: reference_row is outside the source embedding array")
        if [source_ids[position] for position in rows] != current_ids:
            raise ValueError(f"{name}: current observation IDs do not match source rows")
        if [source_hashes[position] for position in rows] != current_hashes:
            raise ValueError(f"{name}: current image hashes do not match source rows")

        raw = np.load(array_path, mmap_mode="r", allow_pickle=False)
        declared = encoder_metadata(name).to_dict()
        dimension = int(_metadata_value(metadata, "embedding_dimension", "feature_dimension"))
        if raw.shape != (len(source), dimension) or dimension != declared["feature_dimension"]:
            raise ValueError(f"{name}: array/metadata/registry dimension mismatch")
        if raw.dtype != np.float32:
            raise ValueError(f"{name}: expected float32 reference features, found {raw.dtype}")
        aliases = {str(value) for value in spec.get("source_encoder_aliases", [name])}
        if str(metadata.get("encoder")) not in aliases:
            raise ValueError(f"{name}: source encoder label is not an allowed exact alias")
        if metadata.get("frozen") is not True or metadata.get("embedding_semantics_verified") is not True:
            raise ValueError(f"{name}: source metadata does not verify frozen feature semantics")
        selected = raw if np.array_equal(rows, np.arange(len(source))) else np.asarray(raw[rows], dtype=np.float32)
        if not np.isfinite(selected).all():
            raise ValueError(f"{name}: reference features contain non-finite values")
        array_sha256 = sha256(array_path)
        metadata_sha256 = sha256(metadata_path)
        creation_costs = {
            "device": "historical validated extraction; see source metadata",
            "extraction_seconds": _metadata_value(
                metadata, "extraction_seconds", "extraction_seconds_new_images"
            ),
            "images_per_second": _metadata_value(
                metadata, "images_per_second", "images_per_second_new_images"
            ),
            "parameter_count": _metadata_value(
                metadata, "parameter_count", "vision_parameter_count"
            ),
            "peak_gpu_memory_bytes": metadata.get("peak_gpu_memory_bytes"),
        }
        features[name] = selected
        audits[name] = {
            "ordered_observation_id_sha256": current_id_digest,
            "ordered_image_hash_sha256": current_image_digest,
            "feature_values_sha256": _feature_digest(selected),
            "encoder_metadata": declared,
            "source_array_sha256": array_sha256,
            "source_metadata_sha256": metadata_sha256,
            "creation_costs": creation_costs,
        }
        records[name] = {
            "status": "PASS",
            "source_rows": len(source),
            "analysis_rows": len(frame),
            "dimension": dimension,
            "dtype": "float32",
            "source_row_digest": source_id_digest,
            "source_image_digest": source_image_digest,
            "analysis_row_digest": current_id_digest,
            "analysis_image_digest": current_image_digest,
            "source_array_sha256": array_sha256,
            "source_metadata_sha256": metadata_sha256,
            "interpretation": (
                "eligible for provided-verified downstream reuse; this is not fresh "
                "LookAgain-ML checkpoint inference"
            ),
        }
    return features, audits, records
