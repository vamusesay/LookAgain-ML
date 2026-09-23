from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lookagain_ml import DISTRIBUTION_NAME, IMPORT_NAME
from lookagain_ml.cache import (
    CACHE_SCHEMA_VERSION,
    EmbeddingCache,
    build_cache_identity,
)
from lookagain_ml.encoders import EncoderMetadata
from lookagain_ml.exceptions import CacheValidationError

METADATA = EncoderMetadata(
    name="cache_test",
    display_name="Cache test",
    checkpoint="test-checkpoint",
    checkpoint_revision="immutable-revision",
    feature_dimension=3,
    input_size=(8, 8),
    preprocessing="deterministic test transform",
    pooling="test pooling",
    backend="test",
)


def identity(
    *,
    observation_digest: str = "a" * 64,
    image_digest: str = "b" * 64,
    torch_version: str = "2.test",
    metadata: EncoderMetadata = METADATA,
):
    return build_cache_identity(
        metadata,
        row_count=4,
        ordered_observation_id_sha256=observation_digest,
        ordered_image_hash_sha256=image_digest,
        runtime_dependencies={"torch": torch_version},
    )


def test_cache_roundtrip_verifies_complete_identity_and_checksum(tmp_path):
    cache_identity = identity()
    assert cache_identity["cache_schema_version"] == CACHE_SCHEMA_VERSION == "4"
    assert cache_identity["distribution_name"] == DISTRIBUTION_NAME == "lookagain-ml"
    assert cache_identity["import_name"] == IMPORT_NAME == "lookagain_ml"
    cache = EmbeddingCache(tmp_path / "cache")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    created = cache.save(cache_identity, values, creation_costs={"extraction_seconds": 1.25})
    assert created.metadata["status"] == "created"
    reused = cache.load(cache_identity)
    assert reused is not None
    assert reused.metadata["status"] == "reused"
    assert reused.metadata["creation_costs"]["extraction_seconds"] == 1.25
    np.testing.assert_array_equal(reused.features, values)
    assert not list((tmp_path / "cache").rglob("*.partial.*"))


def test_cache_recovers_once_if_new_atomic_directory_transiently_disappears(tmp_path, monkeypatch):
    cache = EmbeddingCache(tmp_path / "cache")
    original_open = Path.open
    removed_once = False

    def transient_open(path, *args, **kwargs):
        nonlocal removed_once
        if not removed_once and path.name.endswith(".partial.npy"):
            removed_once = True
            path.parent.rmdir()
            raise FileNotFoundError(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", transient_open)
    saved = cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    assert removed_once
    assert saved.metadata["status"] == "created"
    assert cache.load(identity()) is not None


def test_cache_never_reuses_a_different_row_order(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    assert cache.load(identity(observation_digest="c" * 64)) is None


def test_cache_never_reuses_different_ordered_image_content(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    assert cache.load(identity(image_digest="d" * 64)) is None


def test_cache_never_reuses_different_preprocessing_metadata(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    changed = EncoderMetadata(
        **{**METADATA.to_dict(), "preprocessing": "different deterministic transform"}
    )
    assert cache.load(identity(metadata=changed)) is None


def test_cache_never_reuses_a_different_numerical_runtime(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    assert cache.load(identity(torch_version="2.changed")) is None


def test_cache_rejects_tampered_identity(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    saved = cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    metadata_path = next((tmp_path / "cache").rglob(f"{saved.metadata['cache_key']}.json"))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["identity"]["feature_dimension"] = 999
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CacheValidationError, match="stored identity does not match"):
        cache.load(identity())


def test_cache_rejects_corrupt_array(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    saved = cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    array_path = next((tmp_path / "cache").rglob(f"{saved.metadata['cache_key']}.npy"))
    with array_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CacheValidationError, match="checksum failed"):
        cache.load(identity())


def test_cache_rejects_incomplete_array_metadata_pair(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    saved = cache.save(identity(), np.ones((4, 3), dtype=np.float32))
    metadata_path = next((tmp_path / "cache").rglob(f"{saved.metadata['cache_key']}.json"))
    metadata_path.unlink()
    with pytest.raises(CacheValidationError, match="array/metadata pair is incomplete"):
        cache.load(identity())


def test_cache_refuses_nonfinite_or_wrong_shape(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    with pytest.raises(CacheValidationError, match="Refused to cache"):
        cache.save(identity(), np.full((4, 3), np.nan, dtype=np.float32))
    with pytest.raises(CacheValidationError, match="Refused to cache"):
        cache.save(identity(), np.ones((3, 3), dtype=np.float32))
