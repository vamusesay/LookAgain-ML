from __future__ import annotations

import hashlib

import numpy as np
import pytest
from conftest import TEST_METADATA

from lookagain_ml import InputValidationError, analyze
from lookagain_ml.hashing import digest_strings, sha256_file


def _audit(images, observation_ids, features):
    return {
        "ordered_observation_id_sha256": digest_strings(observation_ids),
        "ordered_image_hash_sha256": digest_strings([sha256_file(path) for path in images]),
        "feature_values_sha256": hashlib.sha256(memoryview(features).cast("B")).hexdigest(),
        "encoder_metadata": TEST_METADATA.to_dict(),
        "source_array_sha256": "a" * 64,
        "source_metadata_sha256": "b" * 64,
        "creation_costs": {"device": "reference-cuda", "extraction_seconds": 12.5},
    }


def test_provided_features_require_and_preserve_row_identity(
    image_factory, simple_api_encoder, tmp_path
):
    images = [
        image_factory(f"provided-{index}.png", (10 + index * 7, 30, 50))
        for index in range(18)
    ]
    observation_ids = [f"stable-id-{index}" for index in range(18)]
    features = np.random.default_rng(7).normal(size=(18, 6)).astype(np.float32)
    splits = ["train"] * 12 + ["validation"] * 3 + ["test"] * 3
    with pytest.warns(UserWarning, match="provided-verified"):
        result = analyze(
            images,
            np.linspace(0.0, 1.0, 18),
            groups=[f"group-{index}" for index in range(18)],
            observation_ids=observation_ids,
            split_labels=splits,
            precomputed_features={"resnet50": features},
            precomputed_feature_audits={
                "resnet50": _audit(images, observation_ids, features)
            },
            cache=True,
            cache_dir=tmp_path / "unused-cache",
        )

    assert result.manifest["observation_id"].tolist() == observation_ids
    assert result.cache_metadata["resnet50"]["status"] == "provided-verified"
    assert result.costs["device"] == "provided-precomputed-no-inference"
    assert result.configuration["analysis"]["cache_enabled"] is False
    assert result.configuration["analysis"]["representation_source"] == (
        "provided-verified frozen features"
    )


def test_provided_features_reject_row_or_value_changes(
    image_factory, simple_api_encoder
):
    images = [
        image_factory(f"provided-reject-{index}.png", (20 + index * 6, 40, 60))
        for index in range(18)
    ]
    observation_ids = [f"stable-{index}" for index in range(18)]
    features = np.random.default_rng(11).normal(size=(18, 6)).astype(np.float32)
    audit = _audit(images, observation_ids, features)
    common = {
        "images": images,
        "y": np.linspace(0.0, 1.0, 18),
        "groups": [f"g-{index}" for index in range(18)],
        "observation_ids": observation_ids,
        "split_labels": ["train"] * 12 + ["validation"] * 3 + ["test"] * 3,
        "precomputed_features": {"resnet50": features},
        "cache": False,
    }

    wrong_rows = {**audit, "ordered_observation_id_sha256": "0" * 64}
    with pytest.raises(InputValidationError, match="ordered observation IDs"):
        analyze(precomputed_feature_audits={"resnet50": wrong_rows}, **common)

    changed = features.copy()
    changed[0, 0] += 1
    with pytest.raises(InputValidationError, match="feature-value digest"):
        analyze(
            precomputed_features={"resnet50": changed},
            precomputed_feature_audits={"resnet50": audit},
            **{key: value for key, value in common.items() if key != "precomputed_features"},
        )


def test_observation_ids_must_be_unique(image_factory, simple_api_encoder):
    images = [image_factory(f"duplicate-id-{index}.png", (index + 1, 2, 3)) for index in range(18)]
    with pytest.raises(InputValidationError, match="observation_ids must be unique"):
        analyze(
            images,
            np.arange(18, dtype=float),
            observation_ids=["same"] * 18,
            groups=[f"g-{index}" for index in range(18)],
        )
