from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from lookagain_ml.encoders import encoder_metadata, list_encoders
from lookagain_ml.encoders.huggingface_models import (
    SIGLIP2_REVISION,
    HuggingFaceFrozenEncoder,
)
from lookagain_ml.encoders.torchvision_models import TorchvisionFrozenEncoder


def test_required_stage2_registry_and_metadata_are_exact():
    required = {"resnet50", "dinov2_vitb14", "siglip2_b16", "convnext_b", "vit_b16"}
    assert required.issubset(list_encoders())
    assert "mobilenet_v2" in list_encoders()
    expected = {
        "dinov2_vitb14": (768, "facebook/dinov2-base", "class token"),
        "siglip2_b16": (768, "google/siglip2-base-patch16-224", "pooled vision"),
        "convnext_b": (1024, "IMAGENET1K_V1", "global-average"),
        "vit_b16": (768, "IMAGENET1K_V1", "class token"),
        "mobilenet_v2": (1280, "IMAGENET1K_V1", "global-average"),
    }
    for name, (dimension, checkpoint_text, pooling_text) in expected.items():
        metadata = encoder_metadata(name)
        assert metadata.feature_dimension == dimension
        assert checkpoint_text in metadata.checkpoint
        assert pooling_text in metadata.pooling
        assert metadata.preprocessing
        assert metadata.checkpoint_revision
        assert metadata.source_url


@pytest.mark.torch
@pytest.mark.parametrize(
    "name,dimension",
    [("convnext_b", 1024), ("vit_b16", 768), ("mobilenet_v2", 1280)],
)
def test_torchvision_stage2_encoder_cpu_untrained_smoke(name, dimension, image_factory, tmp_path):
    image = image_factory(f"{name}.png", (40, 80, 120))
    encoder = TorchvisionFrozenEncoder(
        name, pretrained=False, device="cpu", model_cache_dir=tmp_path / "models"
    )
    values = encoder.encode([image], batch_size=1)
    assert values.shape == (1, dimension)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert encoder.metadata.checkpoint.startswith("untrained weights")


@pytest.mark.torch
def test_siglip2_uses_auto_dispatch_and_pooled_vision_output(monkeypatch, image_factory, tmp_path):
    import torch

    calls = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls.append(("processor", checkpoint, kwargs["revision"]))
            return cls()

        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((len(images), 3, 224, 224))}

    class FakeModel:
        config = types.SimpleNamespace(_commit_hash=SIGLIP2_REVISION)

        def __init__(self):
            self._vision_parameter = torch.nn.Parameter(torch.zeros(3))
            self._text_parameter = torch.nn.Parameter(torch.zeros(5))
            self.vision_model = types.SimpleNamespace(parameters=lambda: [self._vision_parameter])

        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls.append(("model", checkpoint, kwargs["revision"]))
            return cls()

        def eval(self):
            return self

        def to(self, device):
            return self

        def parameters(self):
            return [self._vision_parameter, self._text_parameter]

        def get_image_features(self, **inputs):
            batch = inputs["pixel_values"].shape[0]
            return types.SimpleNamespace(pooler_output=torch.full((batch, 768), 2.0))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeProcessor
    fake_transformers.AutoModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    image = image_factory("siglip2-test.png", (20, 40, 60))
    encoder = HuggingFaceFrozenEncoder(
        "siglip2_b16", device="cpu", model_cache_dir=tmp_path / "models"
    )
    values = encoder.encode([image], batch_size=1)
    assert calls == [
        ("processor", "google/siglip2-base-patch16-224", SIGLIP2_REVISION),
        ("model", "google/siglip2-base-patch16-224", SIGLIP2_REVISION),
    ]
    assert values.shape == (1, 768)
    np.testing.assert_array_equal(values, np.full((1, 768), 2.0, dtype=np.float32))
    assert encoder.costs["parameter_count"] == 3
    assert encoder.costs["loaded_parameter_count"] == 8


@pytest.mark.real_encoder
@pytest.mark.skipif(
    os.environ.get("LOOKAGAIN_RUN_REAL_SIGLIP2") != "1",
    reason="set LOOKAGAIN_RUN_REAL_SIGLIP2=1 on a resource-appropriate host",
)
def test_siglip2_real_published_checkpoint_smoke(image_factory, tmp_path):
    model_cache = Path(
        os.environ.get("LOOKAGAIN_MODEL_CACHE_DIR", tmp_path / "models")
    )
    first = image_factory("siglip2-real-red.png", (200, 30, 40))
    second = image_factory("siglip2-real-blue.png", (30, 40, 200))
    device = os.environ.get("LOOKAGAIN_RUNTIME_DEVICE", "auto")
    encoder = HuggingFaceFrozenEncoder(
        "siglip2_b16", device=device, model_cache_dir=model_cache
    )
    values = encoder.encode([first, second], batch_size=1)
    assert values.shape == (2, 768)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert encoder.metadata.checkpoint_revision == SIGLIP2_REVISION
    assert encoder.costs["extraction_seconds"] > 0
    assert encoder.costs["images_per_second"] > 0
