from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

# Tests shipped in the sdist use privacy-safe replication protocol helpers from
# its extracted root. Adding only that root does not expose ``src/lookagain_ml``;
# the package under test must still resolve from the installed wheel when the
# hosted control gate requests it.
TEST_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_SOURCE_ROOT))

import numpy as np
import pytest
from PIL import Image

import lookagain_ml
from lookagain_ml.encoders import EncoderMetadata, ImageEncoder

if os.environ.get("LOOKAGAIN_EXPECT_INSTALLED_WHEEL") == "1":
    origin = Path(lookagain_ml.__file__).resolve()
    if not ({"site-packages", "dist-packages"} & set(origin.parts)):
        raise RuntimeError(
            "Installed-wheel control resolved lookagain_ml outside site-packages: "
            f"{origin}"
        )
    if importlib.metadata.version("lookagain-ml") != "1.0.0":
        raise RuntimeError("Installed-wheel control requires lookagain-ml 1.0.0.")

TEST_METADATA = EncoderMetadata(
    name="resnet50",
    display_name="Deterministic test image statistics",
    checkpoint="generated-test-adapter",
    feature_dimension=6,
    input_size=(16, 16),
    preprocessing="RGB conversion and channel moments",
    pooling="channel means and standard deviations",
    backend="Pillow/NumPy test fixture",
)


class TestImageStatisticsEncoder(ImageEncoder):
    metadata = TEST_METADATA

    def __init__(self, *, pretrained: bool = True, device: str = "auto") -> None:
        del pretrained, device
        self.costs = {"device": "cpu", "feature_dimension": 6}

    def encode(self, paths, *, batch_size: int = 32):
        del batch_size
        rows = []
        for path in paths:
            with Image.open(path) as image:
                values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(
                np.concatenate(
                    [values.mean(axis=(0, 1)), values.std(axis=(0, 1))]
                )
            )
        return np.asarray(rows, dtype=np.float32)


@pytest.fixture
def simple_api_encoder():
    from lookagain_ml.encoders import get_registration, register_encoder

    original = get_registration("resnet50")
    register_encoder(
        "resnet50",
        lambda **kwargs: TestImageStatisticsEncoder(**kwargs),
        TEST_METADATA,
        replace=True,
    )
    try:
        yield
    finally:
        register_encoder(
            original.name, original.factory, original.metadata, replace=True
        )


@pytest.fixture
def image_factory(tmp_path: Path):
    def make(name: str, rgb: tuple[int, int, int], *, mode: str = "RGB") -> Path:
        path = tmp_path / name
        if mode == "L":
            image = Image.new("L", (20, 20), color=rgb[0])
        else:
            image = Image.new("RGB", (20, 20), color=rgb)
        image.save(path)
        return path

    return make


