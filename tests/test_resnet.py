from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from lookagain_ml.encoders.resnet import resolve_device
from lookagain_ml.exceptions import EncoderUnavailableError, InputValidationError


class _FakeCuda:
    def __init__(self, *, available: bool, count: int = 0):
        self._available = available
        self._count = count

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count


def _fake_torch(*, cuda: bool = False, cuda_count: int = 0, mps: bool = False):
    return SimpleNamespace(
        cuda=_FakeCuda(available=cuda, count=cuda_count),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def _real_cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def test_device_auto_prefers_cuda_then_mps_then_cpu():
    assert resolve_device("auto", _fake_torch(cuda=True, cuda_count=1, mps=True)) == "cuda"
    assert resolve_device("auto", _fake_torch(mps=True)) == "mps"
    assert resolve_device("auto", _fake_torch()) == "cpu"


def test_indexed_cuda_request_is_validated_before_model_use():
    torch_module = _fake_torch(cuda=True, cuda_count=2)
    assert resolve_device("CUDA:1", torch_module) == "cuda:1"
    with pytest.raises(EncoderUnavailableError, match="reports 2 available"):
        resolve_device("cuda:2", torch_module)
    with pytest.raises(InputValidationError, match="cuda:0"):
        resolve_device("cuda:any", torch_module)


def test_explicit_unavailable_accelerators_fail_with_remedy():
    with pytest.raises(EncoderUnavailableError, match="Use device='cpu' or 'auto'"):
        resolve_device("cuda", _fake_torch())
    with pytest.raises(EncoderUnavailableError, match="Use device='cpu' or 'auto'"):
        resolve_device("mps", _fake_torch())


@pytest.mark.torch
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None,
    reason="torch/torchvision are not installed",
)
def test_resnet50_untrained_cpu_smoke_preserves_row_order(image_factory):
    from lookagain_ml.encoders.resnet import ResNet50Encoder

    first = image_factory("resnet-red.png", (200, 10, 10))
    second = image_factory("resnet-blue.png", (10, 10, 200))
    grayscale = image_factory("resnet-gray.png", (90, 0, 0), mode="L")
    encoder = ResNet50Encoder(pretrained=False, device="cpu")
    values = encoder.encode([first, second, grayscale], batch_size=1)
    assert values.shape == (3, 2048)
    assert encoder.metadata.checkpoint.startswith("untrained")
    assert encoder.costs["feature_dimension"] == 2048
    assert encoder.costs["extraction_seconds"] > 0
    assert encoder.costs["images_per_second"] > 0
    assert encoder.costs["peak_gpu_memory_bytes"] is None


@pytest.mark.torch
@pytest.mark.skipif(not _real_cuda_available(), reason="a CUDA runtime is not available")
def test_resnet50_untrained_cuda_smoke(image_factory):
    from lookagain_ml.encoders.resnet import ResNet50Encoder

    image = image_factory("resnet-cuda.png", (120, 40, 200))
    encoder = ResNet50Encoder(pretrained=False, device="cuda")
    values = encoder.encode([image], batch_size=1)
    assert values.shape == (1, 2048)
    assert encoder.costs["device"].startswith("cuda")
    assert encoder.costs["peak_gpu_memory_bytes"] > 0

