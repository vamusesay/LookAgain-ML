"""Torchvision ResNet50 frozen-feature adapter."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..exceptions import EncoderUnavailableError, ImageLoadError, InputValidationError
from .base import EncoderMetadata, ImageEncoder

RESNET50_METADATA = EncoderMetadata(
    name="resnet50",
    display_name="ResNet50",
    checkpoint="torchvision::ResNet50_Weights.IMAGENET1K_V1",
    checkpoint_revision="IMAGENET1K_V1",
    feature_dimension=2048,
    input_size=(224, 224),
    preprocessing=(
        "official weight transform: resize shorter side to 256 with bilinear interpolation, "
        "center crop 224, rescale to [0,1], ImageNet mean/std normalization"
    ),
    pooling="2048-dimensional global-average-pooled penultimate feature",
    backend="torchvision",
    source_url="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html",
    license="BSD-3-Clause (torchvision code); checkpoint terms follow upstream",
)


def resolve_device(requested: str, torch_module: Any) -> str:
    """Resolve auto/CUDA/MPS/CPU and reject unavailable requested devices."""

    if not isinstance(requested, str) or not requested.strip():
        raise InputValidationError(
            "device must be 'auto', 'cpu', 'mps', 'cuda', or a CUDA device such as 'cuda:0'."
        )
    value = requested.strip().lower()
    if value not in {"auto", "cpu", "mps", "cuda"} and re.fullmatch(
        r"cuda:\d+", value
    ) is None:
        raise InputValidationError(
            "device must be 'auto', 'cpu', 'mps', 'cuda', or a CUDA device such as 'cuda:0'."
        )
    if value == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(torch_module.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if value.startswith("cuda") and not torch_module.cuda.is_available():
        raise EncoderUnavailableError(
            "CUDA was requested but PyTorch cannot access a CUDA device. Use device='cpu' or 'auto'."
        )
    if value.startswith("cuda:"):
        index = int(value.split(":", 1)[1])
        device_count = int(torch_module.cuda.device_count())
        if index >= device_count:
            raise EncoderUnavailableError(
                f"CUDA device {value!r} was requested, but PyTorch reports {device_count} "
                "available CUDA device(s). Choose an available index, 'cuda', 'cpu', or 'auto'."
            )
    if value == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise EncoderUnavailableError(
                "MPS was requested but is unavailable. Use device='cpu' or 'auto'."
            )
    return value


class ResNet50Encoder(ImageEncoder):
    """Frozen ImageNet ResNet50 with classifier removed."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        device: str = "auto",
        model_cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
            from torchvision import transforms
            from torchvision.models import ResNet50_Weights, resnet50
            from torchvision.transforms import InterpolationMode
        except ImportError as error:
            raise EncoderUnavailableError(
                "ResNet50 requires torch and torchvision. Install LookAgain-ML's declared dependencies "
                "and retry."
            ) from error

        self._torch = torch
        self._device = resolve_device(device, torch)
        if model_cache_dir is not None:
            torch.hub.set_dir(str(Path(model_cache_dir).expanduser().resolve() / "torch"))
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = resnet50(weights=weights)
        except Exception as error:
            if pretrained:
                raise EncoderUnavailableError(
                    "The ResNet50 IMAGENET1K_V1 checkpoint could not be loaded. The first run may "
                    "need internet access to download torchvision weights. Original error: "
                    f"{error}"
                ) from error
            raise
        model.fc = nn.Identity()
        model.eval().to(self._device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._transform = (
            weights.transforms()
            if weights is not None
            else transforms.Compose(
                [
                    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )
        self.metadata = (
            RESNET50_METADATA
            if pretrained
            else EncoderMetadata(
                **{
                    **RESNET50_METADATA.to_dict(),
                    "checkpoint": "untrained weights (testing/infrastructure checks only)",
                    "checkpoint_revision": None,
                }
            )
        )
        self.costs: dict[str, Any] = {
            "device": self._device,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "parameter_count_scope": "frozen image representation encoder",
        }

    def _load(self, path: str | Path):
        try:
            with Image.open(path) as image:
                return self._transform(image.convert("RGB"))
        except Exception as error:
            raise ImageLoadError(
                f"Image {str(path)!r} passed manifest validation but failed during encoding: {error}"
            ) from error

    def encode(self, paths: Sequence[str | Path], *, batch_size: int = 32) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        torch = self._torch
        rows: list[np.ndarray] = []
        started = time.perf_counter()
        if self._device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self._device)
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                tensors = [self._load(path) for path in paths[start : start + batch_size]]
                batch = torch.stack(tensors).to(self._device)
                values = self._model(batch).detach().float().cpu().numpy()
                if values.ndim != 2 or values.shape[1] != self.metadata.feature_dimension:
                    raise RuntimeError(
                        f"ResNet50 returned {values.shape}; expected (*, {self.metadata.feature_dimension})."
                    )
                rows.append(values)
                self._report_batch_progress(min(start + batch_size, len(paths)), len(paths))
        elapsed = time.perf_counter() - started
        output = np.vstack(rows).astype(np.float32, copy=False)
        if output.shape[0] != len(paths) or not np.isfinite(output).all():
            raise RuntimeError("ResNet50 features are missing, misaligned, or non-finite.")
        self.costs.update(
            {
                "extraction_seconds": elapsed,
                "images_per_second": len(paths) / max(elapsed, 1e-12),
                "feature_dimension": self.metadata.feature_dimension,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self._device))
                    if self._device.startswith("cuda")
                    else None
                ),
            }
        )
        return output


def create_resnet50(
    *,
    pretrained: bool = True,
    device: str = "auto",
    model_cache_dir: str | Path | None = None,
) -> ResNet50Encoder:
    return ResNet50Encoder(
        pretrained=pretrained, device=device, model_cache_dir=model_cache_dir
    )
