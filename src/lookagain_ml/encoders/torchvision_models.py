"""Frozen torchvision representations used by the comparison benchmark."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..exceptions import EncoderUnavailableError, ImageLoadError
from .base import EncoderMetadata, ImageEncoder
from .resnet import resolve_device

VGG16_METADATA = EncoderMetadata(
    name="vgg16",
    display_name="VGG16 (optional legacy baseline)",
    checkpoint="torchvision::VGG16_Weights.IMAGENET1K_V1",
    checkpoint_revision="IMAGENET1K_V1",
    feature_dimension=512,
    input_size=(224, 224),
    preprocessing=(
        "official weight transform: resize shorter side to 256 with bilinear interpolation, "
        "center crop 224, rescale to [0,1], ImageNet mean/std normalization"
    ),
    pooling="global average over the final convolutional feature map",
    backend="torchvision",
    source_url="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html",
    license="BSD-3-Clause (torchvision code); checkpoint terms follow upstream",
)

INCEPTION_V3_METADATA = EncoderMetadata(
    name="inception_v3",
    display_name="InceptionV3 (optional legacy baseline)",
    checkpoint="torchvision::Inception_V3_Weights.IMAGENET1K_V1",
    checkpoint_revision="IMAGENET1K_V1",
    feature_dimension=2048,
    input_size=(299, 299),
    preprocessing=(
        "official weight transform: resize shorter side to 342 with bilinear interpolation, "
        "center crop 299, rescale to [0,1], ImageNet mean/std normalization"
    ),
    pooling="2048-dimensional global-average-pooled penultimate feature; fc and auxiliary head removed",
    backend="torchvision",
    source_url="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html",
    license="BSD-3-Clause (torchvision code); checkpoint terms follow upstream",
)

CONVNEXT_B_METADATA = EncoderMetadata(
    name="convnext_b",
    display_name="ConvNeXt-B",
    checkpoint="torchvision::ConvNeXt_Base_Weights.IMAGENET1K_V1",
    checkpoint_revision="IMAGENET1K_V1",
    feature_dimension=1024,
    input_size=(224, 224),
    preprocessing=(
        "official weight transform: resize shorter side to 232 with bilinear interpolation, "
        "center crop 224, rescale to [0,1], ImageNet mean/std normalization"
    ),
    pooling="global-average pool, classifier LayerNorm, flatten; final linear layer removed",
    backend="torchvision",
    source_url="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html",
    license="BSD-3-Clause (torchvision code); checkpoint terms follow upstream",
)

VIT_B16_METADATA = EncoderMetadata(
    name="vit_b16",
    display_name="ViT-B/16",
    checkpoint="torchvision::ViT_B_16_Weights.IMAGENET1K_V1",
    checkpoint_revision="IMAGENET1K_V1",
    feature_dimension=768,
    input_size=(224, 224),
    preprocessing=(
        "official weight transform: resize shorter side to 256 with bilinear interpolation, "
        "center crop 224, rescale to [0,1], ImageNet mean/std normalization"
    ),
    pooling="final encoder class token; classification heads removed",
    backend="torchvision",
    source_url="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html",
    license="BSD-3-Clause (torchvision code); checkpoint terms follow upstream",
)

MOBILENET_V2_METADATA = EncoderMetadata(
    name="mobilenet_v2",
    display_name="MobileNetV2 (optional lightweight baseline)",
    checkpoint="torchvision::MobileNet_V2_Weights.IMAGENET1K_V1",
    checkpoint_revision="IMAGENET1K_V1",
    feature_dimension=1280,
    input_size=(224, 224),
    preprocessing=(
        "official weight transform: resize shorter side to 256 with bilinear interpolation, "
        "center crop 224, rescale to [0,1], ImageNet mean/std normalization"
    ),
    pooling="1280-dimensional adaptive-global-average-pooled feature; classifier removed",
    backend="torchvision",
    source_url="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html",
    license="BSD-3-Clause (torchvision code); checkpoint terms follow upstream",
)


class TorchvisionFrozenEncoder(ImageEncoder):
    """Shared extraction implementation for the Stage 2 torchvision adapters."""

    def __init__(
        self,
        name: str,
        *,
        pretrained: bool = True,
        device: str = "auto",
        model_cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
            from torchvision.models import (
                ConvNeXt_Base_Weights,
                Inception_V3_Weights,
                MobileNet_V2_Weights,
                VGG16_Weights,
                ViT_B_16_Weights,
                convnext_base,
                inception_v3,
                mobilenet_v2,
                vgg16,
                vit_b_16,
            )
        except ImportError as error:
            raise EncoderUnavailableError(
                "This encoder requires torch and torchvision. Install LookAgain-ML's "
                "declared dependencies and retry."
            ) from error

        definitions = {
            "convnext_b": (
                CONVNEXT_B_METADATA,
                convnext_base,
                ConvNeXt_Base_Weights.IMAGENET1K_V1,
                lambda model: model.classifier.__setitem__(2, nn.Identity()),
            ),
            "vit_b16": (
                VIT_B16_METADATA,
                vit_b_16,
                ViT_B_16_Weights.IMAGENET1K_V1,
                lambda model: setattr(model, "heads", nn.Identity()),
            ),
            "mobilenet_v2": (
                MOBILENET_V2_METADATA,
                mobilenet_v2,
                MobileNet_V2_Weights.IMAGENET1K_V1,
                lambda model: setattr(model, "classifier", nn.Identity()),
            ),
            "vgg16": (
                VGG16_METADATA,
                vgg16,
                VGG16_Weights.IMAGENET1K_V1,
                lambda model: None,
            ),
            "inception_v3": (
                INCEPTION_V3_METADATA,
                lambda **kwargs: inception_v3(aux_logits=True, **kwargs),
                Inception_V3_Weights.IMAGENET1K_V1,
                lambda model: (setattr(model, "fc", nn.Identity()), setattr(model, "AuxLogits", None)),
            ),
        }
        if name not in definitions:
            raise ValueError(f"Unknown torchvision Stage 2 encoder {name!r}.")
        metadata, builder, published_weights, strip_classifier = definitions[name]
        self._torch = torch
        self._device = resolve_device(device, torch)
        if model_cache_dir is not None:
            torch.hub.set_dir(str(Path(model_cache_dir).expanduser().resolve() / "torch"))
        weights = published_weights if pretrained else None
        try:
            model = builder(weights=weights)
        except Exception as error:
            if pretrained:
                raise EncoderUnavailableError(
                    f"The published {metadata.display_name} checkpoint could not be loaded. "
                    "The first run may need internet access and sufficient model-cache space. "
                    f"Original error: {error}"
                ) from error
            raise
        strip_classifier(model)
        model.eval().to(self._device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._name = name
        self._transform = published_weights.transforms()
        self.metadata = metadata if pretrained else EncoderMetadata(
            **{
                **metadata.to_dict(),
                "checkpoint": "untrained weights (testing/infrastructure checks only)",
                "checkpoint_revision": None,
            }
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
                if self._name == "vgg16":
                    values_tensor = self._model.features(batch).mean(dim=(2, 3))
                else:
                    values_tensor = self._model(batch)
                values = values_tensor.detach().float().cpu().numpy()
                if values.ndim != 2 or values.shape[1] != self.metadata.feature_dimension:
                    raise RuntimeError(
                        f"{self.metadata.display_name} returned {values.shape}; expected "
                        f"(*, {self.metadata.feature_dimension})."
                    )
                rows.append(values)
                self._report_batch_progress(min(start + batch_size, len(paths)), len(paths))
        elapsed = time.perf_counter() - started
        output = np.vstack(rows).astype(np.float32, copy=False)
        if output.shape[0] != len(paths) or not np.isfinite(output).all():
            raise RuntimeError(
                f"{self.metadata.display_name} features are missing, misaligned, or non-finite."
            )
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


def create_convnext_b(**kwargs: Any) -> TorchvisionFrozenEncoder:
    return TorchvisionFrozenEncoder("convnext_b", **kwargs)


def create_vit_b16(**kwargs: Any) -> TorchvisionFrozenEncoder:
    return TorchvisionFrozenEncoder("vit_b16", **kwargs)


def create_mobilenet_v2(**kwargs: Any) -> TorchvisionFrozenEncoder:
    return TorchvisionFrozenEncoder("mobilenet_v2", **kwargs)


def create_vgg16(**kwargs: Any) -> TorchvisionFrozenEncoder:
    return TorchvisionFrozenEncoder("vgg16", **kwargs)


def create_inception_v3(**kwargs: Any) -> TorchvisionFrozenEncoder:
    return TorchvisionFrozenEncoder("inception_v3", **kwargs)
