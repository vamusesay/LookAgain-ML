"""Optional semantic class-share representations used in the ICLR analysis."""

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

ADE20K_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"

COCO_DEEPLAB_METADATA = EncoderMetadata(
    name="coco_deeplab",
    display_name="COCO DeepLab semantic class shares (optional)",
    checkpoint=(
        "torchvision::DeepLabV3_MobileNet_V3_Large_Weights."
        "COCO_WITH_VOC_LABELS_V1"
    ),
    checkpoint_revision="COCO_WITH_VOC_LABELS_V1",
    feature_dimension=21,
    input_size=(224, 224),
    preprocessing=(
        "paper-validated transform: RGB, direct 224x224 bilinear resize, rescale to "
        "[0,1], ImageNet mean/std normalization. Upstream's generic inference transform "
        "uses resize 520; 224 is retained to reproduce the validated paper extraction."
    ),
    pooling="softmax over 21 segmentation logits, then spatial mean class probabilities",
    backend="torchvision",
    source_url=(
        "https://docs.pytorch.org/vision/stable/models/generated/"
        "torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html"
    ),
    license="BSD-3-Clause (torchvision code); checkpoint terms follow upstream",
)

ADE20K_SEGFORMER_METADATA = EncoderMetadata(
    name="ade20k_segformer",
    display_name="ADE20K SegFormer semantic class shares (optional)",
    checkpoint="nvidia/segformer-b0-finetuned-ade-512-512",
    checkpoint_revision=ADE20K_REVISION,
    feature_dimension=150,
    input_size=(512, 512),
    preprocessing=(
        "checkpoint SegformerImageProcessor mean/std and 512x512 bilinear resize; RGB input"
    ),
    pooling="hard argmax segmentation map followed by normalized 150-class pixel histogram",
    backend="transformers",
    source_url="https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512",
    license="NVIDIA model-card terms; verify upstream dataset/model terms for the intended use",
)


class CocoDeepLabEncoder(ImageEncoder):
    def __init__(
        self,
        *,
        pretrained: bool = True,
        device: str = "auto",
        model_cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from torchvision import transforms
            from torchvision.models.segmentation import (
                DeepLabV3_MobileNet_V3_Large_Weights,
                deeplabv3_mobilenet_v3_large,
            )
            from torchvision.transforms import InterpolationMode
        except ImportError as error:
            raise EncoderUnavailableError(
                "coco_deeplab requires torch and torchvision."
            ) from error
        self._torch = torch
        self._device = resolve_device(device, torch)
        if model_cache_dir is not None:
            torch.hub.set_dir(str(Path(model_cache_dir).expanduser().resolve() / "torch"))
        published = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        try:
            model = deeplabv3_mobilenet_v3_large(
                weights=published if pretrained else None,
                weights_backbone=None,
            )
        except Exception as error:
            raise EncoderUnavailableError(
                "The COCO-with-VOC-labels DeepLab checkpoint could not be loaded. "
                f"Original error: {error}"
            ) from error
        model.eval().to(self._device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self.metadata = COCO_DEEPLAB_METADATA if pretrained else EncoderMetadata(
            **{
                **COCO_DEEPLAB_METADATA.to_dict(),
                "checkpoint": "untrained weights (testing/infrastructure checks only)",
                "checkpoint_revision": None,
            }
        )
        self.costs: dict[str, Any] = {
            "device": self._device,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "parameter_count_scope": "frozen semantic segmentation model",
        }

    def _load(self, path: str | Path):
        try:
            with Image.open(path) as image:
                return self._transform(image.convert("RGB"))
        except Exception as error:
            raise ImageLoadError(f"Could not load image {str(path)!r}: {error}") from error

    def encode(self, paths: Sequence[str | Path], *, batch_size: int = 32) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        torch = self._torch
        rows: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                batch = torch.stack(
                    [self._load(path) for path in paths[start : start + batch_size]]
                ).to(self._device)
                logits = self._model(batch)["out"]
                rows.append(logits.softmax(dim=1).mean(dim=(2, 3)).cpu().numpy())
                self._report_batch_progress(min(start + batch_size, len(paths)), len(paths))
        output = np.vstack(rows).astype(np.float32, copy=False)
        if output.shape != (len(paths), 21) or not np.isfinite(output).all():
            raise RuntimeError("COCO semantic features are missing, misaligned, or non-finite.")
        elapsed = time.perf_counter() - started
        self.costs.update(
            extraction_seconds=elapsed,
            images_per_second=len(paths) / max(elapsed, 1e-12),
            feature_dimension=21,
        )
        return output


class Ade20kSegformerEncoder(ImageEncoder):
    def __init__(
        self,
        *,
        pretrained: bool = True,
        device: str = "auto",
        model_cache_dir: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        if not pretrained:
            raise EncoderUnavailableError(
                "ade20k_segformer has no untrained public mode because the representation "
                "is defined by the published fine-tuned checkpoint."
            )
        try:
            import torch
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except ImportError as error:
            raise EncoderUnavailableError(
                "ade20k_segformer requires the modern extra: pip install 'lookagain-ml[modern]'."
            ) from error
        requested_revision = revision or ADE20K_REVISION
        cache_dir = str(Path(model_cache_dir).expanduser().resolve()) if model_cache_dir else None
        self._torch = torch
        self._device = resolve_device(device, torch)
        try:
            processor = SegformerImageProcessor.from_pretrained(
                ADE20K_SEGFORMER_METADATA.checkpoint,
                revision=requested_revision,
                cache_dir=cache_dir,
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                ADE20K_SEGFORMER_METADATA.checkpoint,
                revision=requested_revision,
                cache_dir=cache_dir,
            )
        except Exception as error:
            raise EncoderUnavailableError(
                "The ADE20K SegFormer checkpoint could not be loaded. "
                f"Original error: {error}"
            ) from error
        model.eval().to(self._device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        resolved_revision = getattr(model.config, "_commit_hash", None) or requested_revision
        self.metadata = EncoderMetadata(
            **{
                **ADE20K_SEGFORMER_METADATA.to_dict(),
                "checkpoint_revision": str(resolved_revision),
            }
        )
        self._processor = processor
        self._model = model
        self.costs: dict[str, Any] = {
            "device": self._device,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "parameter_count_scope": "frozen semantic segmentation model",
            "checkpoint_revision": str(resolved_revision),
        }

    def encode(self, paths: Sequence[str | Path], *, batch_size: int = 8) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        torch = self._torch
        rows: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                images: list[Image.Image] = []
                try:
                    for path in paths[start : start + batch_size]:
                        with Image.open(path) as image:
                            images.append(image.convert("RGB").copy())
                except Exception as error:
                    raise ImageLoadError(f"Could not load an ADE20K input image: {error}") from error
                values = self._processor(images=images, return_tensors="pt")
                pixel_values = values["pixel_values"].to(self._device)
                logits = self._model(pixel_values=pixel_values).logits
                labels = logits.argmax(dim=1)
                shares = torch.nn.functional.one_hot(labels, num_classes=150).float().mean(dim=(1, 2))
                rows.append(shares.cpu().numpy())
                self._report_batch_progress(min(start + batch_size, len(paths)), len(paths))
        output = np.vstack(rows).astype(np.float32, copy=False)
        if output.shape != (len(paths), 150) or not np.isfinite(output).all():
            raise RuntimeError("ADE20K semantic features are missing, misaligned, or non-finite.")
        elapsed = time.perf_counter() - started
        self.costs.update(
            extraction_seconds=elapsed,
            images_per_second=len(paths) / max(elapsed, 1e-12),
            feature_dimension=150,
        )
        return output


def create_coco_deeplab(**kwargs: Any) -> CocoDeepLabEncoder:
    return CocoDeepLabEncoder(**kwargs)


def create_ade20k_segformer(**kwargs: Any) -> Ade20kSegformerEncoder:
    return Ade20kSegformerEncoder(**kwargs)
