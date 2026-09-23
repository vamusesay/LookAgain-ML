"""DINOv2 and SigLIP 2 frozen-image adapters backed by Transformers."""

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

DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


DINOV2_METADATA = EncoderMetadata(
    name="dinov2_vitb14",
    display_name="DINOv2-B/14",
    checkpoint="facebook/dinov2-base",
    checkpoint_revision=DINOV2_REVISION,
    feature_dimension=768,
    input_size=(224, 224),
    preprocessing=(
        "official AutoImageProcessor: RGB, resize shorter side to 256 with bicubic "
        "interpolation, center crop 224, ImageNet mean/std normalization"
    ),
    pooling="final hidden-state class token (last_hidden_state[:, 0])",
    backend="transformers",
    source_url="https://huggingface.co/facebook/dinov2-base",
    license="Apache-2.0",
)

SIGLIP2_METADATA = EncoderMetadata(
    name="siglip2_b16",
    display_name="SigLIP 2-B/16",
    checkpoint="google/siglip2-base-patch16-224",
    checkpoint_revision=SIGLIP2_REVISION,
    feature_dimension=768,
    input_size=(224, 224),
    preprocessing=(
        "official image processor for the pinned SigLIP 2 checkpoint: RGB, resize to "
        "224x224, rescale to [0,1], mean/std 0.5"
    ),
    pooling="768-dimensional pooled vision output returned by get_image_features",
    backend="transformers",
    source_url="https://huggingface.co/google/siglip2-base-patch16-224",
    license="Apache-2.0",
)


class HuggingFaceFrozenEncoder(ImageEncoder):
    """Shared batched inference for the manuscript-specified HF encoders."""

    def __init__(
        self,
        name: str,
        *,
        pretrained: bool = True,
        device: str = "auto",
        model_cache_dir: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        if not pretrained:
            raise EncoderUnavailableError(
                f"{name} has no untrained public mode. Its representation is defined by the "
                "published checkpoint; use pretrained=True."
            )
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise EncoderUnavailableError(
                "DINOv2 and SigLIP 2 require LookAgain-ML's modern extra: "
                "pip install 'lookagain-ml[modern]'."
            ) from error

        definitions = {
            "dinov2_vitb14": DINOV2_METADATA,
            "siglip2_b16": SIGLIP2_METADATA,
        }
        if name not in definitions:
            raise ValueError(f"Unknown Hugging Face Stage 2 encoder {name!r}.")
        metadata = definitions[name]
        requested_revision = revision or metadata.checkpoint_revision
        assert requested_revision is not None
        cache_dir = str(Path(model_cache_dir).expanduser().resolve()) if model_cache_dir else None
        self._torch = torch
        self._device = resolve_device(device, torch)
        try:
            processor = AutoImageProcessor.from_pretrained(
                metadata.checkpoint, revision=requested_revision, cache_dir=cache_dir
            )
            model = AutoModel.from_pretrained(
                metadata.checkpoint, revision=requested_revision, cache_dir=cache_dir
            )
        except Exception as error:
            raise EncoderUnavailableError(
                f"The published {metadata.display_name} checkpoint could not be loaded. "
                "Install the modern extra and ensure the project model cache has sufficient "
                f"space and network access. Original error: {error}"
            ) from error
        model.eval().to(self._device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        resolved_revision = getattr(model.config, "_commit_hash", None) or requested_revision
        if str(resolved_revision) != str(requested_revision):
            raise EncoderUnavailableError(
                f"{metadata.display_name} resolved checkpoint revision {resolved_revision}, but "
                f"LookAgain-ML requested the pinned revision {requested_revision}."
            )
        self.metadata = EncoderMetadata(
            **{**metadata.to_dict(), "checkpoint_revision": str(resolved_revision)}
        )
        self._name = name
        self._processor = processor
        self._model = model
        representation_model = (
            getattr(model, "vision_model", model) if name == "siglip2_b16" else model
        )
        self.costs: dict[str, Any] = {
            "device": self._device,
            "parameter_count": int(
                sum(parameter.numel() for parameter in representation_model.parameters())
            ),
            "parameter_count_scope": "frozen image representation encoder",
            "loaded_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "checkpoint_revision": str(resolved_revision),
        }

    @staticmethod
    def _open_images(paths: Sequence[str | Path]) -> list[Image.Image]:
        images: list[Image.Image] = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
        except Exception as error:
            raise ImageLoadError(
                f"An image passed manifest validation but failed during encoding: {error}"
            ) from error
        return images

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
                images = self._open_images(paths[start : start + batch_size])
                inputs = self._processor(images=images, return_tensors="pt")
                inputs = {
                    key: value.to(self._device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                if self._name == "dinov2_vitb14":
                    values = self._model(**inputs).last_hidden_state[:, 0]
                else:
                    values = self._model.get_image_features(**inputs)
                    if not hasattr(values, "detach") and hasattr(values, "pooler_output"):
                        values = values.pooler_output
                array = values.detach().float().cpu().numpy()
                if array.ndim != 2 or array.shape[1] != self.metadata.feature_dimension:
                    raise RuntimeError(
                        f"{self.metadata.display_name} returned {array.shape}; expected "
                        f"(*, {self.metadata.feature_dimension})."
                    )
                rows.append(array)
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


def create_dinov2_vitb14(**kwargs: Any) -> HuggingFaceFrozenEncoder:
    return HuggingFaceFrozenEncoder("dinov2_vitb14", **kwargs)


def create_siglip2_b16(**kwargs: Any) -> HuggingFaceFrozenEncoder:
    return HuggingFaceFrozenEncoder("siglip2_b16", **kwargs)
