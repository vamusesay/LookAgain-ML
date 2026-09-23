"""Encoder interface shared by built-in and third-party adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncoderMetadata:
    """Auditable description of a frozen image representation."""

    name: str
    display_name: str
    checkpoint: str
    feature_dimension: int
    input_size: tuple[int, int]
    preprocessing: str
    pooling: str
    backend: str
    checkpoint_revision: str | None = None
    source_url: str | None = None
    license: str | None = None
    frozen: bool = True
    schema_version: str = "2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageEncoder(ABC):
    """Minimal adapter contract for a frozen image encoder."""

    metadata: EncoderMetadata
    costs: dict[str, Any]

    def set_batch_progress_callback(
        self, callback: Callable[[int, int], None] | None
    ) -> None:
        """Register an optional completed-row callback for built-in encoders."""

        self._batch_progress_callback = callback

    def _report_batch_progress(self, completed: int, total: int) -> None:
        callback = getattr(self, "_batch_progress_callback", None)
        if callback is not None:
            callback(int(completed), int(total))

    @abstractmethod
    def encode(self, paths: Sequence[str | Path], *, batch_size: int = 32) -> np.ndarray:
        """Return one finite feature row per path, preserving input order."""

