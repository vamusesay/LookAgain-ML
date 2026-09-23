"""Public exception hierarchy for LookAgain."""

from __future__ import annotations

from typing import Any


class LookAgainError(Exception):
    """Base class for errors raised by LookAgain."""


class InputValidationError(LookAgainError, ValueError):
    """Raised when inputs cannot be analyzed safely."""

    def __init__(self, message: str, *, audit: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.audit = audit


class LeakageError(LookAgainError, ValueError):
    """Raised when groups or exact duplicate images cross partitions."""


class EncoderUnavailableError(LookAgainError, ImportError):
    """Raised when an encoder dependency or checkpoint is unavailable."""


class CacheValidationError(LookAgainError, RuntimeError):
    """Raised when an existing representation cache cannot be verified safely."""


class CheckpointValidationError(LookAgainError, RuntimeError):
    """Raised when a run checkpoint is partial, corrupt, or incompatible."""


class ImageLoadError(LookAgainError, OSError):
    """Raised when an image fails during representation extraction."""

