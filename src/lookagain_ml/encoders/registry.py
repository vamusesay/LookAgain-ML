"""Small, explicit encoder registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..exceptions import InputValidationError
from .base import EncoderMetadata, ImageEncoder

EncoderFactory = Callable[..., ImageEncoder]


@dataclass(frozen=True)
class EncoderRegistration:
    name: str
    factory: EncoderFactory
    metadata: EncoderMetadata


_REGISTRY: dict[str, EncoderRegistration] = {}


def register_encoder(
    name: str,
    factory: EncoderFactory,
    metadata: EncoderMetadata,
    *,
    replace: bool = False,
) -> None:
    """Register an adapter factory and metadata under a stable lowercase key."""

    key = name.strip().lower()
    if not key or key != metadata.name:
        raise ValueError("The registry key must be nonempty and match metadata.name.")
    if key in _REGISTRY and not replace:
        raise ValueError(f"Encoder {key!r} is already registered.")
    _REGISTRY[key] = EncoderRegistration(key, factory, metadata)


def unregister_encoder(name: str) -> EncoderRegistration:
    """Remove and return an adapter registration (primarily useful in tests)."""

    return _REGISTRY.pop(name.strip().lower())


def get_registration(name: str) -> EncoderRegistration:
    key = name.strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as error:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise InputValidationError(
            f"Unknown encoder {name!r}. Registered encoders: {available}."
        ) from error


def create_encoder(name: str, **kwargs: Any) -> ImageEncoder:
    """Instantiate one registered encoder adapter."""

    registration = get_registration(name)
    encoder = registration.factory(**kwargs)
    if encoder.metadata.name != registration.name:
        raise RuntimeError(
            f"Encoder factory {registration.name!r} returned metadata for {encoder.metadata.name!r}."
        )
    return encoder


def list_encoders() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def encoder_metadata(name: str) -> EncoderMetadata:
    return get_registration(name).metadata



