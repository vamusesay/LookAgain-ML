"""Exact, streaming file hashing used for duplicate safeguards."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, block_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    candidate = Path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def digest_strings(values: list[str] | tuple[str, ...]) -> str:
    """Create an order-sensitive digest for manifest identity checks."""

    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()



