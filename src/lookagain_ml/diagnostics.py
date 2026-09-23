"""Representation diagnostics that never use locked-test outcomes for decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import InputValidationError

SEMANTIC_REPRESENTATIONS = frozenset({"coco_deeplab", "ade20k_segformer"})


@dataclass(frozen=True)
class LinearCKAResult:
    """Pairwise linear CKA plus an explicit sampling/eligibility audit."""

    matrix: pd.DataFrame
    audit: dict[str, Any]


def linear_cka(
    features: Mapping[str, np.ndarray],
    *,
    rows: Sequence[int] | np.ndarray | None = None,
    semantic_representations: Sequence[str] = tuple(SEMANTIC_REPRESENTATIONS),
) -> LinearCKAResult:
    """Compute centered linear CKA for aligned pooled neural features.

    Semantic class-share vectors are excluded by default because the paper's
    CKA comparison is defined only for pooled neural representation blocks.
    No outcomes are accepted by this function.
    """

    if not features:
        raise InputValidationError("linear_cka requires at least one feature block.")
    excluded = set(map(str, semantic_representations))
    names = [str(name) for name in features if str(name) not in excluded]
    omitted = [str(name) for name in features if str(name) in excluded]
    if not names:
        raise InputValidationError(
            "No CKA-eligible pooled neural representation remains after semantic exclusion."
        )
    row_count = None
    checked: dict[str, np.ndarray] = {}
    for name in names:
        value = np.asarray(features[name], dtype=np.float64)
        if value.ndim != 2 or value.shape[1] < 1 or not np.isfinite(value).all():
            raise InputValidationError(f"Feature block {name!r} must be finite and two-dimensional.")
        row_count = value.shape[0] if row_count is None else row_count
        if value.shape[0] != row_count:
            raise InputValidationError("All CKA feature blocks must have the same row count.")
        checked[name] = value
    assert row_count is not None
    selected = np.arange(row_count) if rows is None else np.asarray(rows, dtype=int)
    if selected.ndim != 1 or len(selected) < 2:
        raise InputValidationError("linear_cka requires at least two selected rows.")
    if np.any(selected < 0) or np.any(selected >= row_count) or len(np.unique(selected)) != len(selected):
        raise InputValidationError("CKA row indices must be unique and in bounds.")
    grams: dict[str, np.ndarray] = {}
    for name, value in checked.items():
        centered = value[selected] - value[selected].mean(axis=0, keepdims=True)
        gram = centered @ centered.T
        gram -= gram.mean(axis=0, keepdims=True)
        gram -= gram.mean(axis=1, keepdims=True)
        gram += gram.mean()
        grams[name] = gram
    output = np.eye(len(names), dtype=float)
    for left_index, left in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right = names[right_index]
            numerator = float(np.sum(grams[left] * grams[right]))
            denominator = float(
                np.sqrt(np.sum(grams[left] ** 2) * np.sum(grams[right] ** 2))
            )
            if denominator <= 0:
                raise InputValidationError(
                    f"CKA is undefined for constant centered feature block {left!r} or {right!r}."
                )
            output[left_index, right_index] = output[right_index, left_index] = numerator / denominator
    return LinearCKAResult(
        matrix=pd.DataFrame(output, index=names, columns=names),
        audit={
            "method": "centered linear CKA",
            "row_count": len(selected),
            "row_scope": "caller-selected; use training rows for model-selection diagnostics",
            "outcomes_used": False,
            "included_representations": names,
            "excluded_semantic_representations": omitted,
            "interpretation": "descriptive representation similarity, not predictive superiority",
        },
    )
