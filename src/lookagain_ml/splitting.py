"""Deterministic train/validation/test splitting with leakage safeguards."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .exceptions import InputValidationError, LeakageError

SPLIT_NAMES = ("train", "validation", "test")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class SplitResult:
    labels: np.ndarray
    effective_groups: np.ndarray


def combine_leakage_groups(
    groups: Iterable[Any], image_hashes: Iterable[str]
) -> np.ndarray:
    """Union user groups and exact hashes into indivisible split units."""

    group_values = np.asarray(list(groups), dtype=object)
    hash_values = np.asarray(list(image_hashes), dtype=object)
    if len(group_values) != len(hash_values):
        raise InputValidationError("groups and image hashes must have the same length.")
    union = _UnionFind(len(group_values))
    for values in (group_values, hash_values):
        first: dict[Any, int] = {}
        for index, value in enumerate(values):
            if value in first:
                union.union(index, first[value])
            else:
                first[value] = index
    roots = [union.find(i) for i in range(len(group_values))]
    root_to_name = {root: f"effective-group-{j:08d}" for j, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_name[root] for root in roots], dtype=object)


def _normalize_split_labels(values: Iterable[Any]) -> np.ndarray:
    aliases = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
    normalized: list[str] = []
    for position, raw in enumerate(values):
        key = str(raw).strip().lower()
        if key not in aliases:
            raise InputValidationError(
                f"split_labels row {position} is {raw!r}; use train, validation/val, or test."
            )
        normalized.append(aliases[key])
    labels = np.asarray(normalized, dtype=object)
    missing = set(SPLIT_NAMES) - set(labels)
    if missing:
        raise InputValidationError(
            "split_labels must include nonempty train, validation, and test partitions; "
            f"missing: {sorted(missing)}."
        )
    return labels


def validate_partition_safety(
    labels: Iterable[str], groups: Iterable[Any], image_hashes: Iterable[str]
) -> None:
    """Fail if a user group or exact image hash crosses partitions."""

    label_values = np.asarray(list(labels), dtype=object)
    for name, values in (("group", groups), ("exact image hash", image_hashes)):
        by_value: dict[Any, set[str]] = {}
        for value, label in zip(values, label_values):
            by_value.setdefault(value, set()).add(str(label))
        crossing = [value for value, seen in by_value.items() if len(seen) > 1]
        if crossing:
            example = ", ".join(repr(value) for value in crossing[:3])
            raise LeakageError(
                f"Severe leakage: {len(crossing)} {name}(s) cross train/validation/test "
                f"partitions (examples: {example}). Keep each {name} in one partition."
            )


def _classification_penalty(y: np.ndarray, selected: np.ndarray) -> float:
    classes, overall = np.unique(y, return_counts=True)
    overall_share = overall / overall.sum()
    observed_classes, observed = np.unique(y[selected], return_counts=True)
    observed_map = dict(zip(observed_classes.tolist(), observed.tolist()))
    share = np.asarray([observed_map.get(value, 0) for value in classes], dtype=float)
    share /= max(share.sum(), 1.0)
    missing_penalty = float(np.sum(share == 0)) * 10.0
    return missing_penalty + float(np.abs(share - overall_share).sum())


def _missing_class_count(y: np.ndarray, selected: np.ndarray) -> int:
    """Count classes excluded from a candidate keep partition."""

    return int(len(set(np.unique(y)) - set(np.unique(y[selected]))))


def _choose_group_holdout(
    indices: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    fraction: float,
    *,
    task: str,
    random_state: int,
    minimum_keep_groups_per_class: int = 1,
    require_holdout_class_coverage: bool = False,
    trials: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups[indices])
    if len(unique_groups) < 2:
        raise InputValidationError(
            "Not enough independent groups to create the requested partitions. "
            "Provide at least three leakage-safe groups overall."
        )
    splitter = GroupShuffleSplit(
        n_splits=trials, test_size=fraction, random_state=random_state
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    dummy = np.zeros(len(indices), dtype=np.uint8)
    class_group_counts: np.ndarray | None = None
    group_class_matrix: np.ndarray | None = None
    local_group_codes: np.ndarray | None = None
    if task == "classification":
        _, local_class_codes = np.unique(y[indices], return_inverse=True)
        _, local_group_codes = np.unique(groups[indices], return_inverse=True)
        group_class_matrix = np.zeros(
            (local_group_codes.max() + 1, local_class_codes.max() + 1),
            dtype=bool,
        )
        group_class_matrix[local_group_codes, local_class_codes] = True
        class_group_counts = group_class_matrix.sum(axis=0)
    for keep_local, hold_local in splitter.split(dummy, groups=groups[indices]):
        keep, hold = indices[keep_local], indices[hold_local]
        size_penalty = abs(len(hold) / len(indices) - fraction)
        if task == "classification":
            # The keep side becomes train+validation in the first call and the
            # actual training partition in the second. Missing a class there
            # makes the fitted classification head undefined. The holdout may
            # legitimately omit a rare class, so preserving keep-side support
            # takes priority over matching every class in the holdout.
            assert class_group_counts is not None
            assert group_class_matrix is not None
            assert local_group_codes is not None
            hold_group_codes = np.unique(local_group_codes[hold_local])
            hold_group_counts = group_class_matrix[hold_group_codes].sum(axis=0)
            keep_group_counts = class_group_counts - hold_group_counts
            keep_missing = int(np.sum(keep_group_counts == 0))
            keep_support_shortfall = int(
                np.sum(keep_group_counts < minimum_keep_groups_per_class)
            )
            hold_missing = (
                int(np.sum(hold_group_counts == 0))
                if require_holdout_class_coverage
                else 0
            )
            class_penalty = (
                1_000.0 * (keep_missing + keep_support_shortfall + hold_missing)
                + _classification_penalty(y[indices], hold_local)
            )
        else:
            class_penalty = 0.0
        score = size_penalty + class_penalty
        if best is None or score < best[0]:
            best = (score, keep, hold)
    assert best is not None
    return best[1], best[2]


def make_split(
    y: Iterable[Any],
    groups: Iterable[Any],
    image_hashes: Iterable[str],
    *,
    task: str,
    random_state: int = 20260818,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_labels: Iterable[Any] | None = None,
) -> SplitResult:
    """Create or validate one locked, group- and duplicate-safe partition."""

    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise InputValidationError(
            "train_fraction, validation_fraction, and test_fraction must be positive and sum to 1."
        )
    y_values = np.asarray(list(y), dtype=object)
    group_values = np.asarray(list(groups), dtype=object)
    hash_values = np.asarray(list(image_hashes), dtype=object)
    if not (len(y_values) == len(group_values) == len(hash_values)):
        raise InputValidationError("y, groups, and image hashes must have equal lengths.")
    effective = combine_leakage_groups(group_values, hash_values)

    if split_labels is not None:
        labels = _normalize_split_labels(split_labels)
        if len(labels) != len(y_values):
            raise InputValidationError(
                f"split_labels has {len(labels)} rows but the validated manifest has {len(y_values)}."
            )
        validate_partition_safety(labels, group_values, hash_values)
        validate_partition_safety(labels, effective, hash_values)
        return SplitResult(labels=labels, effective_groups=effective)

    if len(np.unique(effective)) < 3:
        raise InputValidationError(
            "At least three independent groups are required for train/validation/test splitting."
        )
    all_indices = np.arange(len(y_values))
    full_three_way_class_coverage = False
    if task == "classification":
        full_three_way_class_coverage = all(
            len(np.unique(effective[y_values == value])) >= 3
            for value in np.unique(y_values)
        )
    train_val, test = _choose_group_holdout(
        all_indices,
        effective,
        y_values,
        test_fraction,
        task=task,
        random_state=random_state,
        minimum_keep_groups_per_class=(
            2 if full_three_way_class_coverage else 1
        ),
    )
    relative_validation = validation_fraction / (train_fraction + validation_fraction)
    train, validation = _choose_group_holdout(
        train_val,
        effective,
        y_values,
        relative_validation,
        task=task,
        random_state=random_state + 1,
        require_holdout_class_coverage=full_three_way_class_coverage,
    )
    if task == "classification" and set(np.unique(y_values[train])) != set(
        np.unique(y_values)
    ):
        raise InputValidationError(
            "A group-safe training partition containing every class could not be "
            "constructed. Ensure that rare classes occur in enough independent "
            "groups or supply a validated split explicitly."
        )
    if (
        task == "classification"
        and full_three_way_class_coverage
        and set(np.unique(y_values[validation])) != set(np.unique(y_values))
    ):
        raise InputValidationError(
            "A group-safe validation partition containing every class could not be "
            "constructed even though each class occurs in at least three independent "
            "groups. Increase the number of independent observations or supply a "
            "validated split explicitly."
        )
    labels = np.empty(len(y_values), dtype=object)
    labels[train], labels[validation], labels[test] = "train", "validation", "test"
    validate_partition_safety(labels, group_values, hash_values)
    validate_partition_safety(labels, effective, hash_values)
    return SplitResult(labels=labels, effective_groups=effective)
