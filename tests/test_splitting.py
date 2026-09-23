import numpy as np
import pytest

from lookagain_ml.exceptions import LeakageError
from lookagain_ml.splitting import (
    combine_leakage_groups,
    make_split,
    validate_partition_safety,
)


def test_duplicate_hash_connects_distinct_user_groups():
    effective = combine_leakage_groups(
        ["a", "b", "b", "c"], ["same", "same", "other", "last"]
    )
    assert effective[0] == effective[1] == effective[2]
    assert effective[3] != effective[0]


def test_group_safe_split_is_deterministic_and_leakage_free():
    y = np.asarray([index % 2 for index in range(60)])
    groups = np.asarray([f"group-{index // 2}" for index in range(60)])
    hashes = np.asarray([f"hash-{index}" for index in range(60)])
    first = make_split(y, groups, hashes, task="classification", random_state=17)
    second = make_split(y, groups, hashes, task="classification", random_state=17)
    assert np.array_equal(first.labels, second.labels)
    assert set(first.labels) == {"train", "validation", "test"}
    validate_partition_safety(first.labels, groups, hashes)
    for group in np.unique(groups):
        assert len(set(first.labels[groups == group])) == 1
    expected = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for split_name, expected_share in expected.items():
        selected = first.labels == split_name
        assert abs(selected.mean() - expected_share) <= 0.05
        assert abs(y[selected].mean() - y.mean()) <= 0.10


def test_explicit_test_is_never_repartitioned():
    labels = ["train"] * 6 + ["validation"] * 3 + ["test"] * 3
    groups = [f"g-{i}" for i in range(12)]
    hashes = [f"h-{i}" for i in range(12)]
    result = make_split(
        range(12),
        groups,
        hashes,
        task="regression",
        split_labels=labels,
    )
    assert result.labels.tolist() == labels


def test_explicit_group_leakage_fails_loudly():
    labels = ["train", "validation", "test", "train"]
    groups = ["crossing", "crossing", "safe-1", "safe-2"]
    hashes = ["h1", "h2", "h3", "h4"]
    with pytest.raises(LeakageError, match="Severe leakage"):
        make_split(
            range(4),
            groups,
            hashes,
            task="regression",
            split_labels=labels,
        )


def test_explicit_hash_leakage_fails_loudly():
    labels = ["train", "validation", "test"]
    groups = ["a", "b", "c"]
    hashes = ["duplicate", "duplicate", "other"]
    with pytest.raises(LeakageError, match="exact image hash"):
        validate_partition_safety(labels, groups, hashes)


def test_generated_multiclass_split_keeps_single_group_rare_class_in_training():
    """A rare class must not be sacrificed to make the holdout look stratified."""

    y = ["common"] * 20 + ["rare"]
    groups = [f"common-{index}" for index in range(20)] + ["rare-only-group"]
    hashes = [f"hash-{index}" for index in range(21)]

    result = make_split(
        y,
        groups,
        hashes,
        task="classification",
        random_state=20261827,
    )

    assert result.labels[-1] == "train"
    assert set(np.asarray(y, dtype=object)[result.labels == "train"]) == {
        "common",
        "rare",
    }


def test_generated_multiclass_split_preserves_feasible_validation_coverage():
    """Validation-only combination fitting needs every feasible class."""

    y = ["common"] * 30 + ["rare"] * 4
    groups = [f"common-{index}" for index in range(30)] + [
        f"rare-{index}" for index in range(4)
    ]
    hashes = [f"hash-{index}" for index in range(34)]

    result = make_split(
        y,
        groups,
        hashes,
        task="classification",
        random_state=20261827,
    )

    values = np.asarray(y, dtype=object)
    assert set(values[result.labels == "train"]) == {"common", "rare"}
    assert set(values[result.labels == "validation"]) == {"common", "rare"}
