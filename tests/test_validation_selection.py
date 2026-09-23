from __future__ import annotations

import numpy as np

from lookagain_ml import nested_validation_selection


def test_outer_validation_outcomes_do_not_fit_candidate() -> None:
    rng = np.random.default_rng(17)
    n = 180
    groups = np.asarray([f"g{i // 2}" for i in range(n)], dtype=object)
    split = np.asarray(["train"] * 120 + ["validation"] * 30 + ["test"] * 30)
    signal = rng.normal(size=n)
    outcome = 2.0 * signal + rng.normal(scale=0.4, size=n)
    features = {
        "a": np.column_stack([signal, rng.normal(size=n)]),
        "b": np.column_stack([0.7 * signal + rng.normal(scale=0.5, size=n), rng.normal(size=n)]),
    }
    first = nested_validation_selection(
        features,
        outcome,
        groups,
        split,
        task="regression",
        encoder_order=["a", "b"],
        head_max_components=None,
        inner_folds=3,
        random_state=9,
    )
    changed = outcome.copy()
    changed[split == "validation"] += 1000.0
    second = nested_validation_selection(
        features,
        changed,
        groups,
        split,
        task="regression",
        encoder_order=["a", "b"],
        head_max_components=None,
        inner_folds=3,
        random_state=9,
    )
    assert np.allclose(first.validation_predictions["linear_stack"], second.validation_predictions["linear_stack"])
    assert np.allclose(first.validation_predictions["greedy_average"], second.validation_predictions["greedy_average"])
    assert first.audit["outer_validation_outcomes_used_to_fit_candidates"] is False


def test_group_safe_nested_classification_is_finite() -> None:
    rng = np.random.default_rng(23)
    groups = np.asarray([f"g{i // 2}" for i in range(240)], dtype=object)
    split = np.asarray(["train"] * 160 + ["validation"] * 40 + ["test"] * 40)
    x = rng.normal(size=(240, 4))
    y = (x[:, 0] + 0.2 * rng.normal(size=240) > 0).astype(int)
    result = nested_validation_selection(
        {"a": x, "b": np.column_stack([x[:, 0] + rng.normal(size=240), x[:, 1:]])},
        y,
        groups,
        split,
        task="classification",
        encoder_order=["a", "b"],
        head_max_components=None,
        inner_folds=4,
        random_state=11,
    )
    assert result.preferred_method in {"a", "b", "equal_average", "linear_stack", "greedy_average", "feature_concatenation"}
    assert all(np.isfinite(value).all() for value in result.test_predictions.values())
    assert result.audit["test_outcomes_used_for_fitting_or_selection"] is False
