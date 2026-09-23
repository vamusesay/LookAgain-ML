"""Existing source imports remain aliases of the canonical implementations."""
import inspect
import numpy as np

from lookagain_ml import nested_validation_selection, NestedValidationSelectionResult
from lookagain_ml import honest_nested_selection, HonestNestedSelectionResult
from lookagain_ml import honest_selection as legacy_selection
from lookagain_ml import validation_selection
from lookagain_ml import honest_neural_heads as legacy_neural
from lookagain_ml import inner_selected_neural_heads as neural


def test_selection_legacy_aliases_share_implementation_and_signature():
    assert honest_nested_selection is nested_validation_selection
    assert HonestNestedSelectionResult is NestedValidationSelectionResult
    assert legacy_selection.honest_nested_selection is nested_validation_selection
    assert legacy_selection._inner_holdout is validation_selection._inner_holdout
    assert inspect.signature(honest_nested_selection) == inspect.signature(nested_validation_selection)


def test_neural_legacy_aliases_share_implementation_and_signature():
    assert legacy_neural.fit_honest_neural_heads is neural.fit_inner_selected_neural_heads
    assert legacy_neural.HonestNeuralHeads is neural.InnerSelectedNeuralHeads
    assert legacy_neural.refit_locked_neural_heads is neural.refit_locked_neural_heads
    assert legacy_neural.array_digest is neural.array_digest
    assert inspect.signature(legacy_neural.fit_honest_neural_heads) == inspect.signature(neural.fit_inner_selected_neural_heads)


def test_selection_alias_deterministic_outputs_and_serialized_protocol():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(90, 3))
    y = x[:, 0] + rng.normal(scale=0.2, size=90)
    groups = np.repeat(np.arange(45), 2)
    splits = np.array(['train'] * 54 + ['validation'] * 18 + ['test'] * 18)
    args = dict(task='regression', encoder_order=('a', 'b'), inner_folds=3,
                head_max_components=None, ridge_alphas=(0.1, 1.0), random_state=19)
    features = {'a': x, 'b': x[:, :2]}
    current = nested_validation_selection(features, y, groups, splits, **args)
    legacy = honest_nested_selection(features, y, groups, splits, **args)
    assert current.audit == legacy.audit
    assert current.selected_hyperparameters == legacy.selected_hyperparameters
    assert current.test_metrics == legacy.test_metrics
    assert current.validation_metrics == legacy.validation_metrics
    assert current.audit['protocol'] == 'honest_nested_outer_validation_v1'
    for name in current.test_predictions:
        np.testing.assert_array_equal(current.test_predictions[name], legacy.test_predictions[name])
    for name in current.validation_predictions:
        np.testing.assert_array_equal(current.validation_predictions[name], legacy.validation_predictions[name])


def test_neural_alias_deterministic_outputs_and_serialized_protocol():
    rng = np.random.default_rng(11)
    features = {name: rng.normal(size=(24, 4)) for name in ('resnet50', 'dinov2_vitb14')}
    y = features['resnet50'][:, 0] + 10
    groups = np.repeat(np.arange(12), 2)
    args = dict(task='regression', max_components=2, initialization_seeds=(101,),
                configurations={'tiny': dict(hidden=(4,), dropout=0.2, lr=0.001, weight_decay=0.0001)},
                max_epochs=1, random_state=20260818)
    current = neural.fit_inner_selected_neural_heads(features, y, groups, **args)
    legacy = legacy_neural.fit_honest_neural_heads(features, y, groups, **args)
    assert current.decisions == legacy.decisions
    assert current.audit() == legacy.audit()
    assert current.audit()['protocol'] == 'honest_inner_selected_neural_v1'
    for name in features:
        np.testing.assert_array_equal(current.predict(features)[name], legacy.predict(features)[name])
