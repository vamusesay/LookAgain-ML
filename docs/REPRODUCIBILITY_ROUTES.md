# Reproducibility routes

Source/wheel relationship: DOCUMENTED SCOPE DIFFERENCE.

| Route | Code and installation | Scientific scope |
|---|---|---|
| Validated 1.0.0 wheel | Local wheel under `release_assets/` | General `analyze`, encoder/cache/QC APIs, typed configuration, repeated splits, uncertainty and specialized cross-fitted regression APIs |
| Current integrated API | `python -m pip install .` | `nested_validation_selection` and inner-selected neural heads in addition to the general API |
| Paper replication | Current source plus `replication/main_paper/analysis/` | Dataset-specific settings, nested selection, neural heads, adaptation and diagnostics |
| Public Flowers workflow | Current source plus `workflow_demo/tensorflow_flowers/` | Paper sample, three-fold inner training OOF tuning/stacking, 12 repeated splits and 2,000 paired draws |
| Manuscript artifacts | Saved aggregates plus supplied generators | Tables, figures and scalar formatting without scientific model fitting |

## Method boundaries

`validation_based_selection` compares candidate procedures on validation data.
The current `nested_validation_selection` routine tunes base heads inside
outer training and fits its `cross_fitted_stack` using `out_of_fold_predictions`.
It then compares frozen candidates on outer validation and refits locked
decisions on training plus validation before `test_set_isolated_evaluation`.
These descriptive terms are not additional keyword arguments. The existing
method token `linear_stack` is preserved in calls and serialized results.

The general `analyze(..., combinations=("linear_stack",))` route fits the stack
using validation predictions and outcomes, including in the unchanged wheel.
Its specialized group-cross-fitted regression function is also real wheel
functionality, but does not implement the entire later multi-encoder nested
selection procedure. Both routes exclude test outcomes from fitting and selection.

Recorded wheel Flowers runs used ten encoders, native-dimensional heads
(`head_max_components=None`), typed configurations, cache reuse, 12 splits and
2,000 bootstrap draws. The submitted Flowers analysis uses three inner folds and a PCA cap
of 128. These distinct procedures have distinct references; historical results
are retained only in private execution records.

The paper’s primary analyses used the later nested source module. Current neural
analyses used the later inner-selection/locked-refit module. Adaptation and some
diagnostics use separate scripts. Other scripts aggregate saved outputs and
create manuscript artifacts. A shared version string is not execution identity.
Package-root options in scientific drivers explicitly select the supplied
checkout; validate `environment/source_identity.json` before use.

## Aggregate-only verification and artifacts

From the repository root:

```bash
python replication/verify_checksums.py
python workflow_demo/tensorflow_flowers/analysis/verify_reference_results.py
python workflow_demo/tensorflow_flowers/analysis/generate_flowers_artifacts.py
```

The checksum verifier hashes UTF-8 text with LF-normalized line endings and
binary assets byte-for-byte, so Git checkouts on Windows and Unix agree.

The generator reads saved `reference_results/`; it does not use newly fitted
outputs automatically. Fresh Flowers results go to the notebook's `WORK/results`
and must be validated separately before any reference replacement. Paper
generators and their dependencies are documented in
[the paper map](../replication/main_paper/paper_map.md). Run generators on a working
copy: shipped artifact bytes and checksums describe the approved delivery.

## Legacy compatibility

Existing source imports of `honest_nested_selection` and
`HonestNestedSelectionResult` remain aliases for `nested_validation_selection`
and `NestedValidationSelectionResult`. The old module `honest_selection` is a
thin compatibility shim. Existing imports of `fit_honest_neural_heads` and
`HonestNeuralHeads` through `honest_neural_heads` likewise alias
`fit_inner_selected_neural_heads` and `InnerSelectedNeuralHeads` in
`inner_selected_neural_heads`. No statistical implementation is duplicated and
no runtime deprecation warning is emitted. These later source aliases are not
asserted to exist in the validated wheel.

The serialized protocol IDs `honest_nested_outer_validation_v1` and
`honest_inner_selected_neural_v1` are retained solely for existing saved-output
compatibility; they are not the canonical method descriptions.

## Execution and access limits

Exact-final-notebook GPU/end-to-end execution was not performed during final
packaging. CPU unit tests and static notebook checks do not certify a new
ten-encoder scientific run. Hardware-dependent timings, checkpoint access and
third-party terms remain relevant. Restricted datasets must be obtained
independently; no restricted observations or identifying derivatives are in
either repository. Public Flowers is the unrestricted-data demonstration.
