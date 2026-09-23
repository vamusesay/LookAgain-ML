# Current source API reference

Install the current integrated source with `python -m pip install .`.
The source identity ledger distinguishes this interface from the unchanged
1.0.0 wheel. See [reproducibility routes](REPRODUCIBILITY_ROUTES.md).

## General image-analysis API

```python
analyze(images: Iterable[str], y: Iterable[Any], groups: Iterable[Any] | None=None, *, observation_ids: Iterable[Any] | None=None, covariates: Any | None=None, split_labels: Iterable[Any] | None=None, task: str='auto', encoder: str='resnet50', encoders: Iterable[str] | None=None, device: str='auto', batch_size: int=32, pretrained: bool=True, cache: bool=True, cache_dir: str | Path='.lookagain_cache', model_cache_dir: str | Path | None=None, strict: bool=True, verify_images: bool=True, random_state: int=20260818, ridge_alphas: Iterable[float]=DEFAULT_RIDGE_ALPHAS, logistic_cs: Iterable[float]=DEFAULT_LOGISTIC_CS, head_max_components: int | None=None, head_pca_iterated_power: int=3, logistic_max_iter: int=2000, head_working_dtype: str='float64', combinations: Iterable[str] | None=None, repeated_splits: int=1, bootstrap_repetitions: int=0, bootstrap_metrics: Iterable[str] | None=None, logme: bool=False, logme_max_components: int=128, stack_alpha: float=10.0, stack_c: float=0.1, feature_concatenation_max_components: int=128, structured_models: Iterable[str] | None=None, integration_methods: Iterable[str] | None=None, rf_score_encoder: str | None=None, rf_structured_configurations: Iterable[dict[str, Any]] | None=None, rf_score_configurations: Iterable[dict[str, Any]] | None=None, rf_seeds: Iterable[int]=DEFAULT_RF_SEEDS, rf_n_estimators: int=500, rf_score_folds: int=5, rf_score_image_max_components: int=32, covariates_preprocessed: bool=False, numeric_missing_indicators: bool=True, score_integration_alpha: float=0.0, score_integration_c: float=0.1, joint_image_max_components: int=128, nn_max_iter: int=300, precomputed_features: Mapping[str, Any] | None=None, precomputed_feature_audits: Mapping[str, Mapping[str, Any]] | None=None, advanced_config: AdvancedConfig | None=None) -> LookAgainResults
```

The decorated public function preserves this signature. Regression and
binary/multiclass classification are supported; explicitly request regression
for numerical age labels. `auto` infers target type. Heads select by validation
MSE for regression or log loss for classification. `analyze` has no MAE-selection
keyword and no `inner_folds` or `output_dir` keyword. Its combination methods are
`equal_average`, `linear_stack`, `greedy_average` and `feature_concatenation`.
The validation-fitted `linear_stack` here differs from the nested routine below.

Defaults shown as constants are: `DEFAULT_RIDGE_ALPHAS=(0.1,1,10,100,1000,10000)`,
`DEFAULT_LOGISTIC_CS=(0.03,0.1,0.3,1,3)` and
`DEFAULT_RF_SEEDS=(2026082601,2026082602,2026082603)`.

| Argument | Accepted forms | Default | Required? | Behavior and validation | Output |
|---|---|---|---|---|---|
| images | Iterable[str] | REQUIRED | required | Aligned iterable of local image paths; scalar string rejected; at least one image and matching outcomes. Paths/readability/SHA checked; retained row order enters manifest. | manifest, audit, representations |
| y | Iterable[Any] | REQUIRED | required | One scalar outcome per image; regression numeric finite; classification comparable scalar labels with at least two classes. | classes, metrics, predictions |
| groups | Iterable[Any] \| None | None | optional | One nonmissing scalar group per image; absent means per-row user groups; exact-hash groups are still unioned transitively. Splits reject crossing groups/hashes. | manifest.effective_group_id, qc |
| observation_ids | Iterable[Any] \| None | None | optional | One unique nonblank stable identifier per requested image; generated row identifiers when absent; required matching digests for external features. | manifest, predictions, cache identity |
| covariates | Any \| None | None | optional | Optional aligned DataFrame or numeric matrix; enables generic structured/integration route, with train-fitted preprocessing. Not used in recorded image-only paper runs. | covariate_results, integration_results |
| split_labels | Iterable[Any] \| None | None | optional | One label per requested row; normalized to train/validation/test; all partitions and group/hash safety checked. Cannot combine with repeated_splits > 1. | manifest.split, qc |
| task | str | 'auto' | optional | auto/regression/classification, stripped/lowercased; auto uses sklearn target type; unsupported/multilabel targets rejected. | task, classes |
| encoder | str | 'resnet50' | optional | Single registered name; default resnet50 is ignored when encoders is supplied; a nondefault encoder plus encoders is rejected. | encoder_results |
| encoders | Iterable[str] \| None | None | optional | Nonempty iterable of distinct registered names; a plain string, duplicates or unknown names rejected. Order breaks equal-loss ties. | comparison, selected_encoder |
| device | str | 'auto' | optional | auto/cpu/mps/cuda/cuda:N; device availability checked at encoder creation; typed runtime config may override. | costs, configuration |
| batch_size | int | 32 | optional | Positive non-boolean integer; encoding batch size; typed runtime may override. | costs, configuration |
| pretrained | bool | True | optional | Boolean; false is infrastructure-only random weights and disables reusable embedding cache; incompatible with provided verified features. | cache_metadata, configuration |
| cache | bool | True | optional | Boolean requested embedding reuse; identities and payloads verified; checkpointing may retain an embedding store even when ordinary cache is disabled. | cache_metadata |
| cache_dir | str \| Path | '.lookagain_cache' | optional | Path/string to embedding-cache root; cache key covers rows/images/checkpoint/preprocessing/dimension/runtime/schema. | cache metadata and on-disk cache |
| model_cache_dir | str \| Path \| None | None | optional | Optional path/string for public model checkpoints; not the same as embedding cache. | encoder loading, configuration |
| strict | bool | True | optional | Boolean; true raises on any failed image. False omits failed rows with warnings and full audit; it does NOT retain corrupt images in the analytical sample. | audit requested/successful/failed rows |
| verify_images | bool | True | optional | Boolean; enables PIL verification/decoding during manifest construction. False does not bypass file hashing/path checks or encoder decoding. | audit, image_sha256 |
| random_state | int | 20260818 | optional | Non-boolean integer; generated split, PCA/tuning and repetitions. Repeated seed=base+1009*(index+1); analyze bootstrap seed=base+40. | configuration, repeated_split_results |
| ridge_alphas | Iterable[float] | DEFAULT_RIDGE_ALPHAS | optional | Iterable of positive finite candidate ridge penalties; validation-tuned head grid; typed linear_head can override. | selected_hyperparameters, encoder_results |
| logistic_cs | Iterable[float] | DEFAULT_LOGISTIC_CS | optional | Iterable of positive finite inverse regularization strengths; validation log-loss head grid; typed linear_head can override. | selected_hyperparameters, encoder_results |
| head_max_components | int \| None | None | optional | None or positive integer PCA cap for common heads; training-only preprocessing; typed linear_head can override. None preserves native dimensions. | head audits, selected_hyperparameters |
| head_pca_iterated_power | int | 3 | optional | Positive integer randomized PCA iteration count; typed linear_head can override. | head audit/configuration |
| logistic_max_iter | int | 2000 | optional | Positive integer solver budget; convergence warning is not evidence of a different automatically enlarged budget. | configuration, head audit |
| head_working_dtype | str | 'float64' | optional | float32 or float64; typed linear_head can override; does not change the requirement that provided representations are float32. | configuration, head audit |
| combinations | Iterable[str] \| None | None | optional | None/empty disables combinations; iterable from equal_average/linear_stack/greedy_average/feature_concatenation. Unknown methods rejected. | stack_results, method_predictions |
| repeated_splits | int | 1 | optional | Positive integer; >1 runs complete alternative group-safe downstream refits and requires split_labels=None. Representations can be reused. | repeated_split_results |
| bootstrap_repetitions | int | 0 | optional | Integer 0 disables; otherwise integer >=100. Fixed predictions resampled jointly by effective group, no model retraining. | uncertainty |
| bootstrap_metrics | Iterable[str] \| None | None | optional | None chooses r2/mse/mae or accuracy/balanced_accuracy/macro_f1/log_loss; otherwise iterable of task-valid metric names, not a plain string. | uncertainty |
| logme | bool | False | optional | Boolean; optional training-only LogME screening, not a replacement selection rule. | logme_results |
| logme_max_components | int | 128 | optional | Positive integer screening PCA cap. | logme_results |
| stack_alpha | float | 10.0 | optional | Positive finite regression stack ridge penalty; applies to validation-fitted analyze stack, not later nested API by implication. | stack_results |
| stack_c | float | 0.1 | optional | Positive finite classification stack inverse penalty. | stack_results |
| feature_concatenation_max_components | int | 128 | optional | Positive integer per-encoder training PCA cap for concatenated feature blocks. | stack_results feature union |
| structured_models | Iterable[str] \| None | None | optional | With covariates, iterable of linear/nn/rf model names; absent uses the structured default; rejected if supplied without covariates. | covariate_results |
| integration_methods | Iterable[str] \| None | None | optional | With covariates, linear_score/nn_score/rf_score/joint_linear/joint_nn; linear_score default; rf_score regression-only. | integration_results |
| rf_score_encoder | str \| None | None | optional | Registered selected encoder name among requested encoders, or None to use validation-selected single; generic RF-score route only. | integration_results, configuration |
| rf_structured_configurations | Iterable[dict[str, Any]] \| None | None | optional | Nonempty iterable of RF configuration dictionaries; None selects DEFAULT_RF_CONFIGURATIONS; typed random_forest can override. | covariate_results |
| rf_score_configurations | Iterable[dict[str, Any]] \| None | None | optional | Nonempty RF grid for score integration; None uses default grid; typed random_forest can override. | integration_results |
| rf_seeds | Iterable[int] | DEFAULT_RF_SEEDS | optional | Nonempty iterable of non-boolean integer seeds; generic forest ensemble; typed random_forest can override. | covariate/integration audits |
| rf_n_estimators | int | 500 | optional | Positive integer trees per forest; typed random_forest can override. | covariate/integration audits |
| rf_score_folds | int | 5 | optional | Positive integer at public option gate; score fitting additionally requires feasible group folds; typed random_forest can override. | OOF score audit |
| rf_score_image_max_components | int | 32 | optional | Positive integer image-score PCA cap; typed rf_score can override. | OOF score audit |
| covariates_preprocessed | bool | False | optional | Boolean declaration for supplied structured matrix; does not license test-informed preprocessing. | covariate audit |
| numeric_missing_indicators | bool | True | optional | Boolean controls numeric missingness indicators in train-fitted structured preprocessing. | covariate audit |
| score_integration_alpha | float | 0.0 | optional | Finite nonnegative regression linear-score regularization. | integration_results |
| score_integration_c | float | 0.1 | optional | Positive finite classification linear-score inverse penalty. | integration_results |
| joint_image_max_components | int | 128 | optional | Positive integer image PCA cap for joint X+I models. | integration_results |
| nn_max_iter | int | 300 | optional | Positive integer for generic structured/joint NN estimators; not the paper GELU max_epochs control. | covariate/integration results |
| precomputed_features | Mapping[str, Any] \| None | None | optional | Mapping keyed exactly by requested encoders, paired with audits; finite float32 arrays of exact row/feature shape; pretrained=True required. | cache status provided-verified; no fresh inference |
| precomputed_feature_audits | Mapping[str, Mapping[str, Any]] \| None | None | optional | Companion mapping: ordered-ID/image hashes, feature-value digest, encoder metadata, source array and metadata digests must match; no silent row realignment. | verification audits |
| advanced_config | AdvancedConfig \| None | None | optional | AdvancedConfig instance or None; populated typed sections override corresponding flat options. Not an unrestricted kwargs bag. | configuration.analysis.advanced_config, progress/checkpoints |

## Results and output controls

`summary()`, `comparison()` and `cost_summary()` return DataFrames. `qc`,
`audit`, `configuration`, `cache_metadata`, `repeated_split_results`,
`uncertainty` and `logme_results` are attributes, not methods.
`save(directory, include_predictions=True, include_manifest=True)` and
`export(directory, **kwargs)` serialize results. Saving defaults can include
row-level information; saving a result does not authorize its publication.
For an aggregate-only export, explicitly disable predictions and the manifest
and review the remaining content under the applicable data terms.

## Current nested validation selection

The canonical module is `lookagain_ml.validation_selection`, also exported
from `lookagain_ml`. Input feature arrays, outcomes, groups and split labels
must share row order. All outer partitions must be nonempty and groups must not
cross them. Classification labels are zero-based integer codes, and every inner
training fold must contain all classes. Construct/verify image-hash-safe groups
before supplying features; this array API does not decode images.

Base-head tuning uses group-safe folds within outer training; candidate stacks
fit on training OOF predictions. Validation chooses among frozen candidates;
locked decisions are refit on train+validation, with new group-safe OOF
predictions for the final stack. Feature-union tuning uses an inner group
holdout. Test outcomes are used for assessment only.

```python
from lookagain_ml.validation_selection import nested_validation_selection
nested_validation_selection(features: dict[str, np.ndarray], outcome: np.ndarray, groups: np.ndarray, split_labels: np.ndarray, *, task: str, encoder_order: Iterable[str], methods: Iterable[str]=('equal_average', 'linear_stack', 'greedy_average', 'feature_concatenation'), regression_selection_metric: str='mse', inner_folds: int=5, random_state: int=20260818, ridge_alphas: Iterable[float]=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0), logistic_cs: Iterable[float]=(0.03, 0.1, 0.3, 1.0, 3.0), stack_alpha: float=10.0, stack_c: float=0.1, head_max_components: int | None=256, feature_concatenation_max_components: int=128, pca_iterated_power: int=3, logistic_max_iter: int=2000, working_dtype: str='float64')
```

```python
from lookagain_ml.inner_selected_neural_heads import fit_inner_selected_neural_heads
fit_inner_selected_neural_heads(features, outcomes, groups, *, task, regression_selection_metric='mse', random_state=20260818, inner_folds=3, max_components=128, configurations=PAPER_NEURAL_CONFIGS, initialization_seeds=PAPER_NEURAL_SEEDS, max_epochs=70, patience=7, batch_size=1024)
```

```python
from lookagain_ml.inner_selected_neural_heads import refit_locked_neural_heads
refit_locked_neural_heads(features, outcomes, decisions, *, fit_partition)
```

`NestedValidationSelectionResult` exposes selected encoder/preferred method,
selection metric, validation/test metrics and predictions, selected parameters
and the audit. `InnerSelectedNeuralHeads` exposes `predict`, `predict_members`,
`transform`, `decisions` and `audit()`. Inner neural architecture/epoch selection
uses only outer training, followed by locked refits; no test outcomes are
accepted by that fitting API. Five seed defaults are 101, 202, 303, 404 and 505.
Default nested folds are 5 and head cap 256; paper runners explicitly use 3 folds
and cap 128. General defaults alone do not reproduce an experiment protocol.

Other established APIs include typed configuration, encoder registry utilities,
`linear_cka`, `fit_group_cross_fitted_regression_stack`, `fit_paper_neural_heads`,
`paired_locked_test_bootstrap` (in `lookagain_ml.uncertainty`), and generic
structured-covariate utilities. The separate covariate application is not
included. [Legacy compatibility names](REPRODUCIBILITY_ROUTES.md#legacy-compatibility)
remain available only to preserve existing source imports.
