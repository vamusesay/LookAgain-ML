# Advanced configuration

The [API reference](API_REFERENCE.md) enumerates all 51 `analyze` arguments,
defaults and validation rules. Populated `AdvancedConfig` sections override the
corresponding flat options; omitted sections leave flat options authoritative.
These typed settings exist in both the validated wheel and current source;
the source-only nested and inner-neural routines have separate signatures.

## Typed objects

### LinearHeadConfig

```python
working_precision: Precision = 'float64'
pca_component_cap: int | None = None
pca_iterated_power: int = 3
ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
logistic_cs: tuple[float, ...] = DEFAULT_LOGISTIC_CS
logistic_max_iter: int = 2000
```

### RandomForestConfig

```python
structured_configurations: tuple[dict[str, Any], ...] = field(default_factory=lambda: tuple((dict(value) for value in DEFAULT_RF_CONFIGURATIONS)))
score_configurations: tuple[dict[str, Any], ...] = field(default_factory=lambda: tuple((dict(value) for value in DEFAULT_RF_CONFIGURATIONS)))
seeds: tuple[int, ...] = DEFAULT_RF_SEEDS
n_estimators: int = 500
score_folds: int = 5
n_jobs: int | None = -1
```

### RFScoreConfig

```python
image_max_components: int = 32
reduction_random_state: int | None = None
group_fold_random_state: int | None = None
standardize_reduced_scores: bool = False
refit_image_reduction_on_train_validation: bool = True
working_precision: Precision = 'float64'
```

### AnalysisRuntimeConfig

```python
encoders: tuple[str, ...] | None = None
structured_models: tuple[str, ...] | None = None
integration_methods: tuple[str, ...] | None = None
device: str = 'auto'
batch_size: int = 32
```

### ExecutionConfig

```python
profile: ExecutionProfile = 'standard'
progress: Literal['auto'] | bool = 'auto'
progress_callback: Callable[[Any], None] | None = field(default=None, repr=False)
heartbeat_interval_seconds: float = 30.0
checkpoint_dir: str | Path | None = None
resume: bool = True
run_id: str | None = None
protocol_identifier: str | None = None
protocol_fingerprint: str | None = None
completed_run_policy: CompletedRunPolicy = 'reuse'
persistent_event_interval_seconds: float = 30.0
stale_lock_timeout_seconds: float | None = None
```

### AdvancedConfig

```python
linear_head: LinearHeadConfig | None = None
random_forest: RandomForestConfig | None = None
rf_score: RFScoreConfig | None = None
runtime: AnalysisRuntimeConfig | None = None
execution: ExecutionConfig | None = None
```

Default constant definitions and factory expressions above refer to
`lookagain_ml.configuration`, `heads` and `structured`; use the typed class to
construct them rather than copying unresolved names into an example.

## Encoders and information boundaries

The ten registered encoders are `resnet50`, `vgg16`, `inception_v3`,
`mobilenet_v2`, `coco_deeplab`, `ade20k_segformer`, `dinov2_vitb14`,
`siglip2_b16`, `convnext_b` and `vit_b16`. Inspect metadata without constructing
a model through `lookagain_ml.encoders.encoder_metadata(name)` or enumerate with
`list_encoders()`. Checkpoint revisions, pooling and preprocessing are recorded
in `environment/encoder_checkpoints.csv` and the encoder registry.

Groups are joined transitively with exact-image-hash duplicates for `analyze`.
Observation identifiers must be unique, nonblank and aligned. Supplied splits
are checked for group/hash leakage. `strict=True` raises on an image failure;
`strict=False` drops failed rows with warnings and an audit, not silent retention.
`verify_images=False` does not bypass file hashing or later encoder decoding.

## Runtime, cache and recovery

`device` supports auto/CPU/MPS/CUDA selectors, subject to backend availability;
`batch_size` controls encoder inference batches. Embedding cache identities
include row/image order, checkpoint, preprocessing, dimensions and runtime;
payload hashes are verified. Model-checkpoint storage is a different cache.
Verified precomputed features require aligned finite float32 arrays and
companion audit digests. A cache mismatch must not be bypassed.
`ExecutionConfig` controls progress callbacks, heartbeats, checkpointing,
resume and completed-run reuse/recompute. Scientific choices and execution
recovery controls are distinct. Resume does not prove a prospective protocol
lock or one-time historical test access.

## Fitting and uncertainty

Seed 20260818 is the general default; repeated splits use base+1009*(index+1),
and `analyze` bootstrap uses base+40. Supplied split labels cannot be combined
with multiple generated splits. `repeated_splits=1` is the default;
`bootstrap_repetitions=0` disables bootstrap, otherwise at least 100 is required.
Bootstrap jointly resamples fixed test predictions by group; it does not refit
models. Twelve split refits and 2,000 bootstrap draws represent different
uncertainty summaries.

Heads use optional training-fitted PCA, explicit ridge/logistic grids,
float32/float64 working precision and solver budgets. `analyze` defaults to no
PCA cap, power 3 and 2,000 logistic iterations. The paper's current Flowers
nested caller explicitly sets cap 128, three inner folds, float64, logistic
grid (0.03,0.1,0.3,1,3), max_iter 2000 and stack C 0.1. Current primary drivers
and neural drivers provide their own task-specific configurations. Do not
substitute general defaults for those controls.

General `analyze` combinations fit using validation data. Current nested
`linear_stack` is a cross-fitted stack using training OOF predictions and a
locked OOF refit. The method token is unchanged for saved-output compatibility.
Regression selection in the nested API may be MSE or MAE; classification uses
log loss. Selection never guarantees test improvement.

## Saved outputs

Use `result.summary()`, `comparison()`, `cost_summary()`, `qc`, `audit` and
`configuration` for inspection. Save/export defaults may include row-level
predictions and manifests. See the API reference and data-access documentation
before exporting. Neither repository contains restricted row-level derivatives.
