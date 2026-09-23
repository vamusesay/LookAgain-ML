# LookAgain-ML

LookAgain-ML is a PyTorch-based image-representation analysis library.
Distribution: **lookagain-ml**. Python import: **lookagain_ml**. The validated
wheel is **1.0.0**; this review repository also supplies later integrated source
functionality under the unchanged version metadata. Python **3.10–3.13** is
supported by the declared package bounds.

The library supports pretrained image-representation extraction and comparison,
regression and classification, group-aware splitting, validation-based selection,
representation combinations, repeated-split stability, paired uncertainty,
caching, image verification and quality-control outputs. The current nested API
fits stacks from group-safe out-of-fold training predictions. Validation-based
selection does not guarantee improved test performance.

## Installation

The primary reviewer route installs the current integrated source:

```bash
python -m pip install .
```

Use this for the current API, nested validation-selection, cross-fitted stacking,
public Flowers workflow and paper-replication scripts. For all ten encoders,
also use `python -m pip install ".[modern]"`.

The separate validated-wheel route is:

```bash
python -m pip install release_assets/lookagain_ml-1.0.0-py3-none-any.whl
```

The unchanged wheel provides its verified 1.0.0 API and was used in documented
Flowers runs. It does not expose the entire later integrated API. Use a separate
environment for this route. Neither route obtains LookAgain-ML from PyPI as part
of this review delivery. See [installation](release_assets/INSTALL.md) for exact
identity, dependencies and import checks.

**Source/wheel relationship: DOCUMENTED SCOPE DIFFERENCE.**

## General API example

Supply an aligned image-only analysis manifest you are authorized to use, with
columns `image_path`, `outcome`, `group_id` and `observation_id`. The example
performs extraction and model fitting when executed; it is not a saved result.

```python
import pandas as pd
import lookagain_ml

frame = pd.read_csv("analysis_manifest.csv")
result = lookagain_ml.analyze(
    images=frame["image_path"].tolist(),
    y=frame["outcome"].tolist(),
    groups=frame["group_id"].tolist(),
    observation_ids=frame["observation_id"].tolist(),
    task="regression",
    encoders=("resnet50",),
    combinations=(),
    cache=True,
    strict=True,
    verify_images=True,
)
print(result.summary())
print(result.comparison())
print(result.qc)
print(result.audit)
result.save("work/example_results", include_predictions=False, include_manifest=False)
```

`analyze()` is the general library workflow; its `linear_stack` combines
validation predictions and outcomes. It is not a shortcut to the paper's nested
procedure. The canonical current-source function is
`lookagain_ml.nested_validation_selection`, implemented in
`lookagain_ml.validation_selection`. It tunes within training, compares frozen
candidates on validation, and refits locked decisions before test-set-isolated
evaluation. See the [API reference](docs/API_REFERENCE.md) and
[advanced configuration](docs/ADVANCED_CONFIGURATION.md) for exact signatures,
defaults, inputs, PCA, stacking, validation, seed, cache, device, repetition,
uncertainty and output controls.

## Paper replication and public workflow

The wheel supplied verified package functionality in documented runs. The paper’s
nested-selection and neural-head analyses used later source modules. Separate
scripts implement adaptation, diagnostics, aggregation and manuscript-artifact
creation. Saved, hash-verified outputs support the reported results; the wheel
alone did not generate every current table and figure. Current reviewer
reproduction uses the supplied source and replication scripts.

Start with the [TensorFlow Flowers Tutorial](workflow_demo/tensorflow_flowers/README.md).
It is the current-API public demonstration and requires no restricted dataset.
Its saved reference outputs describe the procedure used in the submitted paper.

| Directory | Contents |
|---|---|
| `src/` | Current integrated package and narrow legacy import aliases |
| `tests/` | Package unit tests and deterministic compatibility checks |
| `docs/` | API, advanced configuration, installation and reproducibility routes |
| `workflow_demo/` | Public Flowers notebooks, scripts and saved references |
| `release_assets/` | Unchanged validated wheel, checksum and route instructions |
| `replication/` | Paper drivers, saved aggregates and artifact generators |
| `environment/` | Dependency/checkpoint records and current source hashes |

The [reproducibility routes](docs/REPRODUCIBILITY_ROUTES.md) distinguish package
APIs, paper-specific analysis and aggregate-only artifact creation. The
[paper map](replication/main_paper/paper_map.md) locates the inputs behind each
manuscript artifact. The `figures/`, `tables/` and `scalars/` folders under
`replication/main_paper/` contain the approved saved artifacts.

## Validation and limitations

Run package tests with `python -m pytest tests` after installing the test extra
(`python -m pip install ".[test]"`). Final assembly validation uses the complete
allowed CPU suite with synthetic fixtures; real encoder extraction, checkpoint
and GPU tests are skipped under the stated execution scope. Notebook validation
is static (schema, syntax, paths and source-route checks).
**Exact-final-notebook GPU/end-to-end execution was not performed during final packaging.**
Saved references are not fabricated notebook outputs or evidence of a newly run
experiment. Timings depend on device, environment and cache conditions; retained
cost records are descriptive, not a standardized hardware benchmark.

No restricted images, manifests, row-level data or predictions, embeddings,
fitted models or representation arrays are distributed in either repository.
Obtain restricted inputs independently under their applicable access terms.
Private repository visibility does not itself authorize data disclosure, and
removing photographs does not anonymize all derivatives. Fresh public-data
reproduction downloads the Flowers archive and public encoder checkpoints;
neither is bundled. See [data access](docs/data_access.md).

## Licence and citation

Project code uses the supplied [MIT licence](LICENSE). Dataset and model terms
are separate; see [Flowers attribution](workflow_demo/tensorflow_flowers/DATA_LICENSE_AND_ATTRIBUTION.md)
and `environment/encoder_checkpoints.csv`. Use the supplied [citation record](CITATION.cff);
authorship is withheld for anonymous review.
