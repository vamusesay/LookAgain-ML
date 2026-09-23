# Installation routes

## Install the current integrated API

From the repository root, in a clean Python 3.10–3.13 environment:

```bash
python -m pip install .
```

Use this route for the current public workflow and current paper-replication API.
For all ten encoders, also install the local source with its optional adapters:

```bash
python -m pip install ".[modern]"
```

PyTorch and torchvision must match your CPU/CUDA platform. The package uses
NumPy, pandas, SciPy, scikit-learn, Pillow and Matplotlib; Transformers,
Hugging Face Hub and safetensors support the optional modern encoders.
Declared dependency bounds are in `pyproject.toml`; recorded environments are
under `environment/`. Python versions outside 3.10–3.13 are not declared supported.

## Install the validated LookAgain-ML 1.0.0 wheel

In a separate clean environment, from the repository root:

```bash
python -m pip install release_assets/lookagain_ml-1.0.0-py3-none-any.whl
```

The unchanged wheel is 131241 bytes. SHA-256:
`D295D1F8BBB4D8CB7EBE189920CC968270810BDC9847BB46C2A9BCBCD84E206D`.
Use this route for the verified LookAgain-ML 1.0.0 API. Recorded Flowers runs
used its `analyze()` API, ten encoders, typed configurations, cache reuse,
12 repeated splits and 2,000 bootstrap draws. It does not contain every module
later integrated into the current source and cannot alone run the current
nested-validation Flowers workflow.

Neither route installs LookAgain-ML from PyPI in this review delivery: the
package is installed from local source or the supplied local wheel. pip may
obtain third-party dependencies from its configured index. Fresh encoder use
may download public checkpoints; those downloads are separate from package
installation and are subject to upstream terms.

## Check the selected environment

```python
import lookagain_ml
print(lookagain_ml.__version__)
print(lookagain_ml.__file__)
# Current source route only:
from lookagain_ml import nested_validation_selection
```

Both source and wheel retain version 1.0.0; version alone is insufficient to
identify the implementation. `environment/source_identity.json` identifies the
current source files; `release_assets/checksums.sha256` identifies the wheel.
The paper drivers explicitly load the supplied checkout through
`--package-root/src`, so that checkout must match the source identity ledger.
Use separate environments to avoid confusing these two scopes.

Source/wheel relationship: DOCUMENTED SCOPE DIFFERENCE.
