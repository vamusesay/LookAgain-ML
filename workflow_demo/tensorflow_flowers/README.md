# LookAgain-ML Public TensorFlow Flowers Demonstration

This workflow uses all ten encoders: ResNet50, VGG16, InceptionV3, MobileNetV2, COCO DeepLab, ADE20K SegFormer, DINOv2-B/14, ConvNeXt-B, ViT-B/16, and SigLIP 2-B/16.

The official archive contains 3,670 JPEG images. Excluding two cross-class duplicate rows leaves 3,668 observations in 3,666 duplicate-safe groups. The locked split contains 2,568 training, 550 validation, and 550 test observations. Preparation verifies the archive checksum, safely extracts it, decodes and hashes every image, excludes cross-class conflicts, groups within-class duplicates, and verifies the manifest and locked-split checksums.

The extraction script uses pinned encoder metadata and verifies image bytes and ordered identifiers. It saves one finite array per encoder and reuses only matching, hash-verified caches. It fits no prediction model. CPU and CUDA are supported; reduce the batch size if device memory is limited. An interruption can be resumed with the same verified extraction cache. An interrupted model-fit run restarts its deterministic fitting procedure; do not infer a completed run from partial outputs.

The fitting script constructs three-fold group-safe out-of-fold predictions inside training. It tunes base heads using training data, fits the all-ten stack on out-of-fold predictions and training outcomes, and compares frozen candidates on the separate validation partition. It locks decisions, refits on training plus validation, then evaluates test predictions. Twelve group-safe splits fixed in the implemented workflow assess partition sensitivity. Paired uncertainty uses 2,000 group-bootstrap draws from fixed predictions, without model refitting.

The current reference results are SigLIP 2 accuracy 0.9836 and log loss 0.0597; the all-ten out-of-fold stack has accuracy 0.9836 and log loss 0.0777. The validation-selected primary procedure is the single encoder. Equal accuracy does not imply equal probabilistic performance. Test differences are descriptive comparisons of locked fits, not an additional selection step.

Both notebooks include installation, checksums, preparation, extraction, cache reuse, fitting, aggregate validation, artifact generation, and interpretation. Full model execution is controlled by `RUN_FULL`; it is disabled initially. No Colab GPU run is claimed. Hardware-specific bitwise reproduction of representations must be checked against the recorded checkpoint and environment identities.

## Validation and data access

The delivered Colab and Jupyter notebooks have static JSON and Python-syntax validation. Earlier successful executions used related workflows. The exact delivered notebooks were not executed end to end during this packaging stage; full GPU execution is not claimed.

Saved-result verification covers retained predictions and selected numerical metrics, displayed result rows, cost rows and scalar definitions. Package tests, compilation and privacy checks are separate checks, not one universal artifact-reload certificate.

Cache hashes and verification records are retained alongside saved metrics and predictions. Cache availability must be checked before attempting a reproduction; a regenerated array must not be represented as the original archived array.

Both repository releases exclude restricted images, row-level metadata, private filesystem paths, predictions, embeddings, fitted models and representation arrays. Obtain restricted datasets independently under their applicable access terms. A private repository does not itself authorize disclosure, and removing photographs alone does not establish anonymity of derivatives. The code license does not grant rights to datasets. The TensorFlow Flowers workflow remains separate and retains its attribution and license information.

## Open and run the current workflow

From the repository root, install the current source with `python -m pip install .`
and its all-encoder extra with `python -m pip install ".[modern]"`. The current
runner imports `nested_validation_selection`; the validated wheel alone is not
the installation route for this workflow.

Open [the Jupyter notebook](lookagain_ml_flowers_jupyter.ipynb) locally from
within the extracted repository. For Colab, open
[the Colab notebook](lookagain_ml_flowers_colab.ipynb), upload the repository ZIP
to the runtime and follow the extraction cell. No personal cloud path or
identifying repository URL is required.

`RUN_FULL=False` initially limits the notebook to source/reference checks and
aggregate artifact generation. Turning it on authorizes the user's own fresh
download, extraction and fitting run. Use a new `work/flowers` root for initial
preparation; preparation intentionally refuses to overwrite an existing staged
archive. Resume extraction against the same verified `work/features` cache;
restart an interrupted fitting stage with the same inputs/settings. Do not
rerun the preparation cell on an already staged root.

`work/flowers` stores public inputs and splits, `work/features` representations,
`work/models` checkpoints and `work/results` new fitted outputs. These local
directories are ignored by Git. `reference_results/` contains saved V2 aggregate
reference outputs, not outputs from the delivered notebook. `analysis/` retains
the authoritative current script filenames; `figures/`, `tables/` and `scalars/`
contain the approved saved presentation artifacts. Generators operate on saved
references; assess fresh fitted results separately before replacing references.

Use [the local checksum ledger](checksums.sha256) and the root verification
script to verify shipped bytes. The final notebooks contain no execution
outputs. **Exact-final-notebook GPU execution was not performed during final
packaging.** See [data licence and attribution](DATA_LICENSE_AND_ATTRIBUTION.md).
