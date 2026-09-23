# Main-paper replication

Generate tables and figures from released aggregates:

```sh
python replication/main_paper/analysis/generate_manuscript_tables.py
python replication/main_paper/analysis/generate_manuscript_figures.py
python replication/main_paper/analysis/generate_neural_assets.py
python replication/main_paper/analysis/generate_cost_table.py
```

These commands write only aggregate publication assets. They do not rerun models. Refer to `paper_map.md` for exact dependencies and unresolved scope differences.

For authorized-data replication, `analysis/run_nested_protocol.py --help` describes fixed and twelve repeated splits. The package root is this repository root; use `configurations/experiment.json`, supply the authorized Toronto cache, and select one dataset. Local source-layout mappings are documented by `_reference_features.py`. The neural-head and adaptation runners document their required local paths through environment variables. They have not been rerun as part of this delivery.

## Validation and data access

The delivered Colab and Jupyter notebooks have static JSON and Python-syntax validation. Earlier successful executions used related workflows. The exact delivered notebooks were not executed end to end during this packaging stage; full GPU execution is not claimed.

Saved-result verification covers retained predictions and selected numerical metrics, displayed result rows, cost rows and scalar definitions. Package tests, compilation and privacy checks are separate checks, not one universal artifact-reload certificate.

Cache hashes and verification records are retained alongside saved metrics and predictions. Cache availability must be checked before attempting a reproduction; a regenerated array must not be represented as the original archived array.

Both repository releases exclude restricted images, row-level metadata, private filesystem paths, predictions, embeddings, fitted models and representation arrays. Obtain restricted datasets independently under their applicable access terms. A private repository does not itself authorize disclosure, and removing photographs alone does not establish anonymity of derivatives. The code license does not grant rights to datasets. The TensorFlow Flowers workflow remains separate and retains its attribution and license information.

## Submitted figure copies

The delivered figure PDFs are synchronized with the approved manuscript
artifacts. The main-paper directory contains the seven assets for Figures 1–5;
the four public Flowers assets used in appendix Figures 6–9 remain under
`workflow_demo/tensorflow_flowers/figures/`. Numerical content was verified
against the corresponding saved-result renderings. Generating figures from the
unchanged aggregate inputs preserves the scientific values; PDF font metrics,
layout and metadata can vary with Matplotlib and the rendering environment.
The delivered approved PDFs were copied without regeneration.
