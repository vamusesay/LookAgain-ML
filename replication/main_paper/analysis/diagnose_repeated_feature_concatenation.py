"""Reconstruct one repeated-split feature union and report aggregate numerical ranges."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--split-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replication_root = args.package_root / "replication"
    workflow_root = replication_root / "representation_risk"
    sys.path.insert(0, str(replication_root))
    sys.path.insert(0, str(workflow_root))

    from _common import load_config, load_manifest
    from _reference_features import known_reference_specifications, load_reference_features
    from lookagain_ml.combination import fit_feature_concatenation
    from lookagain_ml.heads import fitted_linear_head_audit

    config = load_config(args.config)
    spec = next(item for item in config["datasets"] if item["name"] == args.dataset)
    manifest = load_manifest(Path(spec["manifest"]).expanduser().resolve())
    encoders = list(spec.get("encoders", config["encoders"]))
    references = known_reference_specifications(
        str(spec["reference_embedding_layout"]),
        str(config["research_root"]),
        dataset=spec.get("reference_dataset"),
    )
    features, _, verification = load_reference_features(
        manifest, {name: references[name] for name in encoders}
    )
    assignments = pd.read_csv(args.assignments)
    assignments = assignments.loc[assignments.split_index == args.split_index].copy()
    merged = manifest[["observation_id", "outcome"]].merge(
        assignments[["observation_id", "seed", "split"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise RuntimeError("Repeated assignment does not cover the manifest.")
    train = merged.split.eq("train").to_numpy()
    validation = merged.split.eq("validation").to_numpy()
    test = merged.split.eq("test").to_numpy()
    seed = int(merged.seed.iloc[0])
    y = merged.outcome.to_numpy(float)
    fitted, audit = fit_feature_concatenation(
        str(spec["task"]),
        {name: values[train] for name, values in features.items()},
        y[train],
        {name: values[validation] for name, values in features.items()},
        y[validation],
        encoders,
        max_components_per_encoder=128,
        ridge_alphas=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0),
        logistic_cs=(0.03, 0.1, 0.3, 1.0, 3.0),
        random_state=seed,
    )

    partitions = {"train": train, "validation": validation, "test": test}
    report: dict[str, object] = {
        "dataset": args.dataset,
        "split_index": args.split_index,
        "seed": seed,
        "reference_verification_statuses": {
            name: values["status"] for name, values in verification.items()
        },
        "head_validation_grid": audit["head_validation_grid"],
        "head_audit": fitted_linear_head_audit(fitted.head),
        "partitions": {},
    }
    for partition, mask in partitions.items():
        blocks = {
            name: fitted.transforms[name].transform(features[name][mask]) for name in encoders
        }
        union = np.hstack([blocks[name] for name in encoders])
        scaled = fitted.head.pipeline.named_steps["scale"].transform(union)
        prediction = fitted.predict({name: values[mask] for name, values in features.items()})
        report["partitions"][partition] = {
            "rows": int(mask.sum()),
            "maximum_absolute_block_value": {
                name: float(np.max(np.abs(block))) for name, block in blocks.items()
            },
            "maximum_absolute_union_value": float(np.max(np.abs(union))),
            "maximum_absolute_scaled_union_value": float(np.max(np.abs(scaled))),
            "scaled_union_99_99_percentile": float(np.quantile(np.abs(scaled), 0.9999)),
            "maximum_absolute_prediction": float(np.max(np.abs(prediction))),
            "prediction_99_9_percentile": float(np.quantile(np.abs(prediction), 0.999)),
            "prediction_finite": bool(np.isfinite(prediction).all()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
