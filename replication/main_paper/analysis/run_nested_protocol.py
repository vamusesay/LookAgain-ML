"""Run the six-task nested validation ICLR protocol from verified frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ENCODERS = [
    "resnet50",
    "vgg16",
    "inception_v3",
    "mobilenet_v2",
    "coco_deeplab",
    "ade20k_segformer",
    "dinov2_vitb14",
    "siglip2_b16",
    "convnext_b",
    "vit_b16",
]
METHODS = ["equal_average", "linear_stack", "greedy_average", "feature_concatenation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toronto-cache", type=Path, required=True)
    parser.add_argument("--installed-package-root", type=Path)
    parser.add_argument("--repetitions", type=int, default=12)
    return parser.parse_args()


def digest_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_toronto_features(frame: pd.DataFrame, cache: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    features: dict[str, np.ndarray] = {}
    verification: dict[str, object] = {}
    id_digest = digest_strings(frame["observation_id"].astype(str).tolist())
    image_digest = digest_strings(frame["image_sha256"].astype(str).str.lower().tolist())
    for name in ENCODERS:
        directory = cache / "toronto_houses" / "embeddings" / name
        arrays = list(directory.glob("*.npy"))
        records = list(directory.glob("*.json"))
        if len(arrays) != 1 or len(records) != 1:
            raise RuntimeError(f"{name}: expected exactly one cache array and metadata record")
        metadata = json.loads(records[0].read_text(encoding="utf-8"))
        identity = metadata["identity"]
        if identity["ordered_observation_id_sha256"] != id_digest:
            raise RuntimeError(f"{name}: Toronto observation-ID digest mismatch")
        if identity["ordered_image_hash_sha256"] != image_digest:
            raise RuntimeError(f"{name}: Toronto image digest mismatch")
        if hashlib.sha256(arrays[0].read_bytes()).hexdigest().lower() != metadata["array_sha256"].lower():
            raise RuntimeError(f"{name}: cached array digest mismatch")
        array = np.load(arrays[0], mmap_mode="r", allow_pickle=False)
        if len(array) != len(frame) or not np.isfinite(array).all():
            raise RuntimeError(f"{name}: Toronto cache is not finite and row aligned")
        features[name] = array
        verification[name] = {
            "status": "PASS",
            "array": str(arrays[0]),
            "array_sha256": metadata["array_sha256"],
            "rows": len(array),
            "dimension": int(array.shape[1]),
        }
    return features, verification


def metric_rows(dataset: str, scope: str, split_index: int | None, result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, metrics in result.test_metrics.items():
        row: dict[str, object] = {
            "dataset": dataset,
            "scope": scope,
            "split_index": split_index,
            "method": method,
            "method_type": "encoder" if method in ENCODERS else "combination",
            "selected_single": result.selected_encoder,
            "preferred_method": result.preferred_method,
            "validation_selected_single": method == result.selected_encoder,
            "validation_preferred_method": method == result.preferred_method,
            "selection_metric": result.selection_metric,
            "validation_loss": float(result.validation_metrics[method][result.selection_metric]) if method in result.validation_metrics else None,
        }
        row.update({f"test_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return rows


def prediction_frame(frame: pd.DataFrame, labels: np.ndarray, result, split_index: int | None) -> pd.DataFrame:
    test = labels == "test"
    out = pd.DataFrame(
        {
            "split_index": split_index,
            "observation_id": frame.loc[test, "observation_id"].astype(str).to_numpy(),
            "group_id": frame.loc[test, "group_id"].astype(str).to_numpy(),
            "outcome": frame.loc[test, "outcome"].to_numpy(),
            "selected_single": result.selected_encoder,
            "preferred_method": result.preferred_method,
        }
    )
    for name, prediction in result.test_predictions.items():
        if prediction.ndim == 1:
            out[f"pred_{name}"] = prediction
        else:
            for class_index in range(prediction.shape[1]):
                out[f"pred_{name}__class_{class_index}"] = prediction[:, class_index]
    return out


def main() -> None:
    args = parse_args()
    replication_root = args.package_root / "replication"
    sys.path.insert(0, str(args.package_root / "src"))
    if args.installed_package_root is not None:
        sys.path.insert(0, str(args.installed_package_root))
    sys.path.insert(0, str(replication_root))
    from _common import load_config, load_manifest
    from _reference_features import known_reference_specifications, load_reference_features
    from lookagain_ml import nested_validation_selection
    from lookagain_ml.splitting import make_split, validate_partition_safety

    config = load_config(args.config)
    specs = [item for item in config["datasets"] if item["name"] == args.dataset]
    if len(specs) != 1:
        raise ValueError(f"Expected one final-task specification for {args.dataset}")
    spec = specs[0]
    frame = load_manifest(Path(spec["manifest"]).expanduser().resolve())
    if len(frame) != int(spec["expected_n"]):
        raise ValueError("Prepared-manifest row count changed")
    task = str(spec["task"])
    if args.dataset == "facial_age" and task != "regression":
        raise ValueError("Final facial-age task must be continuous regression")

    if args.dataset == "toronto_houses":
        features, verification = load_toronto_features(frame, args.toronto_cache)
    else:
        references = known_reference_specifications(
            str(spec["reference_embedding_layout"]),
            str(config["research_root"]),
            dataset=spec.get("reference_dataset"),
        )
        features, _, verification = load_reference_features(frame, {name: references[name] for name in ENCODERS})

    class_mapping: dict[str, int] | None = None
    if task == "regression":
        y = frame["outcome"].to_numpy(dtype=float)
    else:
        class_names = sorted(frame["outcome"].astype(str).unique().tolist())
        class_mapping = {name: index for index, name in enumerate(class_names)}
        y = frame["outcome"].astype(str).map(class_mapping).to_numpy(dtype=int)
    groups = frame["group_id"].astype(str).to_numpy(object)
    hashes = frame["image_sha256"].astype(str).to_numpy(object)
    labels = frame["split"].astype(str).to_numpy()
    selection_metric = "mae" if args.dataset == "facial_age" else "mse"
    head_components = 128
    output = args.output / args.dataset
    output.mkdir(parents=True, exist_ok=True)

    fixed = nested_validation_selection(
        features,
        y,
        groups,
        labels,
        task=task,
        encoder_order=ENCODERS,
        methods=METHODS,
        regression_selection_metric=selection_metric,
        inner_folds=3,
        random_state=int(config.get("random_state", 20260818)),
        head_max_components=head_components,
        feature_concatenation_max_components=128,
        logistic_max_iter=int(config.get("logistic_max_iter", 2000)),
        working_dtype=str(spec.get("head_working_dtype", "float64")),
    )
    fixed_rows = metric_rows(args.dataset, "fixed", None, fixed)
    pd.DataFrame(fixed_rows).to_csv(output / "fixed_metrics.csv", index=False)
    prediction_frame(frame, labels, fixed, None).to_csv(output / "fixed_test_predictions.csv", index=False)
    write_json(output / "fixed_nested_audit.json", fixed.audit)

    repeated_rows: list[dict[str, object]] = []
    repeated_predictions: list[pd.DataFrame] = []
    assignment_rows: list[pd.DataFrame] = []
    repeated_audits: list[dict[str, object]] = []
    base_seed = int(config.get("random_state", 20260818))
    for split_index in range(args.repetitions):
        seed = base_seed + 1009 * (split_index + 1)
        generated = make_split(y, groups, hashes, task=task, random_state=seed)
        repeat_labels = generated.labels.astype(str)
        validate_partition_safety(repeat_labels, groups, hashes)
        result = nested_validation_selection(
            features,
            y,
            groups,
            repeat_labels,
            task=task,
            encoder_order=ENCODERS,
            methods=METHODS,
            regression_selection_metric=selection_metric,
            inner_folds=3,
            random_state=seed,
            head_max_components=head_components,
            feature_concatenation_max_components=128,
            logistic_max_iter=int(config.get("logistic_max_iter", 2000)),
            working_dtype=str(spec.get("head_working_dtype", "float64")),
        )
        repeated_rows.extend(metric_rows(args.dataset, "repeated", split_index, result))
        repeated_predictions.append(prediction_frame(frame, repeat_labels, result, split_index))
        assignment_rows.append(pd.DataFrame({"split_index": split_index, "seed": seed, "observation_id": frame["observation_id"].astype(str), "group_id": groups, "split": repeat_labels}))
        repeated_audits.append({"split_index": split_index, "seed": seed, **result.audit})
        print(json.dumps({"dataset": args.dataset, "split": split_index + 1, "of": args.repetitions, "selected": result.selected_encoder, "preferred": result.preferred_method}), flush=True)

    pd.DataFrame(repeated_rows).to_csv(output / "repeated_metrics.csv", index=False)
    if repeated_predictions:
        pd.concat(repeated_predictions, ignore_index=True).to_csv(output / "repeated_test_predictions.csv", index=False)
        pd.concat(assignment_rows, ignore_index=True).to_csv(output / "repeated_assignments.csv", index=False)
    write_json(output / "repeated_nested_audits.json", repeated_audits)
    write_json(
        output / "run_record.json",
        {
            "status": "PASS",
            "dataset": args.dataset,
            "task": task,
            "package_commit_at_run": os.environ.get("LOOKAGAIN_PACKAGE_COMMIT"),
            "protocol": "honest_nested_outer_validation_v1",
            "fixed_selected_encoder": fixed.selected_encoder,
            "fixed_preferred_method": fixed.preferred_method,
            "repetitions": args.repetitions,
            "selection_metric": fixed.selection_metric,
            "class_mapping": class_mapping,
            "common_head_pca_cap": head_components,
            "reference_feature_verification": verification,
            "outer_validation_outcomes_used_to_fit_candidates": False,
            "test_outcomes_used_for_selection": False,
            "row_level_predictions_exported_for_every_repeated_test_partition": bool(args.repetitions),
        },
    )


if __name__ == "__main__":
    main()
