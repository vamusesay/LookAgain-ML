"""Run the TensorFlow Flowers demonstration with nested out-of-fold selection."""

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


def digest_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    return parser.parse_args()


def load_features(frame: pd.DataFrame, cache_root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    features: dict[str, np.ndarray] = {}
    verification: dict[str, object] = {}
    id_digest = digest_strings(frame["observation_id"].astype(str).tolist())
    image_digest = digest_strings(frame["image_sha256"].astype(str).str.lower().tolist())
    for name in ENCODERS:
        directory = cache_root / "embeddings" / name
        arrays = list(directory.glob("*.npy"))
        records = list(directory.glob("*.json"))
        if len(arrays) != 1 or len(records) != 1:
            raise RuntimeError(f"{name}: expected one cached array and one metadata record")
        metadata = json.loads(records[0].read_text(encoding="utf-8"))
        identity = metadata["identity"]
        if identity["ordered_observation_id_sha256"] != id_digest:
            raise RuntimeError(f"{name}: observation-ID digest mismatch")
        if identity["ordered_image_hash_sha256"] != image_digest:
            raise RuntimeError(f"{name}: image digest mismatch")
        if hashlib.sha256(arrays[0].read_bytes()).hexdigest().lower() != metadata["array_sha256"].lower():
            raise RuntimeError(f"{name}: cached array digest mismatch")
        array = np.load(arrays[0], mmap_mode="r", allow_pickle=False)
        if array.shape[0] != len(frame) or not np.isfinite(array).all():
            raise RuntimeError(f"{name}: cached feature array is not finite and row-aligned")
        features[name] = array
        verification[name] = {
            "status": "PASS",
            "rows": int(array.shape[0]),
            "dimension": int(array.shape[1]),
            "array_sha256": metadata["array_sha256"],
        }
    return features, verification


def metric_rows(scope: str, split_index: int | None, result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, metrics in result.test_metrics.items():
        rows.append(
            {
                "scope": scope,
                "split_index": split_index,
                "method": method,
                "method_type": "encoder" if method in ENCODERS else "combination",
                "selected_single": result.selected_encoder,
                "preferred_method": result.preferred_method,
                "validation_selected_single": method == result.selected_encoder,
                "validation_preferred_method": method == result.preferred_method,
                "selection_metric": result.selection_metric,
                "validation_loss": float(result.validation_metrics[method][result.selection_metric]),
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
    return rows


def prediction_frame(frame: pd.DataFrame, labels: np.ndarray, result, split_index: int | None) -> pd.DataFrame:
    test = labels == "test"
    output = pd.DataFrame(
        {
            "split_index": split_index,
            "observation_id": frame.loc[test, "observation_id"].astype(str).to_numpy(),
            "group_id": frame.loc[test, "effective_group_id"].astype(str).to_numpy(),
            "outcome": frame.loc[test, "class_index"].to_numpy(dtype=int),
            "selected_single": result.selected_encoder,
            "preferred_method": result.preferred_method,
        }
    )
    for method, prediction in result.test_predictions.items():
        for class_index in range(prediction.shape[1]):
            output[f"pred_{method}__class_{class_index}"] = prediction[:, class_index]
    return output


def paired_bootstrap(frame: pd.DataFrame, result, repetitions: int = 2000) -> dict[str, object]:
    test = frame["split"].eq("test").to_numpy()
    y = frame.loc[test, "class_index"].to_numpy(dtype=int)
    groups = frame.loc[test, "effective_group_id"].astype(str).to_numpy()
    selected = result.test_predictions[result.selected_encoder]
    preferred = result.test_predictions[result.preferred_method]
    unique_groups = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group) for group in unique_groups]
    selected_point = float(np.mean(np.argmax(selected, axis=1) == y))
    preferred_point = float(np.mean(np.argmax(preferred, axis=1) == y))
    rng = np.random.default_rng(20260818)
    draws = np.empty((repetitions, 3), dtype=float)
    for index in range(repetitions):
        sampled = rng.integers(0, len(group_rows), size=len(group_rows))
        rows = np.concatenate([group_rows[value] for value in sampled])
        selected_accuracy = np.mean(np.argmax(selected[rows], axis=1) == y[rows])
        preferred_accuracy = np.mean(np.argmax(preferred[rows], axis=1) == y[rows])
        draws[index] = selected_accuracy, preferred_accuracy, preferred_accuracy - selected_accuracy
    return {
        "method": "paired grouped bootstrap on fixed fitted test predictions",
        "repetitions": repetitions,
        "models_retrained_in_draw": False,
        "selected_single": result.selected_encoder,
        "preferred_method": result.preferred_method,
        "selected_single_accuracy": selected_point,
        "preferred_method_accuracy": preferred_point,
        "preferred_minus_selected": preferred_point - selected_point,
        "selected_interval_95": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
        "preferred_interval_95": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
        "difference_interval_95": np.quantile(draws[:, 2], [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.package_root / "src"))
    from lookagain_ml import nested_validation_selection
    from lookagain_ml.splitting import make_split, validate_partition_safety

    frame = pd.read_csv(args.split_csv, dtype={"class_label": str})
    if frame["split"].value_counts().to_dict() != {"train": 2568, "validation": 550, "test": 550}:
        raise RuntimeError("The locked Flowers split counts changed")
    features, verification = load_features(frame, args.cache_root)
    y = frame["class_index"].to_numpy(dtype=int)
    groups = frame["effective_group_id"].astype(str).to_numpy(object)
    hashes = frame["image_sha256"].astype(str).to_numpy(object)
    labels = frame["split"].astype(str).to_numpy()

    def run(local_labels: np.ndarray, seed: int):
        return nested_validation_selection(
            features,
            y,
            groups,
            local_labels,
            task="classification",
            encoder_order=ENCODERS,
            methods=["linear_stack"],
            inner_folds=3,
            random_state=seed,
            head_max_components=128,
            logistic_cs=(0.03, 0.1, 0.3, 1.0, 3.0),
            stack_c=0.1,
            logistic_max_iter=2000,
            working_dtype="float64",
        )

    fixed = run(labels, 20260818)
    pd.DataFrame(metric_rows("fixed", None, fixed)).to_csv(args.output / "fixed_metrics.csv", index=False)
    prediction_frame(frame, labels, fixed, None).to_csv(args.output / "fixed_test_predictions.csv", index=False)
    write_json(args.output / "fixed_nested_audit.json", fixed.audit)
    write_json(args.output / "fixed_paired_bootstrap.json", paired_bootstrap(frame, fixed))

    repeated_rows: list[dict[str, object]] = []
    repeated_predictions: list[pd.DataFrame] = []
    repeated_audits: list[dict[str, object]] = []
    assignments: list[pd.DataFrame] = []
    for split_index in range(args.repetitions):
        seed = 20260818 + 1009 * (split_index + 1)
        generated = make_split(y, groups, hashes, task="classification", random_state=seed)
        repeat_labels = generated.labels.astype(str)
        validate_partition_safety(repeat_labels, groups, hashes)
        result = run(repeat_labels, seed)
        repeated_rows.extend(metric_rows("repeated", split_index, result))
        repeated_predictions.append(prediction_frame(frame, repeat_labels, result, split_index))
        repeated_audits.append({"split_index": split_index, "seed": seed, **result.audit})
        assignments.append(
            pd.DataFrame(
                {
                    "split_index": split_index,
                    "seed": seed,
                    "observation_id": frame["observation_id"].astype(str),
                    "group_id": groups,
                    "split": repeat_labels,
                }
            )
        )
        print(json.dumps({"split": split_index + 1, "selected": result.selected_encoder, "preferred": result.preferred_method}), flush=True)

    repeated = pd.DataFrame(repeated_rows)
    repeated.to_csv(args.output / "repeated_metrics.csv", index=False)
    pd.concat(repeated_predictions, ignore_index=True).to_csv(args.output / "repeated_test_predictions.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(args.output / "repeated_assignments.csv", index=False)
    write_json(args.output / "repeated_nested_audits.json", repeated_audits)
    write_json(
        args.output / "run_record.json",
        {
            "status": "PASS",
            "package_commit_at_run": os.environ.get("LOOKAGAIN_PACKAGE_COMMIT"),
            "protocol": "honest_nested_outer_validation_v1",
            "fixed_selected_encoder": fixed.selected_encoder,
            "fixed_preferred_method": fixed.preferred_method,
            "repetitions": args.repetitions,
            "candidate_methods": ["selected_single", "linear_stack"],
            "outer_validation_outcomes_used_to_fit_candidates": False,
            "test_outcomes_used_for_selection": False,
            "reference_feature_verification": verification,
        },
    )


if __name__ == "__main__":
    main()
