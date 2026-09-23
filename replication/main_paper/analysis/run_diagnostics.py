"""Run the common-dimension, LogME, capacity, and data diagnostics fixed in the implemented workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_nested_protocol import ENCODERS, load_toronto_features


MODERN = ["resnet50", "dinov2_vitb14", "siglip2_b16", "convnext_b", "vit_b16"]
DATASETS = ["toronto_houses", "obs_horses", "breakhis", "pneumonia", "facial_age", "rice_disease"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nested-root", type=Path, required=True)
    parser.add_argument("--toronto-cache", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--horse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def primary_column(dataset: str) -> str:
    if dataset in {"toronto_houses", "obs_horses"}:
        return "test_r2"
    if dataset == "facial_age":
        return "test_mae"
    return "test_accuracy"


def load_features(dataset: str, spec: dict, config: dict, frame: pd.DataFrame, args: argparse.Namespace):
    if dataset == "toronto_houses":
        return load_toronto_features(frame, args.toronto_cache)
    from _reference_features import known_reference_specifications, load_reference_features
    references = known_reference_specifications(
        str(spec["reference_embedding_layout"]), str(config["research_root"]), dataset=spec.get("reference_dataset")
    )
    features, _, verification = load_reference_features(frame, {name: references[name] for name in ENCODERS})
    return features, verification


def main() -> None:
    args = parse_args()
    replication = args.package_root / "replication"
    sys.path.insert(0, str(args.package_root / "src"))
    sys.path.insert(0, str(replication))
    from _common import load_config, load_manifest
    from lookagain_ml.logme import logme_score

    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    specs = {item["name"]: item for item in config["datasets"]}

    logme_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        spec = specs[dataset]
        frame = load_manifest(Path(spec["manifest"]).expanduser().resolve())
        features, _ = load_features(dataset, spec, config, frame, args)
        labels = frame["split"].astype(str).to_numpy()
        train = labels == "train"
        if spec["task"] == "classification":
            names = sorted(frame["outcome"].astype(str).unique())
            mapping = {name: index for index, name in enumerate(names)}
            y = frame["outcome"].astype(str).map(mapping).to_numpy(int)
        else:
            y = frame["outcome"].to_numpy(float)
        fixed = pd.read_csv(args.nested_root / dataset / "fixed_metrics.csv")
        repeated = pd.read_csv(args.nested_root / dataset / "repeated_metrics.csv")
        encoder_fixed = fixed.loc[fixed.method_type == "encoder"].copy()
        score_map: dict[str, float] = {}
        for offset, name in enumerate(ENCODERS):
            score = logme_score(features[name][train], y[train], task=spec["task"], max_components=128, random_state=20260818 + offset)
            score_map[name] = score
            row = encoder_fixed.loc[encoder_fixed.method == name].iloc[0]
            logme_rows.append(
                {
                    "dataset": dataset,
                    "encoder": name,
                    "logme": score,
                    "validation_loss": float(row.validation_loss),
                    "test_primary": float(row[primary_column(dataset)]),
                }
            )
        for name in ["resnet50", "dinov2_vitb14", "siglip2_b16", "vit_b16"]:
            row = encoder_fixed.loc[encoder_fixed.method == name].iloc[0]
            common_rows.append(
                {
                    "dataset": dataset,
                    "encoder": name,
                    "pca_dimensions": 128,
                    "fit_partition": "outer training; locked refit on train+validation for test",
                    "primary_metric": primary_column(dataset).removeprefix("test_"),
                    "test_primary": float(row[primary_column(dataset)]),
                }
            )
        dino = repeated.loc[repeated.method == "dinov2_vitb14"].sort_values("split_index")
        vit = repeated.loc[repeated.method == "vit_b16"].sort_values("split_index")
        metric = primary_column(dataset)
        values = (vit[metric].to_numpy() - dino[metric].to_numpy()) if dataset == "facial_age" else (dino[metric].to_numpy() - vit[metric].to_numpy())
        for split_index, value in enumerate(values):
            matched_rows.append({"dataset": dataset, "split_index": split_index, "metric": metric.removeprefix("test_"), "dino_advantage": float(value)})

    logme = pd.DataFrame(logme_rows)
    logme.to_csv(args.output / "logme_scores.csv", index=False)
    diagnostics: list[dict[str, object]] = []
    for dataset, group in logme.groupby("dataset", sort=False):
        test_rank_source = -group.test_primary if dataset == "facial_age" else group.test_primary
        top_logme = str(group.loc[group.logme.idxmax(), "encoder"])
        top_validation = str(group.loc[group.validation_loss.idxmin(), "encoder"])
        top_test = str(group.loc[test_rank_source.idxmax(), "encoder"])
        diagnostics.append(
            {
                "dataset": dataset,
                "spearman_logme_vs_validation_rank": float(group.logme.corr(-group.validation_loss, method="spearman")),
                "spearman_logme_vs_test_rank": float(group.logme.corr(test_rank_source, method="spearman")),
                "logme_top": top_logme,
                "validation_top": top_validation,
                "test_top": top_test,
                "validation_winner_agreement": top_logme == top_validation,
                "test_winner_agreement": top_logme == top_test,
            }
        )
    pd.DataFrame(diagnostics).to_csv(args.output / "logme_rank_diagnostics.csv", index=False)
    pd.DataFrame(common_rows).to_csv(args.output / "common_128_control.csv", index=False)
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(args.output / "matched_capacity_split_results.csv", index=False)
    matched.groupby(["dataset", "metric"], as_index=False).dino_advantage.agg(["mean", "std", "min", "max"]).reset_index().to_csv(args.output / "matched_capacity_summary.csv", index=False)

    age_neighbors = pd.read_csv(args.review_root / "classification_checks" / "age_cross_split_nearest_neighbors.csv")
    age_pred = pd.read_csv(args.nested_root / "facial_age" / "fixed_test_predictions.csv")
    age_join = age_neighbors.merge(age_pred[["observation_id", "outcome", "pred_dinov2_vitb14"]], left_on="test_id", right_on="observation_id", validate="one_to_one")
    age_sensitivity: list[dict[str, object]] = []
    for threshold in (0.99, 0.995, 0.999):
        flagged = age_join.cosine_similarity >= threshold
        keep = ~flagged
        age_sensitivity.append(
            {
                "threshold": threshold,
                "test_n": len(age_join),
                "flagged": int(flagged.sum()),
                "flagged_share": float(flagged.mean()),
                "mae_after_exclusion": float(np.mean(np.abs(age_join.loc[keep, "outcome"] - age_join.loc[keep, "pred_dinov2_vitb14"]))),
                "full_mae": float(np.mean(np.abs(age_join.outcome - age_join.pred_dinov2_vitb14))),
            }
        )
    pd.DataFrame(age_sensitivity).to_csv(args.output / "age_near_duplicate_sensitivity.csv", index=False)

    rice_neighbors = pd.read_csv(args.review_root / "classification_checks" / "rice_cross_split_nearest_neighbors.csv")
    rice_pred = pd.read_csv(args.nested_root / "rice_disease" / "fixed_test_predictions.csv")
    rice_run = json.loads((args.nested_root / "rice_disease" / "run_record.json").read_text(encoding="utf-8"))
    inverse = {int(value): key for key, value in rice_run["class_mapping"].items()}
    probability_columns = sorted([name for name in rice_pred if name.startswith("pred_dinov2_vitb14__class_")], key=lambda name: int(name.rsplit("_", 1)[1]))
    predicted_index = rice_pred[probability_columns].to_numpy().argmax(axis=1)
    rice_pred["dino_predicted_label"] = [inverse[int(value)] for value in predicted_index]
    rice_join = rice_neighbors.merge(rice_pred[["observation_id", "outcome", "dino_predicted_label"]], left_on="test_id", right_on="observation_id", validate="one_to_one")
    rice_sensitivity: list[dict[str, object]] = []
    for threshold in (0.99, 0.995, 0.999):
        flagged = rice_join.cosine_similarity >= threshold
        keep = ~flagged
        rice_sensitivity.append(
            {
                "threshold": threshold,
                "test_n": len(rice_join),
                "flagged": int(flagged.sum()),
                "flagged_share": float(flagged.mean()),
                "same_target_among_flagged": float(rice_join.loc[flagged, "same_target"].mean()) if flagged.any() else None,
                "dino_accuracy_after_exclusion": float(np.mean(rice_join.loc[keep, "outcome"] == rice_join.loc[keep, "dino_predicted_label"])),
                "one_nn_accuracy": float(rice_join.same_target.mean()),
            }
        )
    pd.DataFrame(rice_sensitivity).to_csv(args.output / "rice_easy_manifold_sensitivity.csv", index=False)
    learning = pd.read_csv(args.review_root / "classification_checks" / "rice_learning_curve_and_1nn.csv")
    learning.to_csv(args.output / "rice_training_size_sensitivity.csv", index=False)

    horse_master = pd.read_csv(args.horse_root / "obs_horse_dataset" / "analysis" / "results" / "master_photo_index.csv", low_memory=False)
    horse = horse_master.loc[horse_master.in_speed_sample.astype(bool) & horse_master.speed_z_within_sale.notna()].copy()
    group_sizes = horse.groupby("horse_group_id").size()
    links: dict[str, object] = {}
    for kind in ("name_identity_key", "pedigree_identity_key", "photo_identity_key"):
        values = horse[kind].fillna("").astype(str)
        counts = values.loc[values.ne("")].value_counts()
        links[kind] = {"nonblank_rows": int(values.ne("").sum()), "repeated_keys": int((counts > 1).sum()), "rows_in_repeated_keys": int(counts.loc[counts > 1].sum())}
    prepared = pd.read_csv(Path(specs["obs_horses"]["manifest"]).expanduser().resolve(), low_memory=False)
    repeated_assignments = pd.read_csv(args.nested_root / "obs_horses" / "repeated_assignments.csv")
    audit = {
        "analysis_rows": len(horse),
        "final_group_count": int(group_sizes.size),
        "multi_row_groups": int((group_sizes > 1).sum()),
        "max_group_size": int(group_sizes.max()),
        "links_by_identifier_type": links,
        "ambiguities": "Conservative exact normalized name, complete sire/dam/sex/foaling identity, and exact photograph hash are connected transitively; no fuzzy-name or test-informed links are added.",
        "fixed_split_crossing_groups": int(prepared.groupby("group_id").split.nunique().gt(1).sum()),
        "repeated_split_crossing_groups": int(repeated_assignments.groupby(["split_index", "group_id"]).split.nunique().gt(1).sum()),
    }
    write_json(args.output / "horse_grouping_audit.json", audit)


if __name__ == "__main__":
    main()
