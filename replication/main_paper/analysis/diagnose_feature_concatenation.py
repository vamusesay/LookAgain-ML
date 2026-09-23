"""Diagnose numerical scale in a completed package feature cache.

This script is deliberately read-only.  It reconstructs the public package's
feature-concatenation fit from a verified manifest/cache pair and reports only
aggregate numerical ranges, never observations or source paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lookagain_ml.combination import fit_feature_concatenation
from lookagain_ml.heads import DEFAULT_LOGISTIC_CS, DEFAULT_RIDGE_ALPHAS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    encoder_order = [
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
    features = {
        name: np.load(next((args.cache / "embeddings" / name).glob("*.npy")))
        for name in encoder_order
    }
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("validation").to_numpy()
    y = frame["outcome"].to_numpy()
    fitted, audit = fit_feature_concatenation(
        args.task,
        {name: values[train] for name, values in features.items()},
        y[train],
        {name: values[validation] for name, values in features.items()},
        y[validation],
        encoder_order,
        max_components_per_encoder=128,
        ridge_alphas=DEFAULT_RIDGE_ALPHAS,
        logistic_cs=DEFAULT_LOGISTIC_CS,
        random_state=20260818,
    )
    blocks = {
        name: fitted.transforms[name].transform(features[name][validation])
        for name in encoder_order
    }
    union = np.hstack([blocks[name] for name in encoder_order])
    head_scaled = fitted.head.pipeline.named_steps["scale"].transform(union)
    prediction = fitted.predict({name: values[validation] for name, values in features.items()})
    report = {
        "task": args.task,
        "rows": {"train": int(train.sum()), "validation": int(validation.sum())},
        "blocks": {
            name: {
                "maximum_absolute_transformed_validation_value": float(
                    np.max(np.abs(block))
                ),
                "finite": bool(np.isfinite(block).all()),
            }
            for name, block in blocks.items()
        },
        "maximum_absolute_head_scaled_validation_value": float(
            np.max(np.abs(head_scaled))
        ),
        "maximum_absolute_validation_prediction": float(np.max(np.abs(prediction))),
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "head_validation_grid": audit["head_validation_grid"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
