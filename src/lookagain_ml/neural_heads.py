"""Validation-locked GELU/dropout neural-head ablation from the ICLR workflow."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import StandardScaler

from .exceptions import InputValidationError

PAPER_NEURAL_SEEDS = (101, 202, 303, 404, 505)
PAPER_TUNING_SEED = 7001
PAPER_NEURAL_CONFIGS: dict[str, dict[str, Any]] = {
    "compact": {"hidden": (128,), "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4},
    "standard": {"hidden": (256, 64), "dropout": 0.20, "lr": 5e-4, "weight_decay": 1e-4},
    "regularized": {"hidden": (256, 64), "dropout": 0.40, "lr": 3e-4, "weight_decay": 1e-3},
}


def _digest(values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _gelu(value: np.ndarray) -> np.ndarray:
    return 0.5 * value * (1.0 + erf(value / np.sqrt(2.0)))


@dataclass
class FittedGeluHead:
    task: str
    scaler: StandardScaler
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    configuration_name: str
    configuration: dict[str, Any]
    seed: int
    best_epoch: int
    validation_loss: float
    outcome_mean: float
    outcome_scale: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        hidden = self.scaler.transform(np.asarray(features, dtype=float))
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ weight.T + bias
            if index < len(self.weights) - 1:
                hidden = _gelu(hidden)
        if self.task == "regression":
            return hidden[:, 0] * self.outcome_scale + self.outcome_mean
        shifted = hidden - hidden.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        return probability / probability.sum(axis=1, keepdims=True)

    def audit(self) -> dict[str, Any]:
        return {
            "method": "paper neural-head ablation",
            "activation": "GELU",
            "dropout": self.configuration["dropout"],
            "optimizer": "AdamW",
            "learning_rate": self.configuration["lr"],
            "weight_decay": self.configuration["weight_decay"],
            "hidden": list(self.configuration["hidden"]),
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "early_stopping_partition": "validation",
            "preprocessing_fit_partition": "train",
            "test_outcomes_used": False,
            "fitted_state_sha256": _digest(
                {
                    "weights": [value.tolist() for value in self.weights],
                    "biases": [value.tolist() for value in self.biases],
                    "scaler_mean": self.scaler.mean_.tolist(),
                    "scaler_scale": self.scaler.scale_.tolist(),
                    "outcome_mean": self.outcome_mean,
                    "outcome_scale": self.outcome_scale,
                }
            ),
        }


@dataclass
class FittedSeedAveragedGeluHead:
    task: str
    heads: list[FittedGeluHead]
    architecture_selection: dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.mean([head.predict(features) for head in self.heads], axis=0)

    def audit(self) -> dict[str, Any]:
        return {
            "method": "mean prediction over fixed initialization seeds",
            "seeds": [head.seed for head in self.heads],
            "seed_count": len(self.heads),
            "seed_role": "initialization/training only; not partition seeds",
            "best_test_seed_selected": False,
            "architecture_selection": self.architecture_selection,
            "members": [head.audit() for head in self.heads],
            "test_outcomes_used": False,
        }


@dataclass
class PaperNeuralHeadResult:
    heads: dict[str, FittedSeedAveragedGeluHead]
    architecture_selection: dict[str, Any]

    def predict(self, features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = set(self.heads) - set(features)
        if missing:
            raise InputValidationError(f"Missing neural-head feature blocks: {sorted(missing)}.")
        return {name: head.predict(features[name]) for name, head in self.heads.items()}

    def audit(self) -> dict[str, Any]:
        return {
            "architecture_selection": self.architecture_selection,
            "encoders": {name: head.audit() for name, head in self.heads.items()},
            "test_outcomes_used": False,
        }


def _validate_blocks(
    train: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray]
) -> tuple[list[str], int, int]:
    names = [str(name) for name in train]
    if not names or set(names) != set(validation):
        raise InputValidationError("Training and validation feature mappings must have identical names.")
    train_n = validation_n = None
    for name in names:
        left = np.asarray(train[name], dtype=float)
        right = np.asarray(validation[name], dtype=float)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise InputValidationError(f"Neural-head block {name!r} is not column-aligned.")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise InputValidationError(f"Neural-head block {name!r} contains non-finite values.")
        train_n = left.shape[0] if train_n is None else train_n
        validation_n = right.shape[0] if validation_n is None else validation_n
        if left.shape[0] != train_n or right.shape[0] != validation_n:
            raise InputValidationError("All neural-head blocks must share row counts within partitions.")
    assert train_n is not None and validation_n is not None
    return names, train_n, validation_n


def _fit_one(
    task: str,
    train_raw: np.ndarray,
    y_train: np.ndarray,
    validation_raw: np.ndarray,
    y_validation: np.ndarray,
    *,
    configuration_name: str,
    configuration: Mapping[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    regression_selection_metric: str = "mse",
    fixed_epochs: bool = False,
    stable_scaling: bool = False,
) -> FittedGeluHead:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if regression_selection_metric not in {"mse", "mae"}:
        raise InputValidationError("Regression selection metric must be mse or mae.")
    if stable_scaling:
        from .heads import _fit_numerically_stable_scaler
        scaler = _fit_numerically_stable_scaler(train_raw)
    else:
        scaler = StandardScaler().fit(train_raw)
    train = scaler.transform(train_raw).astype(np.float32)
    validation = (np.empty((0, train.shape[1]), dtype=np.float32) if fixed_epochs
                  else scaler.transform(validation_raw).astype(np.float32))
    if task == "regression":
        outcome_mean = float(np.mean(y_train))
        outcome_scale = float(np.std(y_train))
        if outcome_scale < 1e-8:
            outcome_scale = 1.0
        train_y = ((np.asarray(y_train, dtype=np.float32) - outcome_mean) / outcome_scale)[:, None]
        validation_y = np.empty(0) if fixed_epochs else np.asarray(y_validation, dtype=float)
        output_dimension = 1
        criterion: Any = torch.nn.MSELoss()
    elif task == "classification":
        train_y = np.asarray(y_train, dtype=np.int64)
        validation_y = np.empty(0, dtype=np.int64) if fixed_epochs else np.asarray(y_validation, dtype=np.int64)
        classes = np.unique(np.concatenate([train_y, validation_y]))
        if not np.array_equal(np.unique(train_y), classes) or not np.array_equal(classes, np.arange(len(classes))):
            raise InputValidationError("Neural-head training rows must contain every encoded class.")
        output_dimension = len(classes)
        counts = np.bincount(train_y, minlength=output_dimension)
        positive = counts[counts > 0]
        weights = None
        if positive.max() / positive.min() > 2:
            values = np.zeros(output_dimension, dtype=np.float32)
            values[counts > 0] = len(train_y) / (len(positive) * counts[counts > 0])
            weights = torch.tensor(values, dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        outcome_mean, outcome_scale = 0.0, 1.0
    else:
        raise InputValidationError("task must be 'regression' or 'classification'.")
    dims = [train.shape[1], *map(int, configuration["hidden"])]
    layers: list[torch.nn.Module] = []
    for left, right in pairwise(dims):
        layers.extend([torch.nn.Linear(left, right), torch.nn.GELU(), torch.nn.Dropout(float(configuration["dropout"]))])
    layers.append(torch.nn.Linear(dims[-1], output_dimension))
    model = torch.nn.Sequential(*layers).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(configuration["lr"]), weight_decay=float(configuration["weight_decay"])
    )
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train), torch.from_numpy(train_y)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_tensor = torch.from_numpy(validation)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if fixed_epochs:
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            continue
        model.eval()
        with torch.no_grad():
            raw = model(validation_tensor).cpu().numpy()
        if task == "regression":
            prediction = raw[:, 0] * outcome_scale + outcome_mean
            error = validation_y - prediction
            validation_loss = float(np.mean(np.abs(error) if regression_selection_metric == "mae" else error ** 2))
        else:
            shifted = raw - raw.max(axis=1, keepdims=True)
            probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            validation_loss = float(
                -np.mean(np.log(np.clip(probability[np.arange(len(validation_y)), validation_y], 1e-12, 1)))
            )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Neural-head training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    linear = [module for module in model if isinstance(module, torch.nn.Linear)]
    return FittedGeluHead(
        task=task,
        scaler=scaler,
        weights=[layer.weight.detach().numpy().astype(float, copy=True) for layer in linear],
        biases=[layer.bias.detach().numpy().astype(float, copy=True) for layer in linear],
        configuration_name=configuration_name,
        configuration={**configuration, "hidden": tuple(configuration["hidden"])},
        seed=seed,
        best_epoch=best_epoch,
        validation_loss=float("nan") if fixed_epochs else best_loss,
        outcome_mean=outcome_mean,
        outcome_scale=outcome_scale,
    )


def fit_paper_neural_heads(
    task: str,
    train_features: Mapping[str, np.ndarray],
    y_train: Sequence[Any] | np.ndarray,
    validation_features: Mapping[str, np.ndarray],
    y_validation: Sequence[Any] | np.ndarray,
    *,
    architecture_anchors: Sequence[str] = ("resnet50", "dinov2_vitb14"),
    configurations: Mapping[str, Mapping[str, Any]] = PAPER_NEURAL_CONFIGS,
    tuning_seed: int = PAPER_TUNING_SEED,
    initialization_seeds: Sequence[int] = PAPER_NEURAL_SEEDS,
    max_epochs: int = 70,
    patience: int = 7,
    batch_size: int = 1024,
) -> PaperNeuralHeadResult:
    """Fit the paper's dataset-level-selected, five-seed GELU head ensemble.

    The signature deliberately has no test outcomes. Architecture is selected
    once per dataset by mean validation loss over the declared anchor blocks,
    and is then frozen across every representation.
    """

    names, train_n, validation_n = _validate_blocks(train_features, validation_features)
    y_train_array = np.asarray(y_train)
    y_validation_array = np.asarray(y_validation)
    if len(y_train_array) != train_n or len(y_validation_array) != validation_n:
        raise InputValidationError("Outcomes must align with the train and validation feature rows.")
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise InputValidationError("max_epochs, patience, and batch_size must be positive.")
    anchors = [str(value) for value in architecture_anchors]
    missing = set(anchors) - set(names)
    if missing:
        raise InputValidationError(f"Architecture anchor blocks are missing: {sorted(missing)}.")
    if not configurations or not initialization_seeds:
        raise InputValidationError("At least one configuration and initialization seed are required.")
    rows: list[dict[str, Any]] = []
    for configuration_name, configuration in configurations.items():
        for anchor in anchors:
            fitted = _fit_one(
                task,
                np.asarray(train_features[anchor], dtype=float),
                y_train_array,
                np.asarray(validation_features[anchor], dtype=float),
                y_validation_array,
                configuration_name=str(configuration_name),
                configuration=configuration,
                seed=int(tuning_seed),
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
            )
            rows.append(
                {
                    "configuration": str(configuration_name),
                    "anchor": anchor,
                    "validation_loss": fitted.validation_loss,
                    "best_epoch": fitted.best_epoch,
                }
            )
    means = {
        str(name): float(np.mean([row["validation_loss"] for row in rows if row["configuration"] == str(name)]))
        for name in configurations
    }
    selected = min([str(name) for name in configurations], key=lambda name: means[name])
    selection = {
        "rule": "minimum mean validation loss across declared anchor representations",
        "anchors": anchors,
        "tuning_seed": int(tuning_seed),
        "tuning_seed_role": "initialization only; not a partition seed",
        "rows": rows,
        "mean_validation_loss": means,
        "selected_configuration": selected,
        "selected_parameters": {
            **configurations[selected],
            "hidden": list(configurations[selected]["hidden"]),
        },
        "selection_partition": "validation",
        "test_outcomes_used": False,
    }
    fitted_heads: dict[str, FittedSeedAveragedGeluHead] = {}
    for name in names:
        members = [
            _fit_one(
                task,
                np.asarray(train_features[name], dtype=float),
                y_train_array,
                np.asarray(validation_features[name], dtype=float),
                y_validation_array,
                configuration_name=selected,
                configuration=configurations[selected],
                seed=int(seed),
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
            )
            for seed in initialization_seeds
        ]
        fitted_heads[name] = FittedSeedAveragedGeluHead(task, members, selection)
    return PaperNeuralHeadResult(fitted_heads, selection)
