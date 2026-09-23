"""Optional training-only LogME transferability diagnostic."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .exceptions import InputValidationError


def _single_target_logme(features: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    y = y - y.mean()
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    squared = singular * singular
    alpha = beta = 1.0
    vector = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(50):
        denominator = alpha + beta * squared
        vector = vt.T @ ((beta * singular / denominator) * (u.T @ y))
        gamma = float(np.sum(beta * squared / denominator))
        alpha_new = max(gamma / max(float(vector @ vector), 1e-12), 1e-12)
        residual = y - x @ vector
        beta_new = (len(y) - gamma) / max(float(residual @ residual), 1e-12)
        if (
            abs(alpha_new - alpha) / max(alpha, 1e-12) < 1e-5
            and abs(beta_new - beta) / max(beta, 1e-12) < 1e-5
        ):
            alpha, beta = alpha_new, beta_new
            break
        alpha, beta = alpha_new, beta_new
    residual = y - x @ vector
    rank = len(squared)
    logdet = float(np.log(alpha + beta * squared).sum())
    if x.shape[1] > rank:
        logdet += (x.shape[1] - rank) * float(np.log(alpha))
    evidence = (
        x.shape[1] * np.log(alpha) / 2
        + len(y) * np.log(beta) / 2
        - beta * float(residual @ residual) / 2
        - alpha * float(vector @ vector) / 2
        - logdet / 2
        - len(y) * np.log(2 * np.pi) / 2
    )
    return float(evidence / len(y))


def logme_score(
    features_train: np.ndarray,
    y_train: np.ndarray,
    *,
    task: str,
    max_components: int = 128,
    random_state: int = 20260818,
) -> float:
    """Return LogME from training rows only; no validation/test argument exists."""

    x = np.asarray(features_train, dtype=np.float64)
    y = np.asarray(y_train)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise InputValidationError(
            "LogME requires aligned finite training features and targets."
        )
    if max_components < 1:
        raise InputValidationError("logme_max_components must be positive.")
    x = StandardScaler().fit_transform(x)
    component_count = min(max_components, x.shape[0] - 1, x.shape[1])
    if component_count < 1:
        raise InputValidationError("LogME requires at least two training observations.")
    if component_count < x.shape[1]:
        x = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=random_state,
        ).fit_transform(x)
    if task == "regression":
        return _single_target_logme(x, y.astype(float))
    if task != "classification":
        raise InputValidationError("task must be 'regression' or 'classification'.")
    classes = np.unique(y.astype(int))
    scores = [
        _single_target_logme(x, (y == value).astype(float)) for value in classes
    ]
    return float(np.mean(scores))
