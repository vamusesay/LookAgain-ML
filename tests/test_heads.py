import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lookagain_ml.heads import (
    fit_logistic_head,
    fit_ridge_head,
    fitted_linear_head_audit,
)


def test_ridge_selects_on_validation_and_scales_on_training_only():
    rng = np.random.default_rng(4)
    x_train = rng.normal(size=(80, 5))
    x_validation = rng.normal(size=(30, 5))
    beta = np.asarray([1.0, -2.0, 0.5, 0.0, 0.25])
    y_train = x_train @ beta + rng.normal(scale=0.1, size=80)
    y_validation = x_validation @ beta + rng.normal(scale=0.1, size=30)
    head = fit_ridge_head(
        x_train, y_train, x_validation, y_validation, alphas=[0.1, 10.0]
    )
    scaler = head.pipeline.named_steps["scale"]
    assert np.allclose(scaler.mean_, x_train.mean(axis=0))
    assert head.selected_hyperparameter["alpha"] in {0.1, 10.0}
    assert len(head.validation_grid) == 2


def test_ridge_does_not_amplify_numerically_null_training_columns():
    """A validation shift in a null PCA direction must not explode predictions."""

    rng = np.random.default_rng(41)
    signal = rng.normal(size=80)
    x_train = np.column_stack([signal, np.linspace(0.0, 1e-14, 80)])
    x_validation = np.column_stack([rng.normal(size=30), np.ones(30)])
    y_train = 2.0 * signal + rng.normal(scale=0.1, size=80)
    y_validation = 2.0 * x_validation[:, 0]

    head = fit_ridge_head(
        x_train,
        y_train,
        x_validation,
        y_validation,
        alphas=[0.1, 1.0, 10.0],
    )

    prediction = head.predict(x_validation)
    assert np.isfinite(prediction).all()
    assert np.max(np.abs(prediction)) < 100.0


def test_logistic_head_returns_normalized_probabilities():
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(100, 4))
    x_validation = rng.normal(size=(40, 4))
    y_train = (x_train[:, 0] + x_train[:, 1] > 0).astype(int)
    y_validation = (x_validation[:, 0] + x_validation[:, 1] > 0).astype(int)
    head = fit_logistic_head(
        x_train,
        y_train,
        x_validation,
        y_validation,
        c_values=[0.1, 1.0],
    )
    probabilities = head.predict(x_validation)
    assert probabilities.shape == (40, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert head.selection_metric == "log_loss"


def test_head_selection_api_has_no_test_outcome_argument():
    import inspect

    assert "y_test" not in inspect.signature(fit_ridge_head).parameters
    assert "y_test" not in inspect.signature(fit_logistic_head).parameters


def test_float32_pca_head_matches_source_fit_then_transform_sequence():
    rng = np.random.default_rng(19)
    x_train = rng.normal(size=(90, 12)).astype(np.float32)
    x_validation = rng.normal(size=(35, 12)).astype(np.float32)
    y_train = (x_train[:, 0] - 0.4 * x_train[:, 1] > 0).astype(int)
    y_validation = (x_validation[:, 0] - 0.4 * x_validation[:, 1] > 0).astype(int)
    fitted = fit_logistic_head(
        x_train,
        y_train,
        x_validation,
        y_validation,
        c_values=[0.3],
        max_components=5,
        pca_iterated_power=3,
        random_state=23,
        working_dtype="float32",
    )

    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train).astype(np.float32)
    validation = scaler.transform(x_validation).astype(np.float32)
    pca = PCA(
        n_components=5, svd_solver="randomized", random_state=23, iterated_power=3
    ).fit(train)
    train = pca.transform(train).astype(np.float32)
    validation = pca.transform(validation).astype(np.float32)
    expected = LogisticRegression(
        C=0.3, solver="lbfgs", max_iter=2000, random_state=23
    ).fit(train, y_train).predict_proba(validation)
    expected = np.asarray(expected, dtype=float)
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(fitted.predict(x_validation), expected, rtol=0, atol=1e-12)
    audit = fitted_linear_head_audit(fitted)
    assert audit["working_dtype"] == "float32"
    assert audit["pca"]["fit_partition"] == "train"
