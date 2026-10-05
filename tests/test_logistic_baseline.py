"""Comprehensive Unit and Integration Tests for TRINETRA Logistic Regression Baseline."""

import json
import tempfile
from pathlib import Path
import pytest
import numpy as np

from shield_rca.evaluation.logistic_baseline import (
    ClassificationMetricsReport,
    LogisticBaselineConfig,
    LogisticRegressionBaseline,
    generate_synthetic_classification_dataset,
)


def test_model_initialization():
    config = LogisticBaselineConfig(C=0.5, max_iter=500, class_weight="balanced")
    baseline = LogisticRegressionBaseline(config=config)
    assert baseline.config.C == 0.5
    assert baseline.config.max_iter == 500
    assert baseline.config.class_weight == "balanced"
    assert baseline.is_fitted is False


def test_training_and_prediction_on_valid_data():
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_classification_dataset(
        n_samples=100, feature_dim=10, window_size=1, seed=42
    )
    baseline = LogisticRegressionBaseline()
    baseline.fit(X_train, y_train)

    assert baseline.is_fitted is True
    assert baseline.feature_dim == 10

    # Predictions
    preds = baseline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})

    # Probabilities
    probas = baseline.predict_proba(X_test)
    assert probas.shape == (len(X_test), 2)
    assert np.allclose(np.sum(probas, axis=1), 1.0)


def test_strict_determinism():
    X_train, y_train, _, _, X_test, _ = generate_synthetic_classification_dataset(
        n_samples=100, feature_dim=10, window_size=1, seed=42
    )
    m1 = LogisticRegressionBaseline(LogisticBaselineConfig(random_state=42)).fit(X_train, y_train)
    m2 = LogisticRegressionBaseline(LogisticBaselineConfig(random_state=42)).fit(X_train, y_train)

    p1 = m1.predict_proba(X_test)
    p2 = m2.predict_proba(X_test)
    np.testing.assert_allclose(p1, p2, rtol=1e-6, atol=1e-6)


def test_chronological_split_integrity():
    n_samples = 200
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_classification_dataset(
        n_samples=n_samples, feature_dim=10, window_size=1, seed=42
    )
    # Check 70 / 15 / 15 partition sizes
    assert len(X_train) == 140
    assert len(X_val) == 30
    assert len(X_test) == 30
    assert len(X_train) + len(X_val) + len(X_test) == n_samples


def test_training_only_scaler_fitting():
    """Verify that scaler mean and variance are computed ONLY on training data."""
    X_train = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    y_train = np.array([0, 1])

    # Test data with an extreme outlier
    X_test = np.array([[1000.0, 1000.0]], dtype=np.float32)

    baseline = LogisticRegressionBaseline()
    baseline.fit(X_train, y_train)

    # Scaler mean must match X_train mean: [2.0, 2.0]
    np.testing.assert_allclose(baseline.scaler.mean_, np.array([2.0, 2.0]), atol=1e-5)
    # Scaler var must match X_train var: [1.0, 1.0]
    np.testing.assert_allclose(baseline.scaler.var_, np.array([1.0, 1.0]), atol=1e-5)


def test_nan_and_inf_rejection():
    baseline = LogisticRegressionBaseline()
    valid_y = np.array([0, 1, 0, 1])

    # 1. NaN in X
    X_nan = np.array([[1.0, 2.0], [np.nan, 4.0], [1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="contains NaN"):
        baseline.fit(X_nan, valid_y)

    # 2. Inf in X
    X_inf = np.array([[1.0, 2.0], [np.inf, 4.0], [1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="contains infinite"):
        baseline.fit(X_inf, valid_y)

    # 3. NaN in y
    X_valid = np.random.randn(4, 2)
    y_nan = np.array([0, np.nan, 0, 1])
    with pytest.raises(ValueError, match="contains NaN"):
        baseline.fit(X_valid, y_nan)


def test_single_class_training_rejection():
    baseline = LogisticRegressionBaseline()
    X = np.random.randn(10, 5)
    y_single = np.zeros(10, dtype=int)  # Only class 0

    with pytest.raises(ValueError, match="must contain at least 2 distinct classes"):
        baseline.fit(X, y_single)


def test_empty_and_mismatched_sample_handling():
    baseline = LogisticRegressionBaseline()

    # Empty X
    with pytest.raises(ValueError, match="cannot be empty"):
        baseline.fit(np.empty((0, 5)), np.array([0, 1]))

    # Mismatched lengths
    with pytest.raises(ValueError, match="Shape mismatch"):
        baseline.fit(np.random.randn(10, 5), np.array([0, 1]))


def test_sih_metrics_correctness():
    """Verify Precision, Recall, F1, and FPR formulas with known mock values."""
    baseline = LogisticRegressionBaseline()
    X_train = np.random.randn(20, 2)
    y_train = np.array([0] * 10 + [1] * 10)
    baseline.fit(X_train, y_train)

    # Create synthetic test predictions with known confusion counts:
    # y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    # y_pred = [0, 0, 1, 1, 0, 1, 1, 1]
    # TP = 3, TN = 2, FP = 2, FN = 1
    # Precision = 3 / (3 + 2) = 0.60
    # Recall = 3 / (3 + 1) = 0.75
    # F1 = 2 * 0.6 * 0.75 / (0.6 + 0.75) = 0.90 / 1.35 = 0.6667
    # FPR = FP / (FP + TN) = 2 / (2 + 2) = 0.50
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Mock predict method to return exact targets
    baseline.predict = lambda X: np.array([0, 0, 1, 1, 0, 1, 1, 1])
    baseline.predict_proba = lambda X: np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    report = baseline.evaluate(np.zeros((8, 2)), y_true)

    assert report.tp == 3
    assert report.tn == 2
    assert report.fp == 2
    assert report.fn == 1
    assert report.precision == pytest.approx(0.60)
    assert report.recall == pytest.approx(0.75)
    assert report.f1_score == pytest.approx(2 * 0.60 * 0.75 / (0.60 + 0.75))
    assert report.false_positive_rate == pytest.approx(0.50)
    assert report.confusion_matrix == [[2, 2], [1, 3]]


def test_zero_division_guard():
    """Verify metrics return 0.0 without crash when denominators are zero."""
    baseline = LogisticRegressionBaseline()
    X_train = np.random.randn(20, 2)
    y_train = np.array([0] * 10 + [1] * 10)
    baseline.fit(X_train, y_train)

    # All predicted as 0 -> TP=0, FP=0 -> Precision denominator is 0
    y_true = np.array([0, 0, 1, 1])
    baseline.predict = lambda X: np.array([0, 0, 0, 0])
    baseline.predict_proba = lambda X: np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    report = baseline.evaluate(np.zeros((4, 2)), y_true)
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1_score == 0.0
    assert report.false_positive_rate == 0.0  # FP=0, TN=2 -> 0/2 = 0.0


def test_artifact_save_and_reload():
    X_train, y_train, _, _, X_test, y_test = generate_synthetic_classification_dataset(
        n_samples=60, feature_dim=8, window_size=1, seed=42
    )
    baseline = LogisticRegressionBaseline()
    baseline.fit(X_train, y_train)
    report = baseline.evaluate(X_test, y_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "logistic_test"
        baseline.save(out_dir, metrics_report=report)

        # Check files
        assert (out_dir / "model.joblib").exists()
        assert (out_dir / "scaler.joblib").exists()
        assert (out_dir / "model_config.json").exists()
        assert (out_dir / "model_manifest.json").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "confusion_matrix.json").exists()

        # Reload
        reloaded = LogisticRegressionBaseline.load(out_dir)
        assert reloaded.is_fitted is True

        # Assert identical predictions
        preds_orig = baseline.predict(X_test)
        preds_reload = reloaded.predict(X_test)
        np.testing.assert_array_equal(preds_orig, preds_reload)


def test_world_model_production_artifact_preservation():
    """Verify that Logistic Regression operations do not modify artifacts/world_model/best_model.pt."""
    prod_weights = Path("artifacts/world_model/best_model.pt")
    if prod_weights.exists():
        initial_mtime = prod_weights.stat().st_mtime
        initial_size = prod_weights.stat().st_size

        # Train and save a baseline to temporary location
        X_train, y_train, _, _, X_test, y_test = generate_synthetic_classification_dataset(n_samples=50)
        b = LogisticRegressionBaseline().fit(X_train, y_train)
        with tempfile.TemporaryDirectory() as tmpdir:
            b.save(tmpdir)

        # Verify production weights are untouched
        assert prod_weights.stat().st_mtime == initial_mtime
        assert prod_weights.stat().st_size == initial_size
