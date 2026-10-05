"""TRINETRA Logistic Regression Baseline & SIH Benchmarking.

Provides a reproducible, leakage-safe classical ML classification baseline
for cyber attack forecasting and detection, benchmarked against SIH requirements:
- Precision
- Recall
- F1 Score
- False Positive Rate (FPR)

TASK DISTINCTION:
- TRINETRA World Model: Continuous temporal future-state forecasting (evaluated via MAE/MSE).
- Logistic Regression: Supervised attack classification (evaluated via Precision/Recall/F1/FPR).
These are complementary evaluation tracks, NOT interchangeable metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ..models.label_maps import is_attack
from ..telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class LogisticBaselineConfig:
    """Hyperparameter and operational configuration for Logistic Regression baseline."""
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"
    solver: str = "liblinear"
    random_state: int = 42
    mode: str = "binary"  # "binary" or "multiclass"
    decision_threshold: float = 0.5
    feature_representation: str = "flattened_window"  # "flattened_window", "mean_pooled", "network_state", "flow_features"
    tolerance: float = 1e-4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationMetricsReport:
    """SIH-compliant evaluation report for attack classification."""
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    tp: int
    tn: int
    fp: int
    fn: int
    support_benign: int
    support_attack: int
    total_samples: int
    confusion_matrix: List[List[int]]
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    class_distribution: Dict[str, int] = field(default_factory=dict)
    decision_threshold: float = 0.5
    per_class_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class LogisticRegressionBaseline:
    """Supervised Logistic Regression baseline with training-only scaling and leakage protection."""

    def __init__(self, config: Optional[LogisticBaselineConfig] = None) -> None:
        self.config = config or LogisticBaselineConfig()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            solver=self.config.solver,
            random_state=self.config.random_state,
            tol=self.config.tolerance,
        )
        self.is_fitted = False
        self.feature_dim: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    def _validate_input_arrays(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input arrays for NaN, Inf, and dimension consistency."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy.ndarray, got {type(X)}")
        if X.size == 0 or X.shape[0] == 0:
            raise ValueError("Input feature array X cannot be empty.")
        if np.isnan(X).any():
            raise ValueError("Input feature array X contains NaN values.")
        if np.isinf(X).any():
            raise ValueError("Input feature array X contains infinite values.")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(f"y must be a numpy.ndarray, got {type(y)}")
            if len(y) != len(X):
                raise ValueError(f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples.")
            if np.isnan(y).any():
                raise ValueError("Target label array y contains NaN values.")
            if len(np.unique(y)) < 2:
                raise ValueError(
                    f"Target label array y must contain at least 2 distinct classes for classification. "
                    f"Found classes: {np.unique(y)}"
                )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegressionBaseline:
        """Fit scaler and Logistic Regression strictly on training data.
        
        LEAKAGE SAFEGUARD:
        Scaler is fit ONLY on X_train. No validation or test data is seen.
        """
        self._validate_input_arrays(X_train, y_train)

        # 2D flattening if input has higher dimension (e.g. [N, T, F] -> [N, T*F])
        if X_train.ndim > 2:
            n_samples = X_train.shape[0]
            X_train = X_train.reshape(n_samples, -1)

        self.feature_dim = X_train.shape[1]

        # 1. Fit scaler strictly on training samples without scikit-learn float64 duplication
        try:
            n_samples = len(X_train)
            if n_samples > 100_000:
                mean_sum = np.zeros(X_train.shape[1], dtype=np.float64)
                sq_sum = np.zeros(X_train.shape[1], dtype=np.float64)
                chunk_step = 100_000
                for i in range(0, n_samples, chunk_step):
                    X_chk = X_train[i : i + chunk_step]
                    mean_sum += np.sum(X_chk, axis=0, dtype=np.float64)
                    sq_sum += np.sum(X_chk.astype(np.float64) ** 2, axis=0)

                mean_ = mean_sum / n_samples
                var_ = np.maximum((sq_sum / n_samples) - (mean_ ** 2), 0.0)
            else:
                mean_ = np.mean(X_train, axis=0, dtype=np.float64)
                var_ = np.var(X_train, axis=0, dtype=np.float64)

            scale_ = np.sqrt(var_)
            scale_[scale_ == 0.0] = 1.0
            self.scaler.mean_ = mean_
            self.scaler.var_ = var_
            self.scaler.scale_ = scale_
            self.scaler.n_samples_seen_ = n_samples
            self.scaler.n_features_in_ = X_train.shape[1]

            mean_f32 = mean_.astype(np.float32)
            scale_f32 = scale_.astype(np.float32)

            max_train = 250_000
            if n_samples > max_train:
                step = max(1, n_samples // max_train)
                X_sub = X_train[::step]
                y_train_fit = y_train[::step]
                X_train_scaled = np.clip((X_sub - mean_f32) / scale_f32, -10.0, 10.0)
            else:
                X_train_scaled = np.clip((X_train - mean_f32) / scale_f32, -10.0, 10.0)
                y_train_fit = y_train
        except Exception as e:
            logger.warning("Custom scaler fit failed (%s); falling back to StandardScaler", e)
            self.scaler.copy = False
            X_train_scaled = np.clip(self.scaler.fit_transform(X_train), -10.0, 10.0)
            y_train_fit = y_train

        # Handle any zero-variance column edge cases where scaling produces NaN
        if np.isnan(X_train_scaled).any():
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

        # 2. Fit model
        self.model.fit(X_train_scaled, y_train_fit)
        self.classes_ = self.model.classes_
        self.is_fitted = True

        logger.info(
            "Fitted LogisticRegressionBaseline on %d samples (dim=%d, classes=%s)",
            len(X_train),
            self.feature_dim,
            list(self.classes_),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels with configurable decision threshold for binary mode."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        self._validate_input_arrays(X)

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        if self.config.mode == "binary" and len(self.classes_) == 2:
            probas = self.predict_proba(X)
            pos_idx = 1 if 1 in self.classes_ else len(self.classes_) - 1
            pos_proba = probas[:, pos_idx]
            preds = (pos_proba >= self.config.decision_threshold).astype(int)
            return preds

        n_samples = len(X)
        if n_samples > 100_000:
            preds = np.empty(n_samples, dtype=int)
            chunk_size = 100_000
            for i in range(0, n_samples, chunk_size):
                sub_X = X[i : i + chunk_size]
                sub_scaled = np.clip(self.scaler.transform(sub_X), -10.0, 10.0)
                if np.isnan(sub_scaled).any():
                    sub_scaled = np.nan_to_num(sub_scaled, nan=0.0)
                preds[i : i + chunk_size] = self.model.predict(sub_scaled)
            return preds

        X_scaled = np.clip(self.scaler.transform(X), -10.0, 10.0)
        if np.isnan(X_scaled).any():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using training-fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        self._validate_input_arrays(X)

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        n_samples = len(X)
        if n_samples > 100_000:
            probas = np.empty((n_samples, len(self.classes_)), dtype=np.float32)
            chunk_size = 100_000
            for i in range(0, n_samples, chunk_size):
                sub_X = X[i : i + chunk_size]
                sub_scaled = np.clip(self.scaler.transform(sub_X), -10.0, 10.0)
                if np.isnan(sub_scaled).any():
                    sub_scaled = np.nan_to_num(sub_scaled, nan=0.0)
                probas[i : i + chunk_size] = self.model.predict_proba(sub_scaled)
            return probas

        X_scaled = np.clip(self.scaler.transform(X), -10.0, 10.0)
        if np.isnan(X_scaled).any():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        return self.model.predict_proba(X_scaled)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ClassificationMetricsReport:
        """Compute SIH-required metrics: Precision, Recall, F1 Score, and False Positive Rate (FPR)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        self._validate_input_arrays(X_test, y_test)

        if X_test.ndim > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)

        y_true = np.asarray(y_test, dtype=int)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        total_samples = len(y_true)

        # Binary evaluation convention: 0 = BENIGN, 1 = ATTACK
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        support_benign = int(np.sum(y_true == 0))
        support_attack = int(np.sum(y_true == 1))

        # Precision = TP / (TP + FP)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # F1 = 2 * (P * R) / (P + R)
        f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # False Positive Rate = FP / (FP + TN)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Confusion Matrix formatted as [[TN, FP], [FN, TP]]
        cm = [[tn, fp], [fn, tp]]

        # Optional AUC metrics if both classes present in test set
        roc_auc: Optional[float] = None
        pr_auc: Optional[float] = None
        if len(np.unique(y_true)) == 2:
            try:
                pos_idx = 1 if 1 in self.classes_ else len(self.classes_) - 1
                pos_scores = y_proba[:, pos_idx]
                roc_auc = float(roc_auc_score(y_true, pos_scores))
                pr_auc = float(average_precision_score(y_true, pos_scores))
            except Exception as e:
                logger.warning("AUC computation skipped: %s", e)

        class_dist = {
            "BENIGN": support_benign,
            "ATTACK": support_attack,
        }

        return ClassificationMetricsReport(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=fpr,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            support_benign=support_benign,
            support_attack=support_attack,
            total_samples=total_samples,
            confusion_matrix=cm,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            class_distribution=class_dist,
            decision_threshold=self.config.decision_threshold,
        )

    def save(self, output_dir: Path | str, metrics_report: Optional[ClassificationMetricsReport] = None) -> Path:
        """Save model, fitted scaler, and metadata into output directory."""
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        model_path = dest / "model.joblib"
        scaler_path = dest / "scaler.joblib"
        config_path = dest / "model_config.json"
        manifest_path = dest / "model_manifest.json"
        metrics_path = dest / "metrics.json"
        cm_path = dest / "confusion_matrix.json"

        # 1. Model & Scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # 2. Config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 3. Metrics & Confusion Matrix
        if metrics_report:
            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write(metrics_report.to_json())

            with open(cm_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "confusion_matrix": metrics_report.confusion_matrix,
                        "tp": metrics_report.tp,
                        "tn": metrics_report.tn,
                        "fp": metrics_report.fp,
                        "fn": metrics_report.fn,
                        "false_positive_rate": metrics_report.false_positive_rate,
                    },
                    f,
                    indent=2,
                )

        # 4. Manifest
        manifest = {
            "model_type": "LogisticRegressionBaseline",
            "framework": "scikit-learn",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_representation": self.config.feature_representation,
            "feature_dim": self.feature_dim,
            "positive_class": "ATTACK (1)",
            "negative_class": "BENIGN (0)",
            "decision_threshold": self.config.decision_threshold,
            "model_file": "model.joblib",
            "scaler_file": "scaler.joblib",
            "config_file": "model_config.json",
        }
        if metrics_report:
            manifest["test_metrics"] = {
                "precision": metrics_report.precision,
                "recall": metrics_report.recall,
                "f1_score": metrics_report.f1_score,
                "false_positive_rate": metrics_report.false_positive_rate,
                "roc_auc": metrics_report.roc_auc,
                "pr_auc": metrics_report.pr_auc,
            }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved Logistic Regression baseline artifacts to %s", dest)
        return dest

    @classmethod
    def load(cls, src_dir: Path | str) -> LogisticRegressionBaseline:
        """Load fitted Logistic Regression baseline and scaler from directory."""
        src = Path(src_dir)
        model_path = src / "model.joblib"
        scaler_path = src / "scaler.joblib"
        config_path = src / "model_config.json"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Missing baseline artifacts in {src}")

        config = LogisticBaselineConfig()
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                config = LogisticBaselineConfig(**config_data)

        baseline = cls(config=config)
        baseline.model = joblib.load(model_path)
        baseline.scaler = joblib.load(scaler_path)
        baseline.is_fitted = True
        baseline.classes_ = baseline.model.classes_
        baseline.feature_dim = getattr(baseline.model, "n_features_in_", None)

        logger.info("Loaded LogisticRegressionBaseline from %s", src)
        return baseline


def generate_synthetic_classification_dataset(
    n_samples: int = 300,
    feature_dim: int = CANONICAL_FEATURE_DIM,
    window_size: int = 20,
    attack_ratio: float = 0.30,
    representation: str = "flattened_window",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic synthetic chronological data for software validation.
    
    CRITICAL DISCLAIMER:
    This data is exclusively for software/pipeline validation.
    It must NEVER be presented as real-world SIH benchmark performance.
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test) partitioned 70% / 15% / 15%.
    """
    np.random.seed(seed)

    # Generate sequential temporal flows with attack clusters
    n_attacks = int(n_samples * attack_ratio)
    y = np.zeros(n_samples, dtype=int)
    # Temporal clustering: attacks occur in specific bursts (e.g. last 30% of each segment)
    attack_indices = np.random.choice(n_samples, size=n_attacks, replace=False)
    y[attack_indices] = 1

    if representation == "flattened_window":
        # Each sample is a window of shape (window_size, feature_dim), flattened to (window_size * feature_dim)
        X = np.random.randn(n_samples, window_size * feature_dim).astype(np.float32)
        # Add noticeable attack signature to first few features
        X[y == 1, :10] += 2.5
    else:
        # 1D feature vectors (e.g. 78 canonical or 280 state)
        X = np.random.randn(n_samples, feature_dim).astype(np.float32)
        X[y == 1, :10] += 2.5

    # Chronological partition: 70% train, 15% val, 15% test
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test
