"""TRINETRA Unseen Attack Evaluation & Zero-Day Generalization Framework.

Provides a rigorous evaluation protocol to assess how TRINETRA models generalize
to cyber attack categories that were strictly withheld from training:
    Known Attacks + Benign -> Train
    Unseen Attack Category -> Test -> Evaluate Generalization Degradation

TWO EVALUATION TRACKS:
- Track A (Logistic Regression): Measures classification precision, recall, F1, and FPR on unseen categories.
- Track B (TRINETRA World Model): Measures temporal continuous dynamic forecasting error (MAE, MSE) on unseen trajectories.

LEAKAGE CONTROLS:
- Held-out attack categories are programmatically barred from training and validation.
- Scaler fitting strictly uses training samples (Known Attacks + Benign).
- Fail-fast enforcement if any held-out sample appears in training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..models.label_maps import INT_TO_LABEL, LABEL_ALIASES, LABEL_TO_INT, is_attack
from ..network_state.future_dataset import FutureStatePairBatch, MultiHorizonFutureDataset
from ..network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from ..world_model.checkpoint import load_world_model_checkpoint
from ..world_model.evaluator import WorldModelEvaluator
from .logistic_baseline import (
    ClassificationMetricsReport,
    LogisticBaselineConfig,
    LogisticRegressionBaseline,
)

logger = logging.getLogger(__name__)


@dataclass
class UnseenAttackConfig:
    """Configuration for an unseen attack holdout experiment."""
    held_out_categories: List[str]
    known_categories: Optional[List[str]] = None
    include_benign_in_test: bool = True
    seed: int = 42
    decision_threshold: float = 0.50
    experiment_id: str = "unseen_exp_01"

    def validate(self) -> None:
        """Validate holdout configuration."""
        if not self.held_out_categories:
            raise ValueError("held_out_categories cannot be empty.")

        # Normalize and validate each category
        canonical_held_out: List[str] = []
        for cat in self.held_out_categories:
            clean = cat.strip()
            canonical = LABEL_ALIASES.get(clean, clean)
            if canonical == "BENIGN":
                raise ValueError("BENIGN cannot be selected as a held-out attack category.")
            if canonical not in LABEL_TO_INT:
                raise ValueError(
                    f"Unknown attack category '{cat}'. Valid categories: "
                    f"{[v for v in INT_TO_LABEL.values() if v != 'BENIGN']}"
                )
            canonical_held_out.append(canonical)

        self.held_out_categories = canonical_held_out

        if self.known_categories is not None:
            # Check for overlap
            overlap = set(self.held_out_categories).intersection(set(self.known_categories))
            if overlap:
                raise ValueError(f"Held-out categories cannot overlap with known categories. Overlap: {overlap}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnseenAttackDataset:
    """Container for partitioned unseen-attack evaluation datasets."""
    X_train: np.ndarray
    y_train: np.ndarray
    categories_train: List[str]

    X_val: np.ndarray
    y_val: np.ndarray
    categories_val: List[str]

    X_test_known: np.ndarray
    y_test_known: np.ndarray
    categories_test_known: List[str]

    X_test_unseen: np.ndarray
    y_test_unseen: np.ndarray
    categories_test_unseen: List[str]

    def verify_no_leakage(self, held_out_categories: List[str]) -> None:
        """Fail-fast verification that no held-out category leaked into training or validation."""
        held_out_set = set(held_out_categories)
        train_cats = set(self.categories_train)
        val_cats = set(self.categories_val)

        train_intersection = train_cats.intersection(held_out_set)
        if train_intersection:
            raise ValueError(
                f"LEAKAGE DETECTED: Held-out categories {train_intersection} found in training set!"
            )

        val_intersection = val_cats.intersection(held_out_set)
        if val_intersection:
            raise ValueError(
                f"LEAKAGE DETECTED: Held-out categories {val_intersection} found in validation set!"
            )

        # Confirm unseen test set contains the held out category
        unseen_cats = set(self.categories_test_unseen) - {"BENIGN"}
        if not unseen_cats.intersection(held_out_set):
            raise ValueError(
                f"Unseen test set does not contain any held-out categories! Found: {set(self.categories_test_unseen)}"
            )


@dataclass
class UnseenExperimentResult:
    """Comprehensive evaluation record comparing known vs unseen performance."""
    experiment_id: str
    timestamp: str
    held_out_categories: List[str]
    known_categories: List[str]
    data_nature: str  # "SYNTHETIC SOFTWARE VALIDATION" or "REAL BENCHMARK"
    sample_counts: Dict[str, int]
    track_a_logistic: Dict[str, Any]
    track_b_world_model: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class UnseenAttackEvaluator:
    """Orchestrates unseen attack generalization experiments across Track A and Track B."""

    def __init__(
        self,
        config: UnseenAttackConfig,
        output_dir: Path | str = "artifacts/unseen_attack",
    ) -> None:
        config.validate()
        self.config = config
        self.output_dir = Path(output_dir) / self.config.experiment_id

    def evaluate_track_a_logistic(
        self,
        dataset: UnseenAttackDataset,
    ) -> Dict[str, Any]:
        """Track A: Evaluate Logistic Regression classification on Known vs Unseen test sets.
        
        Leakage Control:
        LogisticRegressionBaseline fits scaler and model ONLY on (X_train, y_train).
        """
        dataset.verify_no_leakage(self.config.held_out_categories)

        baseline_config = LogisticBaselineConfig(
            decision_threshold=self.config.decision_threshold,
            random_state=self.config.seed,
        )
        baseline = LogisticRegressionBaseline(config=baseline_config)

        # Fit model on Known Attacks + Benign
        baseline.fit(dataset.X_train, dataset.y_train)

        # Calibrate decision threshold on Known Validation set (strict zero-leakage of unseen category)
        if len(dataset.X_val) > 0 and len(np.unique(dataset.y_val)) > 1:
            try:
                val_probas = baseline.predict_proba(dataset.X_val)
                pos_idx = 1 if 1 in baseline.classes_ else len(baseline.classes_) - 1
                val_pos = val_probas[:, pos_idx]
                best_thresh = baseline.config.decision_threshold
                best_f1 = -1.0
                for th in np.linspace(0.1, 0.9, 17):
                    preds = (val_pos >= th).astype(int)
                    tp = np.sum((dataset.y_val == 1) & (preds == 1))
                    fp = np.sum((dataset.y_val == 0) & (preds == 1))
                    fn = np.sum((dataset.y_val == 1) & (preds == 0))
                    denom = 2 * tp + fp + fn
                    f1 = (2 * tp) / denom if denom > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = float(th)
                baseline.config.decision_threshold = best_thresh
            except Exception:
                pass

        # 1. Evaluate on Known Test Split
        known_report = baseline.evaluate(dataset.X_test_known, dataset.y_test_known)

        # 2. Evaluate on Unseen Test Split
        unseen_report = baseline.evaluate(dataset.X_test_unseen, dataset.y_test_unseen)

        # 3. Calculate Performance Degradation Delta
        delta_f1 = float(known_report.f1_score - unseen_report.f1_score)
        delta_recall = float(known_report.recall - unseen_report.recall)
        delta_fpr = float(unseen_report.false_positive_rate - known_report.false_positive_rate)

        return {
            "known_attack_metrics": known_report.to_dict(),
            "unseen_attack_metrics": unseen_report.to_dict(),
            "performance_degradation": {
                "delta_f1": delta_f1,
                "delta_recall": delta_recall,
                "delta_fpr": delta_fpr,
            },
        }

    def evaluate_track_b_world_model(
        self,
        world_model_dir: Path | str = "artifacts/world_model",
        synthetic_unseen_trajectories: Optional[MultiHorizonFutureDataset] = None,
        synthetic_known_trajectories: Optional[MultiHorizonFutureDataset] = None,
    ) -> Optional[Dict[str, Any]]:
        """Track B: Evaluate World Model multi-horizon continuous prediction on unseen trajectories."""
        wm_path = Path(world_model_dir) / "best_model.pt"
        if not wm_path.exists():
            logger.warning("Production World Model not found in %s. Skipping Track B.", world_model_dir)
            return None

        model, _, _ = load_world_model_checkpoint(world_model_dir)
        evaluator = WorldModelEvaluator(model)

        results: Dict[str, Any] = {}

        if synthetic_known_trajectories is not None:
            known_eval = evaluator.evaluate_multi_horizon(synthetic_known_trajectories)
            results["known_trajectory_eval"] = {
                f"k_{k}": known_eval["results_by_horizon"][f"k_{k}"]["world_model"]["mae"]
                for k in synthetic_known_trajectories.horizons
            }

        if synthetic_unseen_trajectories is not None:
            unseen_eval = evaluator.evaluate_multi_horizon(synthetic_unseen_trajectories)
            results["unseen_trajectory_eval"] = {
                f"k_{k}": unseen_eval["results_by_horizon"][f"k_{k}"]["world_model"]["mae"]
                for k in synthetic_unseen_trajectories.horizons
            }

            if synthetic_known_trajectories is not None:
                # Compare MAEs
                mae_diffs = {}
                for k in synthetic_unseen_trajectories.horizons:
                    k_str = f"k_{k}"
                    mae_k = results["known_trajectory_eval"][k_str]
                    mae_u = results["unseen_trajectory_eval"][k_str]
                    mae_diffs[k_str] = float(mae_u - mae_k)
                results["mae_degradation_delta"] = mae_diffs

        return results

    def run_experiment(
        self,
        dataset: UnseenAttackDataset,
        world_model_dir: Path | str = "artifacts/world_model",
        unseen_wm_dataset: Optional[MultiHorizonFutureDataset] = None,
        known_wm_dataset: Optional[MultiHorizonFutureDataset] = None,
        data_nature: str = "SYNTHETIC SOFTWARE VALIDATION",
    ) -> UnseenExperimentResult:
        """Execute full unseen attack evaluation experiment and serialize artifacts."""
        dataset.verify_no_leakage(self.config.held_out_categories)

        # 1. Track A: Logistic Regression
        track_a = self.evaluate_track_a_logistic(dataset)

        # 2. Track B: World Model (if trajectories provided or model exists)
        track_b = self.evaluate_track_b_world_model(
            world_model_dir=world_model_dir,
            synthetic_unseen_trajectories=unseen_wm_dataset,
            synthetic_known_trajectories=known_wm_dataset,
        )

        counts = {
            "train_samples": len(dataset.X_train),
            "val_samples": len(dataset.X_val),
            "known_test_samples": len(dataset.X_test_known),
            "unseen_test_samples": len(dataset.X_test_unseen),
        }

        # Known categories inferred from dataset
        known_cats = sorted(list(set(dataset.categories_train) - {"BENIGN"}))

        result = UnseenExperimentResult(
            experiment_id=self.config.experiment_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            held_out_categories=self.config.held_out_categories,
            known_categories=known_cats,
            data_nature=data_nature,
            sample_counts=counts,
            track_a_logistic=track_a,
            track_b_world_model=track_b,
        )

        # 3. Save Artifacts to isolated directory
        self.save_artifacts(result)
        return result

    def save_artifacts(self, result: UnseenExperimentResult) -> Path:
        """Save experiment artifacts into artifacts/unseen_attack/<experiment_id>/."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.output_dir / "experiment_config.json"
        manifest_path = self.output_dir / "experiment_manifest.json"
        metrics_path = self.output_dir / "metrics.json"
        results_path = self.output_dir / "results.json"
        cm_path = self.output_dir / "confusion_matrix.json"

        # Config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Results & Metrics
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "track_a_logistic": result.track_a_logistic,
                    "track_b_world_model": result.track_b_world_model,
                },
                f,
                indent=2,
            )

        # Confusion matrices
        cm_data = {
            "known_attack_cm": result.track_a_logistic["known_attack_metrics"]["confusion_matrix"],
            "unseen_attack_cm": result.track_a_logistic["unseen_attack_metrics"]["confusion_matrix"],
        }
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(cm_data, f, indent=2)

        # Manifest
        manifest = {
            "experiment_id": result.experiment_id,
            "created_at": result.timestamp,
            "data_nature": result.data_nature,
            "held_out_categories": result.held_out_categories,
            "known_categories": result.known_categories,
            "sample_counts": result.sample_counts,
            "decision_threshold": self.config.decision_threshold,
            "summary_metrics": {
                "known_f1": result.track_a_logistic["known_attack_metrics"]["f1_score"],
                "unseen_f1": result.track_a_logistic["unseen_attack_metrics"]["f1_score"],
                "f1_degradation": result.track_a_logistic["performance_degradation"]["delta_f1"],
                "unseen_fpr": result.track_a_logistic["unseen_attack_metrics"]["false_positive_rate"],
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved unseen attack experiment artifacts to %s", self.output_dir)
        return self.output_dir


def generate_synthetic_unseen_attack_dataset(
    held_out_category: str = "Bot",
    known_categories: Optional[List[str]] = None,
    samples_per_category: int = 60,
    benign_samples: int = 200,
    feature_dim: int = 78,
    seed: int = 42,
) -> UnseenAttackDataset:
    """Generate deterministic synthetic multi-category dataset for software validation.
    
    CRITICAL DISCLAIMER:
    This dataset is strictly for software validation of category holdout and leakage detection.
    It must NEVER be presented as real-world zero-day performance.
    """
    np.random.seed(seed)

    if known_categories is None:
        known_categories = ["DDoS", "PortScan", "FTP-Patator"]

    # 1. Generate Benign Data
    X_benign = np.random.randn(benign_samples, feature_dim).astype(np.float32)
    y_benign = np.zeros(benign_samples, dtype=int)
    cat_benign = ["BENIGN"] * benign_samples

    # 2. Generate Known Attacks (each with a distinctive signature offset)
    X_known_list: List[np.ndarray] = []
    cat_known_list: List[str] = []

    for i, cat in enumerate(known_categories):
        X_k = np.random.randn(samples_per_category, feature_dim).astype(np.float32)
        # Shift distinct feature subspace for each known attack
        start_feat = (i * 10) % (feature_dim - 10)
        X_k[:, start_feat:start_feat + 10] += 2.5
        X_known_list.append(X_k)
        cat_known_list.extend([cat] * samples_per_category)

    X_known = np.vstack(X_known_list)
    y_known = np.ones(len(X_known), dtype=int)

    # 3. Generate Unseen Attack (distinct signature not present in known attacks)
    X_unseen = np.random.randn(samples_per_category, feature_dim).astype(np.float32)
    # The unseen attack occupies a different feature subspace (e.g. features 60..70)
    X_unseen[:, 60:70] += 2.2
    y_unseen = np.ones(samples_per_category, dtype=int)
    cat_unseen = [held_out_category] * samples_per_category

    # 4. Partition into Train / Val / Known Test / Unseen Test
    # Benign split: 70% train, 15% val, 7.5% known test, 7.5% unseen test
    n_b_train = int(benign_samples * 0.70)
    n_b_val = int(benign_samples * 0.15)
    n_b_test_k = int(benign_samples * 0.075)

    X_b_train = X_benign[:n_b_train]
    y_b_train = y_benign[:n_b_train]
    cat_b_train = cat_benign[:n_b_train]

    X_b_val = X_benign[n_b_train:n_b_train + n_b_val]
    y_b_val = y_benign[n_b_train:n_b_train + n_b_val]
    cat_b_val = cat_benign[n_b_train:n_b_train + n_b_val]

    X_b_test_k = X_benign[n_b_train + n_b_val:n_b_train + n_b_val + n_b_test_k]
    y_b_test_k = y_benign[n_b_train + n_b_val:n_b_train + n_b_val + n_b_test_k]
    cat_b_test_k = cat_benign[n_b_train + n_b_val:n_b_train + n_b_val + n_b_test_k]

    X_b_test_u = X_benign[n_b_train + n_b_val + n_b_test_k:]
    y_b_test_u = y_benign[n_b_train + n_b_val + n_b_test_k:]
    cat_b_test_u = cat_benign[n_b_train + n_b_val + n_b_test_k:]

    # Known attack split: 70% train, 15% val, 15% known test
    n_k_train = int(len(X_known) * 0.70)
    n_k_val = int(len(X_known) * 0.15)

    X_k_train = X_known[:n_k_train]
    y_k_train = y_known[:n_k_train]
    cat_k_train = cat_known_list[:n_k_train]

    X_k_val = X_known[n_k_train:n_k_train + n_k_val]
    y_k_val = y_known[n_k_train:n_k_train + n_k_val]
    cat_k_val = cat_known_list[n_k_train:n_k_train + n_k_val]

    X_k_test = X_known[n_k_train + n_k_val:]
    y_k_test = y_known[n_k_train + n_k_val:]
    cat_k_test = cat_known_list[n_k_train + n_k_val:]

    # Assembled sets
    X_train = np.vstack([X_b_train, X_k_train])
    y_train = np.concatenate([y_b_train, y_k_train])
    categories_train = cat_b_train + cat_k_train

    X_val = np.vstack([X_b_val, X_k_val])
    y_val = np.concatenate([y_b_val, y_k_val])
    categories_val = cat_b_val + cat_k_val

    X_test_known = np.vstack([X_b_test_k, X_k_test])
    y_test_known = np.concatenate([y_b_test_k, y_k_test])
    categories_test_known = cat_b_test_k + cat_k_test

    # Unseen test set: Benign test samples + Unseen Attack samples
    X_test_unseen = np.vstack([X_b_test_u, X_unseen])
    y_test_unseen = np.concatenate([y_b_test_u, y_unseen])
    categories_test_unseen = cat_b_test_u + cat_unseen

    dataset = UnseenAttackDataset(
        X_train=X_train,
        y_train=y_train,
        categories_train=categories_train,
        X_val=X_val,
        y_val=y_val,
        categories_val=categories_val,
        X_test_known=X_test_known,
        y_test_known=y_test_known,
        categories_test_known=categories_test_known,
        X_test_unseen=X_test_unseen,
        y_test_unseen=y_test_unseen,
        categories_test_unseen=categories_test_unseen,
    )

    dataset.verify_no_leakage([held_out_category])
    return dataset
