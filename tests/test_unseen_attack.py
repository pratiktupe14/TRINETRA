"""Comprehensive Unit and Integration Tests for TRINETRA Unseen Attack Evaluation Framework."""

import json
import tempfile
from pathlib import Path
import pytest
import numpy as np

from shield_rca.evaluation.unseen_attack import (
    UnseenAttackConfig,
    UnseenAttackDataset,
    UnseenAttackEvaluator,
    generate_synthetic_unseen_attack_dataset,
)


def test_holdout_config_validation():
    # Valid single category
    c1 = UnseenAttackConfig(held_out_categories=["Bot"])
    c1.validate()
    assert c1.held_out_categories == ["Bot"]

    # Valid with alias resolution (e.g. "DDoS attacks-LOIC-HTTP" -> "DDoS")
    c2 = UnseenAttackConfig(held_out_categories=["DDOS attack-HOIC"])
    c2.validate()
    assert c2.held_out_categories == ["DDoS"]


def test_unknown_category_rejection():
    c = UnseenAttackConfig(held_out_categories=["AlienSuperAttack"])
    with pytest.raises(ValueError, match="Unknown attack category"):
        c.validate()


def test_empty_holdout_rejection():
    c = UnseenAttackConfig(held_out_categories=[])
    with pytest.raises(ValueError, match="cannot be empty"):
        c.validate()


def test_benign_as_holdout_rejection():
    c = UnseenAttackConfig(held_out_categories=["BENIGN"])
    with pytest.raises(ValueError, match="BENIGN cannot be selected as a held-out attack category"):
        c.validate()


def test_known_and_held_out_overlap_rejection():
    c = UnseenAttackConfig(
        held_out_categories=["Bot"],
        known_categories=["Bot", "DDoS"],
    )
    with pytest.raises(ValueError, match="cannot overlap"):
        c.validate()


def test_leakage_detection_fail_fast():
    dataset = generate_synthetic_unseen_attack_dataset(held_out_category="Bot", seed=42)

    # 1. Normal dataset has no leakage
    dataset.verify_no_leakage(["Bot"])

    # 2. Artificially contaminate training set with held-out label
    dataset.categories_train.append("Bot")
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        dataset.verify_no_leakage(["Bot"])


def test_validation_leakage_fail_fast():
    dataset = generate_synthetic_unseen_attack_dataset(held_out_category="Bot", seed=42)
    # Contaminate validation set
    dataset.categories_val.append("Bot")
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        dataset.verify_no_leakage(["Bot"])


def test_unseen_test_must_contain_held_out_category():
    dataset = generate_synthetic_unseen_attack_dataset(held_out_category="Bot", seed=42)
    # Check for another category that is not in unseen test
    with pytest.raises(ValueError, match="does not contain any held-out categories"):
        dataset.verify_no_leakage(["Infiltration"])


def test_strict_deterministic_split():
    d1 = generate_synthetic_unseen_attack_dataset(held_out_category="Bot", seed=42)
    d2 = generate_synthetic_unseen_attack_dataset(held_out_category="Bot", seed=42)

    np.testing.assert_array_equal(d1.X_train, d2.X_train)
    np.testing.assert_array_equal(d1.y_train, d2.y_train)
    np.testing.assert_array_equal(d1.X_test_unseen, d2.X_test_unseen)
    assert d1.categories_train == d2.categories_train


def test_multiple_category_holdout():
    # Support multiple held-out categories
    config = UnseenAttackConfig(held_out_categories=["Bot", "Infiltration"])
    config.validate()
    assert config.held_out_categories == ["Bot", "Infiltration"]


def test_track_a_logistic_regression_unseen_evaluation():
    dataset = generate_synthetic_unseen_attack_dataset(held_out_category="Bot", seed=42)
    config = UnseenAttackConfig(held_out_categories=["Bot"], seed=42, experiment_id="exp_test_01")

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = UnseenAttackEvaluator(config=config, output_dir=tmpdir)
        result = evaluator.run_experiment(dataset=dataset)

        assert result.experiment_id == "exp_test_01"
        assert result.held_out_categories == ["Bot"]

        # Track A metrics
        track_a = result.track_a_logistic
        assert "known_attack_metrics" in track_a
        assert "unseen_attack_metrics" in track_a
        assert "performance_degradation" in track_a

        known_m = track_a["known_attack_metrics"]
        unseen_m = track_a["unseen_attack_metrics"]
        deg = track_a["performance_degradation"]

        assert 0.0 <= known_m["f1_score"] <= 1.0
        assert 0.0 <= unseen_m["f1_score"] <= 1.0
        assert "delta_f1" in deg
        assert "delta_recall" in deg
        assert "delta_fpr" in deg

        # Check saved artifacts
        exp_dir = Path(tmpdir) / "exp_test_01"
        assert (exp_dir / "experiment_config.json").exists()
        assert (exp_dir / "experiment_manifest.json").exists()
        assert (exp_dir / "metrics.json").exists()
        assert (exp_dir / "results.json").exists()
        assert (exp_dir / "confusion_matrix.json").exists()

        # Check manifest contents
        with open(exp_dir / "experiment_manifest.json", "r") as f:
            manifest = json.load(f)
            assert manifest["experiment_id"] == "exp_test_01"
            assert manifest["held_out_categories"] == ["Bot"]
            assert "summary_metrics" in manifest


def test_production_world_model_preservation():
    """Verify that unseen attack evaluation does not overwrite artifacts/world_model/best_model.pt."""
    prod_weights = Path("artifacts/world_model/best_model.pt")
    if prod_weights.exists():
        initial_mtime = prod_weights.stat().st_mtime
        initial_size = prod_weights.stat().st_size

        dataset = generate_synthetic_unseen_attack_dataset(held_out_category="Bot", seed=42)
        config = UnseenAttackConfig(held_out_categories=["Bot"], seed=42, experiment_id="exp_wm_preservation")

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = UnseenAttackEvaluator(config=config, output_dir=tmpdir)
            evaluator.run_experiment(dataset=dataset)

        # Check production weights were untouched
        assert prod_weights.stat().st_mtime == initial_mtime
        assert prod_weights.stat().st_size == initial_size
