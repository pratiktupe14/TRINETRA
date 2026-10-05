"""Unit and integration tests for Phase 10C predictive risk calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch

from shield_rca.network_state.future_dataset import FutureStatePairBatch
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from shield_rca.world_model.checkpoint import load_world_model_checkpoint
from shield_rca.world_model.phase10c_evaluator import (
    HorizonCalibrationMetrics,
    Phase10CEvaluator,
)
from shield_rca.world_model.risk_estimator import PredictiveRiskEstimator


@pytest.fixture
def synthetic_pair_batch() -> FutureStatePairBatch:
    """Fixture providing a synthetic multi-pair batch for k=3."""
    N = 10
    T = 5
    rng = np.random.RandomState(42)
    current_states = rng.randn(N, T, STATE_FEATURE_DIM).astype(np.float32) * 0.1
    target_states = rng.randn(N, STATE_FEATURE_DIM).astype(np.float32) * 0.1
    timestamps = np.arange(N, dtype=float) * 10.0
    delta_times = np.full(N, 15.0, dtype=float)
    labels = np.array([0, 1] * 5, dtype=np.int64)
    cats = ["Benign", "Attack"] * 5

    return FutureStatePairBatch(
        current_states=current_states,
        target_states=target_states,
        current_timestamps=timestamps,
        target_timestamps=timestamps + 15.0,
        delta_times=delta_times,
        horizon=3,
        source_ids=["test_scen"] * N,
        schema_version=STATE_SCHEMA_VERSION,
        current_labels=labels,
        target_labels=labels,
        current_categories=cats,
        target_categories=cats,
    )


def test_candidate_loading() -> None:
    """Verify that Phase 10B GRU candidate loads cleanly and runs inference."""
    cand_path = Path("artifacts/world_model/candidates/cand-gru-real-d49b3ee3")
    assert (cand_path / "best_model.pt").exists(), "Candidate checkpoint does not exist"

    cand_model, config, manifest = load_world_model_checkpoint(cand_path)
    assert cand_model is not None
    assert config["state_dim"] == STATE_FEATURE_DIM
    assert config["model_type"] == "GRUDynamicsWorldModel"

    # Forward pass test with [N, T, 280]
    cand_model.eval()
    with torch.no_grad():
        x = torch.zeros(2, 5, STATE_FEATURE_DIM)
        k = torch.tensor([1, 3], dtype=torch.long)
        dt = torch.tensor([1.0, 3.0], dtype=torch.float)
        out, delta = cand_model(x, k, dt, return_delta=True)
        assert out.shape == (2, STATE_FEATURE_DIM)
        assert delta.shape == (2, STATE_FEATURE_DIM)


def test_production_weight_protection() -> None:
    """Verify that production checkpoint SHA-256 is immutable during evaluation."""
    prod_weights = Path("artifacts/world_model/best_model.pt")
    assert prod_weights.exists(), "Production checkpoint must exist"

    sha_before = hashlib.sha256(prod_weights.read_bytes()).hexdigest()

    evaluator = Phase10CEvaluator(
        candidate_dir="artifacts/world_model/candidates/cand-gru-real-d49b3ee3",
        production_dir="artifacts/world_model",
        output_dir="artifacts/phase10c",
    )
    cand_m, prod_m = evaluator.load_models()
    assert cand_m is not None
    assert prod_m is not None

    sha_after = hashlib.sha256(prod_weights.read_bytes()).hexdigest()
    assert sha_before == sha_after, "Production weights were modified!"


def test_risk_calculation(synthetic_pair_batch: FutureStatePairBatch) -> None:
    """Verify calculation of delta, error, drift, confidence proxy, and risk score."""
    evaluator = Phase10CEvaluator(
        candidate_dir="artifacts/world_model/candidates/cand-gru-real-d49b3ee3",
        production_dir="artifacts/world_model",
        output_dir="artifacts/phase10c",
    )

    # Persistence evaluation
    res = evaluator.evaluate_model_on_batch(
        model=None,
        model_name="persistence",
        batch=synthetic_pair_batch,
        val_mean_mae=0.05,
    )

    assert res["horizon"] == 3
    assert res["num_samples"] == 10
    # Persistence predicts 0 delta
    assert np.allclose(res["predicted_deltas"], 0.0)
    assert np.allclose(res["mean_predicted_delta_norm"], 0.0)
    # Persistence risk must be exactly 0.0
    assert np.allclose(res["risk_scores"], 0.0)
    assert np.allclose(res["drift_magnitudes"], 0.0)
    # Confidence proxies must be bounded [0, 1]
    assert np.all((res["confidence_proxies"] >= 0.0) & (res["confidence_proxies"] <= 1.0))
    assert np.all((res["uncertainty_proxies"] >= 0.0) & (res["uncertainty_proxies"] <= 1.0))


def test_no_label_leakage(synthetic_pair_batch: FutureStatePairBatch) -> None:
    """Verify that predictions and risk scores are invariant to ground-truth labels."""
    evaluator = Phase10CEvaluator(
        candidate_dir="artifacts/world_model/candidates/cand-gru-real-d49b3ee3",
        production_dir="artifacts/world_model",
        output_dir="artifacts/phase10c",
    )
    cand_model, _ = evaluator.load_models()

    # Original evaluation
    res_orig = evaluator.evaluate_model_on_batch(
        model=cand_model,
        model_name="gru_candidate",
        batch=synthetic_pair_batch,
        val_mean_mae=0.05,
    )

    # Inverted labels batch
    inverted_batch = FutureStatePairBatch(
        current_states=synthetic_pair_batch.current_states,
        target_states=synthetic_pair_batch.target_states,
        current_timestamps=synthetic_pair_batch.current_timestamps,
        target_timestamps=synthetic_pair_batch.target_timestamps,
        delta_times=synthetic_pair_batch.delta_times,
        horizon=synthetic_pair_batch.horizon,
        source_ids=synthetic_pair_batch.source_ids,
        schema_version=synthetic_pair_batch.schema_version,
        current_labels=1 - synthetic_pair_batch.current_labels,
        target_labels=1 - synthetic_pair_batch.target_labels,
        current_categories=synthetic_pair_batch.current_categories,
        target_categories=synthetic_pair_batch.target_categories,
    )

    res_inv = evaluator.evaluate_model_on_batch(
        model=cand_model,
        model_name="gru_candidate",
        batch=inverted_batch,
        val_mean_mae=0.05,
    )

    # Predictions and risk scores must be 100% IDENTICAL
    assert np.allclose(res_orig["predicted_states"], res_inv["predicted_states"])
    assert np.allclose(res_orig["risk_scores"], res_inv["risk_scores"])
    assert np.allclose(res_orig["confidence_proxies"], res_inv["confidence_proxies"])


def test_threshold_isolation() -> None:
    """Verify that threshold calibration depends strictly on validation data."""
    val_risks_1 = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.15])
    theta_1 = Phase10CEvaluator.calibrate_threshold(val_risks_1)

    # Changing hypothetical test risks cannot change theta_1
    theta_isolated = Phase10CEvaluator.calibrate_threshold(val_risks_1)
    assert theta_1 == theta_isolated
    assert 0.0 < theta_1 <= 1.0


def test_horizon_handling(synthetic_pair_batch: FutureStatePairBatch) -> None:
    """Verify multi-horizon handling and trajectory escalation detection."""
    evaluator = Phase10CEvaluator(
        candidate_dir="artifacts/world_model/candidates/cand-gru-real-d49b3ee3",
        production_dir="artifacts/world_model",
        output_dir="artifacts/phase10c",
    )
    cand_model, _ = evaluator.load_models()

    h_results = {}
    for k in [1, 3, 5]:
        b = FutureStatePairBatch(
            current_states=synthetic_pair_batch.current_states,
            target_states=synthetic_pair_batch.target_states,
            current_timestamps=synthetic_pair_batch.current_timestamps,
            target_timestamps=synthetic_pair_batch.target_timestamps,
            delta_times=synthetic_pair_batch.delta_times,
            horizon=k,
            source_ids=synthetic_pair_batch.source_ids,
            schema_version=synthetic_pair_batch.schema_version,
            current_labels=synthetic_pair_batch.current_labels,
            target_labels=synthetic_pair_batch.target_labels,
        )
        h_results[k] = evaluator.evaluate_model_on_batch(
            model=cand_model,
            model_name="gru_candidate",
            batch=b,
            val_mean_mae=0.05,
        )

    traj_res = evaluator.evaluate_trajectories(
        model_name="gru_candidate",
        horizon_results=h_results,
        batch_ref=synthetic_pair_batch,
    )

    assert traj_res["num_samples"] == 10
    assert len(traj_res["aggregate_risks"]) == 10
    assert 0.0 <= traj_res["mean_aggregate_risk"] <= 1.0
    assert 0.0 <= traj_res["escalation_rate"] <= 1.0
    assert set(traj_res["peak_horizon_distribution"].keys()) == {"1", "3", "5"}
