"""Comprehensive tests for the CyberHeal World Model."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from shield_rca.network_state.future_dataset import (
    FutureStateDatasetBuilder,
    FutureStatePairBatch,
    MultiHorizonFutureDataset,
)
from shield_rca.network_state.preprocessor import NetworkStatePreprocessor
from shield_rca.network_state.state_encoder import NetworkStateBatch
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from shield_rca.telemetry.dataset_builder import DatasetBuilder
from shield_rca.telemetry.feature_schema import CANONICAL_FEATURES
from shield_rca.telemetry.split import chronological_split
from shield_rca.world_model.baselines import LinearDynamicsBaseline, PersistenceBaseline
from shield_rca.world_model.checkpoint import (
    load_world_model_checkpoint,
    save_world_model_checkpoint,
)
from shield_rca.world_model.evaluator import WorldModelEvaluator
from shield_rca.world_model.inference import WorldModelInference
from shield_rca.world_model.loss import WorldModelLoss
from shield_rca.world_model.model import StateResidualDynamicsModel
from shield_rca.world_model.time_conditioning import TimeHorizonConditioner
from shield_rca.world_model.trainer import WorldModelTrainer


@pytest.fixture
def dummy_pair_batch() -> FutureStatePairBatch:
    """Fixture with 10 synthetic state pairs."""
    np.random.seed(42)
    current = np.random.randn(10, STATE_FEATURE_DIM).astype(np.float32)
    # Target = current + small shift
    target = (current + 0.1 * np.random.randn(10, STATE_FEATURE_DIM)).astype(np.float32)
    cur_ts = np.arange(10, dtype=float) * 10.0
    targ_ts = cur_ts + 10.0
    dts = np.full(10, 10.0, dtype=float)

    return FutureStatePairBatch(
        current_states=current,
        target_states=target,
        current_timestamps=cur_ts,
        target_timestamps=targ_ts,
        delta_times=dts,
        horizon=1,
        source_ids=["host-1"] * 10,
        schema_version=STATE_SCHEMA_VERSION,
    )


@pytest.fixture
def deterministic_toy_dataset() -> tuple[MultiHorizonFutureDataset, MultiHorizonFutureDataset]:
    """Deterministic linear transition dataset: S_{t+1} = 0.8 * S_t + 0.2 * Delta_tau.

    Labeled strictly as SYNTHETIC SANITY CHECK.
    """
    np.random.seed(123)
    N_tr = 40
    N_va = 15

    def make_data(n: int, start_time: float) -> MultiHorizonFutureDataset:
        curr = np.random.randn(n, STATE_FEATURE_DIM).astype(np.float32)
        dts = np.random.uniform(1.0, 5.0, size=n).astype(np.float32)
        # Known ground-truth dynamic: S_{t+1} = 0.8 * S_t + 0.2 * dt
        targ = 0.8 * curr + 0.2 * dts[:, None]

        cur_ts = start_time + np.arange(n) * 10.0
        targ_ts = cur_ts + dts

        batch_1 = FutureStatePairBatch(
            current_states=curr,
            target_states=targ,
            current_timestamps=cur_ts,
            target_timestamps=targ_ts,
            delta_times=dts,
            horizon=1,
            source_ids=["toy"] * n,
            schema_version=STATE_SCHEMA_VERSION,
        )
        return MultiHorizonFutureDataset(
            batches={1: batch_1},
            horizons=[1],
            total_states_input=n + 1,
            dropped_tail_pairs={1: 1},
            schema_version=STATE_SCHEMA_VERSION,
        )

    train_ds = make_data(N_tr, 0.0)
    val_ds = make_data(N_va, 1000.0)
    return train_ds, val_ds


def test_model_input_output_dimensions():
    """Verify forward pass preserves exact state dimension [N, 280] -> [N, 280]."""
    model = StateResidualDynamicsModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64, num_blocks=2)
    N = 8
    states = torch.randn(N, STATE_FEATURE_DIM)
    horizons = torch.tensor([1, 2, 3, 5, 1, 3, 5, 2], dtype=torch.long)
    dts = torch.tensor([1.0, 3.2, 5.0, 10.5, 2.0, 6.0, 12.0, 4.0], dtype=torch.float)

    out = model(states, horizons, dts)
    assert out.shape == (N, STATE_FEATURE_DIM)
    assert not torch.isnan(out).any()


def test_horizon_conditioning_sensitivity():
    """Verify that different horizons k=1 vs k=5 produce distinct state predictions."""
    model = StateResidualDynamicsModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64)
    model.eval()
    state = torch.randn(1, STATE_FEATURE_DIM)
    dt = torch.tensor([5.0])

    with torch.no_grad():
        out_k1 = model(state, torch.tensor([1]), dt)
        out_k5 = model(state, torch.tensor([5]), dt)

    assert not torch.allclose(out_k1, out_k5, atol=1e-5)


def test_delta_time_conditioning_sensitivity():
    """Verify that different elapsed times dt=1.0 vs dt=100.0 produce distinct predictions."""
    model = StateResidualDynamicsModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64)
    model.eval()
    state = torch.randn(1, STATE_FEATURE_DIM)
    k = torch.tensor([1])

    with torch.no_grad():
        out_dt1 = model(state, k, torch.tensor([1.0]))
        out_dt100 = model(state, k, torch.tensor([100.0]))

    assert not torch.allclose(out_dt1, out_dt100, atol=1e-5)


def test_deterministic_inference():
    """Verify identical inputs produce bit-identical predictions with fixed weights."""
    model = StateResidualDynamicsModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64)
    inference = WorldModelInference(model)
    state = np.random.randn(STATE_FEATURE_DIM).astype(np.float32)

    res1 = inference.predict(state, horizon=3, delta_t=12.5)
    res2 = inference.predict(state, horizon=3, delta_t=12.5)

    np.testing.assert_array_equal(res1["predicted_state"], res2["predicted_state"])


def test_persistence_baseline(dummy_pair_batch):
    """Verify persistence baseline computes exact MAE with S_hat_{t+k} = S_t."""
    baseline = PersistenceBaseline()
    metrics = baseline.evaluate(dummy_pair_batch)

    expected_diff = np.abs(dummy_pair_batch.current_states - dummy_pair_batch.target_states)
    assert np.isclose(metrics["mae"], float(np.mean(expected_diff)))
    assert np.isclose(metrics["rmse"], float(np.sqrt(np.mean(expected_diff ** 2))))


def test_checkpoint_save_and_load(tmp_path):
    """Verify model state_dict and configuration are saved and restored identically."""
    model = StateResidualDynamicsModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64, num_blocks=1)
    config = {"state_dim": STATE_FEATURE_DIM, "hidden_dim": 64, "num_blocks": 1, "max_horizon": 16}

    save_dir = tmp_path / "wm_ckpt"
    save_world_model_checkpoint(model, save_dir, config, {"best_val_loss": 0.123})

    loaded_model, loaded_cfg, manifest = load_world_model_checkpoint(save_dir)
    assert loaded_cfg["hidden_dim"] == 64
    assert manifest["best_val_loss"] == 0.123

    # Test forward equality
    s = torch.randn(2, STATE_FEATURE_DIM)
    k = torch.tensor([1, 2])
    dt = torch.tensor([1.0, 2.0])

    model.eval()
    with torch.no_grad():
        out_orig = model(s, k, dt)
        out_loaded = loaded_model(s, k, dt)

    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_overlapping_window_raw_observation_isolation():
    """Regression Test for Part B: Overlapping-window isolation.

    Proves that when raw chronological splitting is performed before windowing,
    the set of raw flow observation IDs appearing in TRAIN windows has ZERO
    intersection with those in TEST windows.
    """
    # Create synthetic raw sequence of 100 flows with explicit unique flow_ids
    N_raw = 100
    df_raw = pd.DataFrame({
        "flow_id": [f"flow_{i}" for i in range(N_raw)],
        "flow_start_time": np.arange(N_raw, dtype=float) * 5.0,
        "scenario_id": ["session-1"] * N_raw,
        "label_int": [0] * N_raw,
    })
    for f in CANONICAL_FEATURES:
        df_raw[f] = np.random.uniform(1.0, 100.0, size=N_raw).astype(np.float32)

    # 1. Chronological split of raw sequence
    splits = chronological_split(df_raw, time_col="flow_start_time", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_raw = splits.train
    test_raw = splits.test

    train_flow_ids = set(train_raw["flow_id"])
    test_flow_ids = set(test_raw["flow_id"])

    # 2. Build windows INSIDE each partition (window_size=20, stride=10, 50% overlap)
    builder = DatasetBuilder(window_size=20, stride=10)
    train_windows = builder.build_windows(train_raw, group_key="scenario_id", time_col="flow_start_time")
    test_windows = builder.build_windows(test_raw, group_key="scenario_id", time_col="flow_start_time")

    # 3. Assert zero leakage between raw flow partitions
    intersection = train_flow_ids.intersection(test_flow_ids)
    assert len(intersection) == 0, f"Leakage detected! Shared flow IDs: {intersection}"
    assert train_windows.num_windows > 0
    assert test_windows.num_windows > 0


def test_synthetic_sanity_check_training(deterministic_toy_dataset):
    """Synthetic Sanity Check (Parts N & O):

    Trains the World Model on a known deterministic transition (S_{t+1} = 0.8 * S_t + 0.2 * dt).
    Verifies that:
    1. Training Huber loss decreases.
    2. Model beats persistence baseline on this known transition.
    """
    train_ds, val_ds = deterministic_toy_dataset
    model = StateResidualDynamicsModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64, num_blocks=2, dropout=0.0)

    trainer = WorldModelTrainer(
        model=model,
        learning_rate=3e-3,
        batch_size=16,
        max_epochs=15,
        patience=5,
        seed=42,
    )
    history = trainer.fit(train_ds, val_ds)

    assert len(history) > 1
    # 1. Training loss decreased
    first_loss = history[0]["train_huber"]
    final_loss = history[-1]["train_huber"]
    assert final_loss < first_loss, f"Loss did not decrease: {first_loss} -> {final_loss}"

    # 2. Model outperforms persistence baseline on validation set
    evaluator = WorldModelEvaluator(model)
    res = evaluator.evaluate_batch(val_ds.batches[1])

    wm_mae = res["world_model"]["mae"]
    pers_mae = res["persistence_baseline"]["mae"]
    assert wm_mae < pers_mae, f"World model MAE ({wm_mae:.4f}) should beat persistence ({pers_mae:.4f})"


def test_inference_rejection_of_invalid_inputs():
    """Verify inference API rejects wrong dimensions, NaNs, and negative time."""
    model = StateResidualDynamicsModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64)
    inference = WorldModelInference(model)

    # 1. Wrong dimension
    with pytest.raises(ValueError, match="State dimension mismatch"):
        inference.predict(np.zeros(100, dtype=np.float32))

    # 2. NaN value
    nan_state = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    nan_state[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        inference.predict(nan_state)

    # 3. Negative delta_t
    with pytest.raises(ValueError, match="delta_t must be >= 0"):
        inference.predict(np.zeros(STATE_FEATURE_DIM, dtype=np.float32), delta_t=-2.0)

    # 4. Invalid horizon
    with pytest.raises(ValueError, match="Horizon k must be >= 1"):
        inference.predict(np.zeros(STATE_FEATURE_DIM, dtype=np.float32), horizon=0)
