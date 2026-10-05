"""Comprehensive tests for K-Step Future-State Dataset Construction in CyberHeal AI."""

from pathlib import Path
import numpy as np
import pytest

from shield_rca.network_state.future_dataset import (
    FutureStateDatasetBuilder,
    FutureStatePairBatch,
    MultiHorizonFutureDataset,
)
from shield_rca.network_state.preprocessor import NetworkStatePreprocessor
from shield_rca.network_state.quality_report import generate_future_dataset_quality_report
from shield_rca.network_state.state_encoder import NetworkStateBatch
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION


@pytest.fixture
def deterministic_known_states() -> NetworkStateBatch:
    """5 states with known sequential timestamps 10.0, 20.0, 30.0, 40.0, 50.0."""
    states = np.zeros((5, STATE_FEATURE_DIM), dtype=np.float32)
    for i in range(5):
        states[i, 0] = float(i * 10.0)  # state signature

    timestamps = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    starts = timestamps - 5.0
    ends = timestamps
    sources = ["scenario-A"] * 5
    feature_names = tuple(f"f_{i}" for i in range(STATE_FEATURE_DIM))

    return NetworkStateBatch(
        states=states,
        timestamps=timestamps,
        window_starts=starts,
        window_ends=ends,
        source_ids=sources,
        feature_names=feature_names,
        schema_version=STATE_SCHEMA_VERSION,
    )


@pytest.fixture
def multi_scenario_states() -> NetworkStateBatch:
    """States across two distinct scenarios A and B."""
    states = np.zeros((8, STATE_FEATURE_DIM), dtype=np.float32)
    # Scenario A: 4 states at times 10, 20, 30, 40
    # Scenario B: 4 states at times 15, 25, 35, 45
    for i in range(8):
        states[i, 0] = float(i)

    timestamps = np.array([10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0], dtype=float)
    starts = timestamps - 5.0
    ends = timestamps
    sources = ["scen_A", "scen_A", "scen_A", "scen_A", "scen_B", "scen_B", "scen_B", "scen_B"]
    feature_names = tuple(f"f_{i}" for i in range(STATE_FEATURE_DIM))

    return NetworkStateBatch(
        states=states,
        timestamps=timestamps,
        window_starts=starts,
        window_ends=ends,
        source_ids=sources,
        feature_names=feature_names,
        schema_version=STATE_SCHEMA_VERSION,
    )


def test_known_sequential_pairing(deterministic_known_states):
    """Prove that S_0, S_1, S_2, S_3, S_4 pairs exactly:
    k=1 -> S_0->S_1, S_1->S_2, S_2->S_3, S_3->S_4 (4 pairs, 1 dropped)
    k=3 -> S_0->S_3, S_1->S_4 (2 pairs, 3 dropped)
    """
    builder = FutureStateDatasetBuilder(horizons=[1, 3])
    ds = builder.build_future_pairs(deterministic_known_states)

    assert ds.total_states_input == 5
    assert ds.horizons == [1, 3]

    # Check k=1
    b1 = ds.get_horizon(1)
    assert b1.num_pairs == 4
    assert ds.dropped_tail_pairs[1] == 1
    # Check exact state pairings
    assert b1.current_states[0, 0] == 0.0 and b1.target_states[0, 0] == 10.0
    assert b1.current_states[1, 0] == 10.0 and b1.target_states[1, 0] == 20.0
    assert b1.current_states[2, 0] == 20.0 and b1.target_states[2, 0] == 30.0
    assert b1.current_states[3, 0] == 30.0 and b1.target_states[3, 0] == 40.0
    # Check timestamps and delta_t
    assert np.allclose(b1.delta_times, 10.0)

    # Check k=3
    b3 = ds.get_horizon(3)
    assert b3.num_pairs == 2
    assert ds.dropped_tail_pairs[3] == 3
    # S_0 -> S_3
    assert b3.current_states[0, 0] == 0.0 and b3.target_states[0, 0] == 30.0
    assert b3.current_timestamps[0] == 10.0 and b3.target_timestamps[0] == 40.0
    assert b3.delta_times[0] == 30.0
    # S_1 -> S_4
    assert b3.current_states[1, 0] == 10.0 and b3.target_states[1, 0] == 40.0
    assert b3.current_timestamps[1] == 20.0 and b3.target_timestamps[1] == 50.0
    assert b3.delta_times[1] == 30.0


def test_scenario_isolation(multi_scenario_states):
    """Ensure states from scen_A are NEVER paired with states from scen_B."""
    builder = FutureStateDatasetBuilder(horizons=[1, 2])
    ds = builder.build_future_pairs(multi_scenario_states)

    b1 = ds.get_horizon(1)
    # scen_A has 4 states -> 3 pairs; scen_B has 4 states -> 3 pairs; Total = 6 pairs
    assert b1.num_pairs == 6

    # Every pair must have matching source_ids and never cross
    for i in range(b1.num_pairs):
        src = b1.source_ids[i]
        assert src in ("scen_A", "scen_B")
        if src == "scen_A":
            # scen_A states have values in [0, 10, 20, 30]
            assert b1.current_states[i, 0] < 4.0
            assert b1.target_states[i, 0] < 4.0
        else:
            # scen_B states have values in [4, 5, 6, 7]
            assert b1.current_states[i, 0] >= 4.0
            assert b1.target_states[i, 0] >= 4.0


def test_no_future_target_before_current(multi_scenario_states):
    """delta_t must always be strictly positive."""
    builder = FutureStateDatasetBuilder(horizons=[1, 2, 3])
    ds = builder.build_future_pairs(multi_scenario_states)

    for k in (1, 2, 3):
        b = ds.get_horizon(k)
        assert np.all(b.delta_times > 0)
        assert np.all(b.target_timestamps > b.current_timestamps)


def test_partitioned_chronological_isolation():
    """Verify that TRAIN, VAL, and TEST datasets are strictly isolated:
    Zero boundary leakage across partitions.
    """
    # Create 3 sequential partitions
    train_states = NetworkStateBatch(
        states=np.ones((6, STATE_FEATURE_DIM), dtype=np.float32) * 1.0,
        timestamps=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        window_starts=np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        window_ends=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        source_ids=["live"] * 6,
        feature_names=tuple(f"f_{i}" for i in range(STATE_FEATURE_DIM)),
        schema_version=STATE_SCHEMA_VERSION,
    )
    val_states = NetworkStateBatch(
        states=np.ones((4, STATE_FEATURE_DIM), dtype=np.float32) * 2.0,
        timestamps=np.array([70.0, 80.0, 90.0, 100.0]),
        window_starts=np.array([65.0, 75.0, 85.0, 95.0]),
        window_ends=np.array([70.0, 80.0, 90.0, 100.0]),
        source_ids=["live"] * 4,
        feature_names=tuple(f"f_{i}" for i in range(STATE_FEATURE_DIM)),
        schema_version=STATE_SCHEMA_VERSION,
    )
    test_states = NetworkStateBatch(
        states=np.ones((4, STATE_FEATURE_DIM), dtype=np.float32) * 3.0,
        timestamps=np.array([110.0, 120.0, 130.0, 140.0]),
        window_starts=np.array([105.0, 115.0, 125.0, 135.0]),
        window_ends=np.array([110.0, 120.0, 130.0, 140.0]),
        source_ids=["live"] * 4,
        feature_names=tuple(f"f_{i}" for i in range(STATE_FEATURE_DIM)),
        schema_version=STATE_SCHEMA_VERSION,
    )

    builder = FutureStateDatasetBuilder(horizons=[1, 2])
    partitioned = builder.build_partitioned_future_datasets(
        train_states=train_states,
        val_states=val_states,
        test_states=test_states,
    )

    train_ds = partitioned["train"]
    val_ds = partitioned["val"]
    test_ds = partitioned["test"]

    # In train: all target states must have timestamp <= 60.0 (no validation at 70.0)
    b_train = train_ds.get_horizon(1)
    assert b_train.target_timestamps.max() <= 60.0
    assert np.all(b_train.target_states == 1.0)

    # In val: all current states >= 70.0 and targets <= 100.0 (no test at 110.0)
    b_val = val_ds.get_horizon(1)
    assert b_val.current_timestamps.min() >= 70.0
    assert b_val.target_timestamps.max() <= 100.0
    assert np.all(b_val.target_states == 2.0)

    # In test: all current states >= 110.0
    b_test = test_ds.get_horizon(1)
    assert b_test.current_timestamps.min() >= 110.0
    assert np.all(b_test.target_states == 3.0)


def test_multi_horizon_tensor(deterministic_known_states):
    """Test extracting multi-horizon tensor for multi-step rollout models."""
    builder = FutureStateDatasetBuilder(horizons=[1, 2])
    ds = builder.build_future_pairs(deterministic_known_states)

    X, Y, ts, dt, sources = ds.to_multi_horizon_tensor()
    assert X.shape == (3, STATE_FEATURE_DIM)
    assert Y.shape == (3, 2, STATE_FEATURE_DIM)  # [N_common, K=2, 280]
    assert dt.shape == (3, 2)
    assert len(ts) == 3


def test_irregular_timestamps():
    """Verify irregular wall-clock intervals are preserved accurately."""
    # Irregular timestamps: intervals of 5, 25, 100, 10
    timestamps = np.array([10.0, 15.0, 40.0, 140.0, 150.0])
    states = np.zeros((5, STATE_FEATURE_DIM), dtype=np.float32)
    batch = NetworkStateBatch(
        states=states,
        timestamps=timestamps,
        window_starts=timestamps - 2.0,
        window_ends=timestamps,
        source_ids=["sensor"] * 5,
        feature_names=tuple(f"f_{i}" for i in range(STATE_FEATURE_DIM)),
        schema_version=STATE_SCHEMA_VERSION,
    )

    builder = FutureStateDatasetBuilder(horizons=[1])
    ds = builder.build_future_pairs(batch)
    b = ds.get_horizon(1)

    expected_dts = [5.0, 25.0, 100.0, 10.0]
    assert np.allclose(b.delta_times, expected_dts)


def test_state_preprocessor(deterministic_known_states):
    """Test NetworkStatePreprocessor fitting strictly on train and transforming test."""
    train_states = deterministic_known_states.states[:3]
    test_states = deterministic_known_states.states[3:]

    preprocessor = NetworkStatePreprocessor(scaler_type="standard")
    assert not preprocessor.is_fitted

    preprocessor.fit(train_states)
    assert preprocessor.is_fitted

    scaled_train = preprocessor.transform(train_states)
    scaled_test = preprocessor.transform(test_states)

    assert scaled_train.shape == (3, STATE_FEATURE_DIM)
    assert scaled_test.shape == (2, STATE_FEATURE_DIM)
    assert np.all(np.isfinite(scaled_train))
    assert np.all(np.isfinite(scaled_test))


def test_quality_report(deterministic_known_states):
    """Verify comprehensive quality report generation."""
    builder = FutureStateDatasetBuilder(horizons=[1, 3])
    ds = builder.build_future_pairs(deterministic_known_states)

    report = generate_future_dataset_quality_report(ds, partition_name="test_run")
    assert report["partition"] == "test_run"
    assert report["total_states_input"] == 5
    assert report["horizons_evaluated"] == [1, 3]
    assert report["usable_pairs_per_horizon"]["horizon_1"] == 4
    assert report["usable_pairs_per_horizon"]["horizon_3"] == 2
    assert report["dropped_incomplete_pairs"]["horizon_1"] == 1
    assert report["dropped_incomplete_pairs"]["horizon_3"] == 3
    assert report["state_dimension"] == 280
    assert "elapsed_time_statistics" in report
    assert report["elapsed_time_statistics"]["min_delta_s"] == 10.0
