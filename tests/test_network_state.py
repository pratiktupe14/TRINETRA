"""Comprehensive tests for Network State Representation in CyberHeal AI."""

from pathlib import Path
import numpy as np
import pytest

from shield_rca.network_state.state_encoder import (
    NetworkState,
    NetworkStateBatch,
    NetworkStateEncoder,
)
from shield_rca.network_state.state_schema import (
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
    get_state_schema_metadata,
)
from shield_rca.telemetry.dataset_builder import WindowBatch


@pytest.fixture
def mock_window() -> np.ndarray:
    """Deterministic [20, 78] window."""
    np.random.seed(42)
    return np.random.uniform(1.0, 50.0, size=(20, 78)).astype(np.float32)


@pytest.fixture
def mock_batch() -> WindowBatch:
    """Mock WindowBatch containing 5 windows of shape [5, 20, 78]."""
    np.random.seed(123)
    X = np.random.uniform(1.0, 50.0, size=(5, 20, 78)).astype(np.float32)
    y = np.array([0, 0, 1, 2, 0], dtype=np.int64)
    starts = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    ends = np.array([120.0, 130.0, 140.0, 150.0, 160.0])
    sources = ["node-1", "node-1", "node-2", "node-2", "node-3"]
    feature_names = [f"f_{i}" for i in range(78)]

    return WindowBatch(
        X=X,
        y=y,
        window_start_times=starts,
        window_end_times=ends,
        source_ids=sources,
        feature_names=feature_names,
        schema_version="1.0.0",
    )


def test_schema_metadata():
    meta = get_state_schema_metadata()
    assert meta.version == STATE_SCHEMA_VERSION
    assert meta.dimension == 280
    assert len(STATE_FEATURE_NAMES) == 280
    assert STATE_FEATURE_DIM == 280


def test_encode_single_window(mock_window):
    encoder = NetworkStateEncoder()
    state = encoder.encode_window(
        mock_window,
        timestamp=100.0,
        window_start=80.0,
        window_end=100.0,
        source_id="test-host",
    )

    assert isinstance(state, NetworkState)
    assert state.dimension == 280
    assert state.state_vector.shape == (280,)
    assert state.state_vector.dtype == np.float32
    assert np.all(np.isfinite(state.state_vector))
    assert state.timestamp == 100.0
    assert state.window_start == 80.0
    assert state.window_end == 100.0
    assert state.source_id == "test-host"
    assert state.schema_version == STATE_SCHEMA_VERSION
    assert isinstance(state.macro_metrics, dict)
    assert len(state.macro_metrics) == 10


def test_encode_batch(mock_batch):
    encoder = NetworkStateEncoder()
    batch_state = encoder.encode_batch(mock_batch)

    assert isinstance(batch_state, NetworkStateBatch)
    assert batch_state.num_states == 5
    assert batch_state.state_dim == 280
    assert batch_state.states.shape == (5, 280)
    assert batch_state.states.dtype == np.float32
    assert np.all(np.isfinite(batch_state.states))
    assert len(batch_state.timestamps) == 5
    assert len(batch_state.source_ids) == 5
    assert batch_state.schema_version == STATE_SCHEMA_VERSION


def test_batch_single_consistency(mock_batch):
    """Batch encoding must produce exact same state vectors as single window encoding."""
    encoder = NetworkStateEncoder()
    batch_state = encoder.encode_batch(mock_batch)

    for i in range(mock_batch.num_windows):
        single_state = encoder.encode_window(
            mock_batch.X[i],
            timestamp=mock_batch.window_end_times[i],
            window_start=mock_batch.window_start_times[i],
            window_end=mock_batch.window_end_times[i],
            source_id=mock_batch.source_ids[i],
        )
        assert np.allclose(batch_state.states[i], single_state.state_vector, atol=1e-6)


def test_determinism(mock_window):
    """Identical input windows must produce bit-identical states."""
    encoder = NetworkStateEncoder()
    state1 = encoder.encode_window(mock_window)
    state2 = encoder.encode_window(mock_window)

    assert np.array_equal(state1.state_vector, state2.state_vector)


def test_no_labels_in_state(mock_batch):
    """Verify ground-truth labels are strictly excluded from state vector."""
    encoder = NetworkStateEncoder()
    batch_state = encoder.encode_batch(mock_batch)

    # State vector length is exactly 280, matching STATE_FEATURE_NAMES
    assert batch_state.state_dim == len(STATE_FEATURE_NAMES)
    for name in STATE_FEATURE_NAMES:
        assert "label" not in name.lower()
        assert "attack_class" not in name.lower()


def test_no_future_leakage(mock_batch):
    """Modifying a subsequent (future) window must NOT affect the state of earlier windows."""
    encoder = NetworkStateEncoder()
    batch_state_orig = encoder.encode_batch(mock_batch)

    # Corrupt future window 4 (last window)
    corrupted_X = mock_batch.X.copy()
    corrupted_X[4] += 1000.0  # Massive change in future window

    corrupted_batch = WindowBatch(
        X=corrupted_X,
        y=mock_batch.y,
        window_start_times=mock_batch.window_start_times,
        window_end_times=mock_batch.window_end_times,
        source_ids=mock_batch.source_ids,
        feature_names=mock_batch.feature_names,
        schema_version=mock_batch.schema_version,
    )

    batch_state_corrupted = encoder.encode_batch(corrupted_batch)

    # Earlier windows 0, 1, 2, 3 must be strictly identical
    for i in range(4):
        assert np.array_equal(batch_state_orig.states[i], batch_state_corrupted.states[i])

    # Window 4 must be different
    assert not np.array_equal(batch_state_orig.states[4], batch_state_corrupted.states[4])


def test_reject_nan_and_inf():
    encoder = NetworkStateEncoder()
    bad_nan = np.zeros((10, 78), dtype=np.float32)
    bad_nan[2, 5] = np.nan
    with pytest.raises(ValueError, match="NaN or Infinite"):
        encoder.encode_window(bad_nan)

    bad_inf = np.zeros((10, 78), dtype=np.float32)
    bad_inf[2, 5] = np.inf
    with pytest.raises(ValueError, match="NaN or Infinite"):
        encoder.encode_window(bad_inf)


def test_reject_wrong_dimensions():
    encoder = NetworkStateEncoder()

    # Wrong feature dimension F=50 instead of 78
    bad_F = np.zeros((10, 50), dtype=np.float32)
    with pytest.raises(ValueError, match="dimension F must be 78"):
        encoder.encode_window(bad_F)

    # Empty window T=0
    bad_T = np.zeros((0, 78), dtype=np.float32)
    with pytest.raises(ValueError, match="length T must be >= 1"):
        encoder.encode_window(bad_T)

    # Wrong tensor ndim (3D passed to single window encoder)
    bad_3d = np.zeros((2, 10, 78), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2D"):
        encoder.encode_window(bad_3d)
