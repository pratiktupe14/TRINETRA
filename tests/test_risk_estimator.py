"""Unit tests for the PredictiveRiskEstimator module in CyberHeal / TRINETRA."""

import numpy as np
import pytest

from shield_rca.network_state.state_schema import STATE_FEATURE_DIM
from shield_rca.world_model.risk_estimator import (
    HorizonRiskEstimate,
    PredictiveRiskEstimator,
    RiskTrajectory,
)


@pytest.fixture
def estimator() -> PredictiveRiskEstimator:
    return PredictiveRiskEstimator()


@pytest.fixture
def base_state() -> np.ndarray:
    np.random.seed(42)
    return np.random.randn(STATE_FEATURE_DIM).astype(np.float32)


def test_valid_state_input(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Test standard evaluation with valid state vectors."""
    pred_state = base_state + 0.1
    res = estimator.estimate_horizon_risk(base_state, pred_state, horizon=1, delta_t=2.0)

    assert isinstance(res, HorizonRiskEstimate)
    assert res.horizon == 1
    assert res.delta_t == 2.0
    assert 0.0 <= res.risk_score <= 1.0
    assert 0.0 <= res.infiltration_score <= 1.0
    assert res.confidence is None


def test_zero_state_change(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Zero state change must yield zero risk and zero infiltration score."""
    res = estimator.estimate_horizon_risk(base_state, base_state.copy(), horizon=1)
    assert res.risk_score == 0.0
    assert res.infiltration_score == 0.0
    assert res.state_change.total_l1_drift == 0.0
    assert res.state_change.total_l2_drift == 0.0
    assert res.details["is_zero_drift"] is True


def test_small_vs_large_state_change(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Monotonic property: larger state changes should produce strictly greater risk."""
    small_change = base_state + 0.02
    large_change = base_state + 2.50

    res_small = estimator.estimate_horizon_risk(base_state, small_change, horizon=1)
    res_large = estimator.estimate_horizon_risk(base_state, large_change, horizon=1)

    assert 0.0 < res_small.risk_score < res_large.risk_score <= 1.0
    assert 0.0 < res_small.infiltration_score <= res_large.infiltration_score <= 1.0


def test_bounded_output(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Massive extreme state change must saturate strictly at <= 1.0."""
    massive_change = base_state + 1e6
    res = estimator.estimate_horizon_risk(base_state, massive_change, horizon=1)

    assert res.risk_score <= 1.0
    assert res.infiltration_score <= 1.0
    assert np.isfinite(res.risk_score)
    assert np.isfinite(res.infiltration_score)


def test_deterministic_output(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Repeated calls with identical inputs must produce exact identical numerical results."""
    pred_state = base_state + 0.35
    res1 = estimator.estimate_horizon_risk(base_state, pred_state, horizon=3, delta_t=5.0)
    res2 = estimator.estimate_horizon_risk(base_state, pred_state, horizon=3, delta_t=5.0)

    assert res1.risk_score == res2.risk_score
    assert res1.infiltration_score == res2.infiltration_score
    assert res1.state_change.total_l1_drift == res2.state_change.total_l1_drift


def test_invalid_shape_rejection(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Input arrays with wrong shapes must be rejected with ValueError."""
    wrong_1d = np.zeros(100, dtype=np.float32)
    wrong_2d = np.zeros((2, STATE_FEATURE_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="dimension mismatch"):
        estimator.estimate_horizon_risk(wrong_1d, base_state)

    with pytest.raises(ValueError, match="must be a 1D vector"):
        estimator.estimate_horizon_risk(wrong_2d, base_state)


def test_nan_and_inf_rejection(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """States containing NaN or Inf must be rejected with ValueError."""
    nan_state = base_state.copy()
    nan_state[10] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        estimator.estimate_horizon_risk(base_state, nan_state)

    inf_state = base_state.copy()
    inf_state[20] = np.inf
    with pytest.raises(ValueError, match="contains non-finite values"):
        estimator.estimate_horizon_risk(inf_state, base_state)


def test_empty_input_rejection(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """None inputs must be rejected with ValueError."""
    with pytest.raises(ValueError, match="cannot be None"):
        estimator.estimate_horizon_risk(None, base_state)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot be None"):
        estimator.estimate_horizon_risk(base_state, None)  # type: ignore[arg-type]


def test_confidence_handling(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Confidence must be preserved if valid, or rejected if out of bounds."""
    pred_state = base_state + 0.1
    res = estimator.estimate_horizon_risk(base_state, pred_state, model_confidence=0.88)
    assert res.confidence == 0.88

    with pytest.raises(ValueError, match="model_confidence must be in"):
        estimator.estimate_horizon_risk(base_state, pred_state, model_confidence=1.5)


def test_trajectory_generation(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Multi-horizon trajectory evaluation over k=1, 3, 5."""
    pred_states = {
        1: base_state + 0.05,
        3: base_state + 0.20,
        5: base_state + 0.60,
    }
    dts = {1: 2.0, 3: 6.0, 5: 10.0}

    traj = estimator.estimate_trajectory_risk(base_state, pred_states, delta_times=dts)
    assert isinstance(traj, RiskTrajectory)
    assert traj.peak_horizon == 5
    assert traj.is_escalating is True
    assert 0.0 <= traj.aggregate_risk <= 1.0
    assert len(traj.horizon_estimates) == 3


def test_trajectory_empty_rejection(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Trajectory estimator must reject empty predicted states."""
    with pytest.raises(ValueError, match="predicted_states cannot be empty"):
        estimator.estimate_trajectory_risk(base_state, {})


def test_invalid_horizon_and_delta_t(estimator: PredictiveRiskEstimator, base_state: np.ndarray):
    """Horizon < 1 or delta_t < 0 must be rejected."""
    with pytest.raises(ValueError, match="Horizon k must be >= 1"):
        estimator.estimate_horizon_risk(base_state, base_state, horizon=0)

    with pytest.raises(ValueError, match="Elapsed time delta_t must be >= 0.0"):
        estimator.estimate_horizon_risk(base_state, base_state, delta_t=-1.0)
