"""Unit tests for ForecastExplainabilityEngine in CyberHeal / TRINETRA."""

import numpy as np
import pytest

from shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_FEATURE_NAMES
from shield_rca.world_model.explainability import (
    FeatureChange,
    ForecastExplainabilityEngine,
    ForecastExplanation,
    GroupAttribution,
)


@pytest.fixture
def engine() -> ForecastExplainabilityEngine:
    return ForecastExplainabilityEngine(default_top_k=5)


@pytest.fixture
def base_state() -> np.ndarray:
    np.random.seed(999)
    return np.random.uniform(0.1, 10.0, size=STATE_FEATURE_DIM).astype(np.float32)


def test_zero_change_state(engine: ForecastExplainabilityEngine, base_state: np.ndarray):
    """Zero change between current and predicted states must yield zero drift and UNCHANGED."""
    res = engine.explain(base_state, base_state.copy(), horizon=1)

    assert isinstance(res, ForecastExplanation)
    assert res.total_drift_magnitude == 0.0
    assert len(res.top_features) == 5
    for feat in res.top_features:
        assert feat.magnitude == 0.0
        assert feat.direction == "UNCHANGED"
        assert feat.relative_share == 0.0
    assert "quiescent" in res.summary_text.lower()


def test_non_zero_state_change(engine: ForecastExplainabilityEngine, base_state: np.ndarray):
    """Test standard feature attribution with surges and drops."""
    pred_state = base_state.copy()
    pred_state[0] += 5.0  # active_max_mean surge
    pred_state[10] -= 2.0  # flow_iat_mean drop

    res = engine.explain(base_state, pred_state, horizon=1)
    assert res.total_drift_magnitude >= 7.0
    assert res.top_features[0].feature_name == STATE_FEATURE_NAMES[0]
    assert res.top_features[0].direction == "INCREASE"
    assert res.top_features[0].magnitude == pytest.approx(5.0, abs=1e-4)

    assert res.top_features[1].feature_name == STATE_FEATURE_NAMES[10]
    assert res.top_features[1].direction == "DECREASE"
    assert res.top_features[1].magnitude == pytest.approx(2.0, abs=1e-4)


def test_top_k_selection(engine: ForecastExplainabilityEngine, base_state: np.ndarray):
    """Verify top_k parameter selects exact requested count."""
    pred_state = base_state + 0.5
    res_k3 = engine.explain(base_state, pred_state, top_k=3)
    res_k8 = engine.explain(base_state, pred_state, top_k=8)

    assert len(res_k3.top_features) == 3
    assert len(res_k8.top_features) == 8


def test_group_attribution_completeness(engine: ForecastExplainabilityEngine, base_state: np.ndarray):
    """All 5 architectural groups must be accounted for and relative shares must sum to 1.0."""
    pred_state = base_state + 1.0
    res = engine.explain(base_state, pred_state, horizon=3)

    assert len(res.group_attributions) == 5
    group_names = {g.group_name for g in res.group_attributions}
    assert group_names == {"continuous", "flags", "shape", "endpoints", "macro"}

    # Relative shares sum to approximately 1.0
    total_share = sum(g.relative_share for g in res.group_attributions)
    assert pytest.approx(total_share, abs=1e-4) == 1.0

    # Absolute sums add up to total_drift_magnitude
    total_group_abs = sum(g.absolute_delta_sum for g in res.group_attributions)
    assert pytest.approx(total_group_abs, abs=1e-4) == res.total_drift_magnitude


def test_deterministic_ranking_and_stable_tie_breaking(
    engine: ForecastExplainabilityEngine, base_state: np.ndarray
):
    """Identical state changes across multiple features must tie-break deterministically."""
    cur_state = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    pred_state = cur_state.copy()
    # Exactly identical delta on feature 5 and feature 8
    pred_state[5] = 3.0
    pred_state[8] = 3.0

    res1 = engine.explain(cur_state, pred_state)
    res2 = engine.explain(cur_state, pred_state)

    # Identical ranking across executions
    assert [f.feature_name for f in res1.top_features] == [f.feature_name for f in res2.top_features]
    # Feature 5 should precede Feature 8 by deterministic index tie-break
    assert res1.top_features[0].feature_name == STATE_FEATURE_NAMES[5]
    assert res1.top_features[1].feature_name == STATE_FEATURE_NAMES[8]


def test_serialization_to_dict(engine: ForecastExplainabilityEngine, base_state: np.ndarray):
    """Ensure explanation object is cleanly serializable to dictionary for UI/API consumption."""
    pred_state = base_state + 0.2
    res = engine.explain(base_state, pred_state)
    d = res.to_dict()

    assert isinstance(d, dict)
    assert d["explanation_method"] == "State Delta Attribution"
    assert "top_features" in d
    assert "group_attributions" in d
    assert "summary_text" in d


def test_invalid_inputs_rejection(engine: ForecastExplainabilityEngine, base_state: np.ndarray):
    """Verify fail-loud behavior on invalid input parameters, dimensions, and NaN/Inf."""
    with pytest.raises(ValueError, match="default_top_k must be >= 1"):
        ForecastExplainabilityEngine(default_top_k=0)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        engine.explain(base_state, base_state, top_k=0)

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        engine.explain(base_state, base_state, horizon=0)

    with pytest.raises(ValueError, match="State dimension mismatch"):
        engine.explain(np.zeros(50), base_state)

    nan_state = base_state.copy()
    nan_state[0] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        engine.explain(nan_state, base_state)
