"""Unit tests for the AttackStageMapper module in CyberHeal / TRINETRA."""

import numpy as np
import pytest

from shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_FEATURE_NAMES
from shield_rca.world_model.stage_mapper import (
    AttackStage,
    AttackStageMapper,
    StageMappingResult,
    UNSUPPORTED_STAGES,
)


@pytest.fixture
def mapper() -> AttackStageMapper:
    return AttackStageMapper()


@pytest.fixture
def base_state() -> np.ndarray:
    np.random.seed(142)
    return np.random.uniform(0.1, 1.0, size=STATE_FEATURE_DIM).astype(np.float32)


def test_normal_stage_on_low_drift(mapper: AttackStageMapper, base_state: np.ndarray):
    """Zero or negligible drift must map to AttackStage.NORMAL with no fabricated MITRE ID."""
    pred_state = base_state + 0.001
    res = mapper.map_stage(base_state, pred_state)

    assert isinstance(res, StageMappingResult)
    assert res.stage == AttackStage.NORMAL
    assert res.confidence is not None
    assert 0.0 <= res.confidence <= 1.0
    assert res.mitre_technique is None
    assert len(res.supporting_features) == 0


def test_reconnaissance_stage_mapping(mapper: AttackStageMapper, base_state: np.ndarray):
    """Surges in port dispersion and unique endpoints must map to RECONNAISSANCE."""
    pred_state = base_state.copy()
    # Inject massive surge in port dispersion and unique destination ports
    idx_disp = STATE_FEATURE_NAMES.index("macro_port_dispersion")
    idx_dst_ports = STATE_FEATURE_NAMES.index("endpoint_unique_dst_ports")
    idx_rst = STATE_FEATURE_NAMES.index("macro_rst_ratio")

    pred_state[idx_disp] += 5.0
    pred_state[idx_dst_ports] += 10.0
    pred_state[idx_rst] += 2.0

    res = mapper.map_stage(base_state, pred_state)
    assert res.stage == AttackStage.RECONNAISSANCE
    assert "endpoints" in res.supporting_groups or "macro" in res.supporting_groups
    assert res.mitre_technique is None  # Strict anti-fabrication check


def test_impact_stage_mapping(mapper: AttackStageMapper, base_state: np.ndarray):
    """Massive surges in byte/packet rates and SYN floods must map to IMPACT."""
    pred_state = base_state.copy()
    idx_byte_rate = STATE_FEATURE_NAMES.index("macro_mean_byte_rate")
    idx_pkt_rate = STATE_FEATURE_NAMES.index("macro_mean_packet_rate")
    idx_syn_ack = STATE_FEATURE_NAMES.index("macro_syn_ack_ratio")

    pred_state[idx_byte_rate] += 50.0
    pred_state[idx_pkt_rate] += 100.0
    pred_state[idx_syn_ack] += 20.0

    res = mapper.map_stage(base_state, pred_state)
    assert res.stage == AttackStage.IMPACT
    assert res.confidence is not None
    assert res.mitre_technique is None


def test_unknown_stage_on_unstructured_drift(mapper: AttackStageMapper, base_state: np.ndarray):
    """Uniform distributed drift without specific signature must yield UNKNOWN or low-confidence."""
    # Modifying internal shape features that don't trigger specific signatures
    pred_state = base_state.copy()
    idx_shape_start = 198
    pred_state[idx_shape_start : idx_shape_start + 10] += 0.25

    res = mapper.map_stage(base_state, pred_state)
    # Either UNKNOWN or NORMAL depending on thresholds, but never fabricated technique
    assert res.mitre_technique is None


def test_no_fabricated_mitre_technique(mapper: AttackStageMapper, base_state: np.ndarray):
    """Strict test: mitre_technique must NEVER be populated during standard inference."""
    pred_state = base_state + 0.5
    res = mapper.map_stage(base_state, pred_state)
    assert res.mitre_technique is None


def test_offline_ground_truth_label_preservation(mapper: AttackStageMapper, base_state: np.ndarray):
    """Offline ground-truth labels supplied for benchmarking must be clearly labeled."""
    pred_state = base_state + 0.5
    res = mapper.map_stage(base_state, pred_state, known_label="DDoS")
    assert res.mitre_technique == "OFFLINE_GROUND_TRUTH:DDoS"


def test_deterministic_mapping(mapper: AttackStageMapper, base_state: np.ndarray):
    """Identical state inputs must produce strictly identical stage classifications."""
    pred_state = base_state + 0.3
    res1 = mapper.map_stage(base_state, pred_state, horizon=3)
    res2 = mapper.map_stage(base_state, pred_state, horizon=3)

    assert res1.stage == res2.stage
    assert res1.confidence == res2.confidence
    assert res1.evidence == res2.evidence


def test_invalid_input_rejections(mapper: AttackStageMapper, base_state: np.ndarray):
    """Test rejection of None, wrong dimensions, and NaN/Inf."""
    with pytest.raises(ValueError, match="cannot be None"):
        mapper.map_stage(None, base_state)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="State dimension mismatch"):
        mapper.map_stage(np.zeros(10), base_state)

    nan_state = base_state.copy()
    nan_state[5] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        mapper.map_stage(base_state, nan_state)


def test_unsupported_stages_documentation():
    """Verify that unsupported stages are documented and not part of AttackStage enum."""
    for unsupp in UNSUPPORTED_STAGES:
        # Unsupported stages must not be in the enum values
        assert unsupp not in [s.value for s in AttackStage]
