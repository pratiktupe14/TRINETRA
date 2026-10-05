"""Tests for flow feature extraction in CyberHeal AI."""

from pathlib import Path
import numpy as np
import pytest

from shield_rca.telemetry.flow_aggregator import aggregate_flows
from shield_rca.telemetry.flow_features import (
    extract_flow_features,
    extract_feature_vector,
)
from shield_rca.telemetry.pcap_parser import iter_packets
from tests.fixtures.generate_test_pcap import create_synthetic_pcap


@pytest.fixture(scope="module")
def synthetic_pcap(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("pcaps") / "test.pcap"
    return create_synthetic_pcap(tmp)


def test_extract_flow_features(synthetic_pcap: Path):
    packets = iter_packets(synthetic_pcap)
    flows = list(aggregate_flows(packets))

    for flow in flows:
        feats = extract_flow_features(flow)
        assert isinstance(feats, dict)
        assert len(feats) >= 40

        # Check required categories
        assert "tot_fwd_pkts" in feats
        assert "tot_bwd_pkts" in feats
        assert "flow_duration_us" in feats
        assert "flow_bytes_per_s" in feats
        assert "syn_flag_cnt" in feats

        # Values should be finite
        for k, v in feats.items():
            if isinstance(v, (int, float)):
                assert np.isfinite(v), f"Feature {k} was not finite: {v}"


def test_extract_feature_vector(synthetic_pcap: Path):
    packets = iter_packets(synthetic_pcap)
    flows = list(aggregate_flows(packets))

    for flow in flows:
        vec, names = extract_feature_vector(flow)
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == len(names)
        assert np.all(np.isfinite(vec))
        assert "flow_id" not in names
        assert "src_ip" not in names
