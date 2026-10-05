"""Tests for flow aggregator in CyberHeal AI."""

from pathlib import Path
import pytest

from shield_rca.telemetry.flow_aggregator import (
    FlowKey,
    FlowRecord,
    aggregate_flows,
)
from shield_rca.telemetry.pcap_parser import iter_packets
from tests.fixtures.generate_test_pcap import create_synthetic_pcap


@pytest.fixture(scope="module")
def synthetic_pcap(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("pcaps") / "test.pcap"
    return create_synthetic_pcap(tmp)


def test_aggregate_flows_bidirectional(synthetic_pcap: Path):
    packets = iter_packets(synthetic_pcap)
    flows = list(aggregate_flows(packets, mode="bidirectional"))

    # Exactly 4 bidirectional flows: HTTP, SSH, DNS, ICMP
    assert len(flows) == 4

    protocols = {f.key.protocol for f in flows}
    assert protocols == {1, 6, 17}

    for flow in flows:
        assert isinstance(flow, FlowRecord)
        assert len(flow.fwd_packets) > 0
        assert len(flow.bwd_packets) > 0
        assert flow.flow_end_time >= flow.flow_start_time


def test_flow_key_canonical():
    k1 = FlowKey("1.1.1.1", "2.2.2.2", 1000, 80, 6)
    k2 = FlowKey("2.2.2.2", "1.1.1.1", 80, 1000, 6)
    assert k1.canonical() == k2.canonical()
