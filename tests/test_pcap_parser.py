"""Tests for PCAP parsing in CyberHeal AI."""

from pathlib import Path
import pytest

from shield_rca.telemetry.pcap_parser import (
    RawPacketRecord,
    iter_packets,
    iter_packets_from_bytes,
)
from tests.fixtures.generate_test_pcap import create_synthetic_pcap


@pytest.fixture(scope="module")
def synthetic_pcap(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("pcaps") / "test.pcap"
    return create_synthetic_pcap(tmp)


def test_iter_packets_from_file(synthetic_pcap: Path):
    packets = list(iter_packets(synthetic_pcap))
    assert len(packets) == 18  # 7 (HTTP) + 7 (SSH) + 2 (DNS) + 2 (ICMP)
    for p in packets:
        assert isinstance(p, RawPacketRecord)
        assert p.protocol in (1, 6, 17)
        assert p.ip_version == 4
        assert p.timestamp > 0


def test_iter_packets_protocol_filter(synthetic_pcap: Path):
    # Only TCP
    tcp_packets = list(iter_packets(synthetic_pcap, include_protocols=(6,)))
    assert len(tcp_packets) == 14  # 7 HTTP + 7 SSH
    assert all(p.protocol == 6 for p in tcp_packets)

    # Only ICMP
    icmp_packets = list(iter_packets(synthetic_pcap, include_protocols=(1,)))
    assert len(icmp_packets) == 2
    assert all(p.protocol == 1 for p in icmp_packets)


def test_iter_packets_from_bytes(synthetic_pcap: Path):
    data = synthetic_pcap.read_bytes()
    packets = list(iter_packets_from_bytes(data))
    assert len(packets) == 18


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        list(iter_packets("non_existent_file.pcap"))
