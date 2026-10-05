"""Tests for Packet-Level Features and Signal Enrichment (Phase 8).

Validates all 32 required tests:
1. IPv4 TTL extraction
2. IPv6 hop-limit handling
3. TTL min/max/mean
4. TTL standard deviation
5. TTL range
6. Missing TTL handling
7. TCP retransmission detection
8. Duplicate/overlapping sequence handling
9. Retransmission rate calculation
10. ACK-only packet handling
11. Insufficient TCP metadata handling
12. Inter-arrival mean
13. Inter-arrival standard deviation
14. IAT min/max
15. IAT percentiles
16. IAT jitter
17. Timestamp ordering
18. Missing timestamp handling
19. Single-packet flow handling
20. UDP flow handling
21. ICMP/non-TCP handling
22. Mixed protocol handling
23. Directional feature correctness
24. Deterministic output
25. Feature schema ordering
26. Feature dimensionality
27. Existing 78-feature schema unchanged
28. Existing 280-D state unchanged
29. Existing World Model input compatibility
30. Malformed packet robustness
31. Empty packet list handling
32. Large-flow bounded processing sanity test
"""

from __future__ import annotations

import time
import numpy as np
import pytest
import torch

from shield_rca.network_state.state_schema import STATE_FEATURE_DIM
from shield_rca.telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES
from shield_rca.telemetry.flow_aggregator import FlowKey, FlowRecord
from shield_rca.telemetry.flow_features import extract_flow_features
from shield_rca.telemetry.packet_feature_schema import (
    MISSING_TTL_SENTINEL,
    PACKET_FEATURE_DIM,
    PACKET_FEATURE_NAMES,
    PACKET_SCHEMA_VERSION,
    PacketFeatureVector,
    validate_packet_features,
)
from shield_rca.telemetry.packet_features import (
    PacketFeatureExtractor,
    build_enriched_features,
    extract_packet_feature_vector,
    extract_packet_features,
)
from shield_rca.telemetry.pcap_parser import RawPacketRecord, _decode_packet
from shield_rca.world_model.model import StateResidualDynamicsModel


def _make_pkt(
    timestamp: float = 1000.0,
    src_ip: str = "192.168.1.10",
    dst_ip: str = "10.0.0.1",
    src_port: int = 54321,
    dst_port: int = 80,
    protocol: int = 6,
    ip_length: int = 60,
    payload_length: int = 0,
    tcp_flags: int = 0x10,
    ip_version: int = 4,
    ttl: int | None = 64,
    tcp_seq: int = 1000,
    tcp_ack: int = 500,
    tcp_window: int = 65535,
) -> RawPacketRecord:
    """Helper to construct synthetic RawPacketRecord instances."""
    return RawPacketRecord(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        ip_length=ip_length,
        payload_length=payload_length,
        tcp_flags=tcp_flags,
        ip_version=ip_version,
        ttl=ttl,
        tcp_seq=tcp_seq,
        tcp_ack=tcp_ack,
        tcp_window=tcp_window,
    )


# ── Test 1: IPv4 TTL extraction ───────────────────────────────────────────────
def test_ipv4_ttl_extraction():
    from scapy.layers.inet import IP, TCP

    pkt = IP(src="1.1.1.1", dst="2.2.2.2", ttl=64) / TCP(sport=1234, dport=80, seq=100)
    record = _decode_packet(pkt, frozenset({1, 6, 17}), frozenset({4, 6}))
    assert record is not None
    assert record.ttl == 64
    assert record.ip_version == 4


# ── Test 2: IPv6 hop-limit handling ───────────────────────────────────────────
def test_ipv6_hop_limit_handling():
    from scapy.layers.inet6 import IPv6
    from scapy.layers.inet import TCP

    pkt = IPv6(src="2001:db8::1", dst="2001:db8::2", hlim=128) / TCP(sport=1234, dport=80, seq=100)
    record = _decode_packet(pkt, frozenset({1, 6, 17}), frozenset({4, 6}))
    assert record is not None
    assert record.ttl == 128
    assert record.ip_version == 6


# ── Test 3: TTL min/max/mean ──────────────────────────────────────────────────
def test_ttl_min_max_mean():
    pkts = [
        _make_pkt(ttl=60),
        _make_pkt(ttl=64),
        _make_pkt(ttl=128),
    ]
    feats = extract_packet_features(pkts)
    assert feats["ttl_min"] == 60.0
    assert feats["ttl_max"] == 128.0
    assert pytest.approx(feats["ttl_mean"], abs=1e-4) == (60.0 + 64.0 + 128.0) / 3.0


# ── Test 4: TTL standard deviation ────────────────────────────────────────────
def test_ttl_standard_deviation():
    # Identical TTLs -> std is 0.0
    pkts_same = [_make_pkt(ttl=64), _make_pkt(ttl=64), _make_pkt(ttl=64)]
    feats_same = extract_packet_features(pkts_same)
    assert feats_same["ttl_std"] == 0.0

    # Known variance: [50, 70] -> mean=60, std=10
    pkts_var = [_make_pkt(ttl=50), _make_pkt(ttl=70)]
    feats_var = extract_packet_features(pkts_var)
    assert pytest.approx(feats_var["ttl_std"], abs=1e-4) == 10.0


# ── Test 5: TTL range ─────────────────────────────────────────────────────────
def test_ttl_range():
    pkts = [_make_pkt(ttl=40), _make_pkt(ttl=100)]
    feats = extract_packet_features(pkts)
    assert feats["ttl_range"] == 60.0


# ── Test 6: Missing TTL handling ──────────────────────────────────────────────
def test_missing_ttl_handling():
    pkts = [_make_pkt(ttl=None), _make_pkt(ttl=None)]
    feats = extract_packet_features(pkts)
    assert feats["ttl_min"] == MISSING_TTL_SENTINEL
    assert feats["ttl_max"] == MISSING_TTL_SENTINEL
    assert feats["ttl_mean"] == MISSING_TTL_SENTINEL
    assert feats["ttl_std"] == MISSING_TTL_SENTINEL
    assert feats["ttl_range"] == MISSING_TTL_SENTINEL
    assert feats["fwd_ttl_mean"] == MISSING_TTL_SENTINEL
    assert feats["bwd_ttl_mean"] == MISSING_TTL_SENTINEL


# ── Test 7: TCP retransmission detection ──────────────────────────────────────
def test_tcp_retransmission_detection():
    # Two packets in the same direction with exact same sequence number and payload
    pkts = [
        _make_pkt(timestamp=1.0, tcp_seq=1000, payload_length=100),
        _make_pkt(timestamp=1.1, tcp_seq=1000, payload_length=100),
    ]
    feats = extract_packet_features(pkts)
    assert feats["tcp_retransmission_count"] == 1.0


# ── Test 8: Duplicate/overlapping sequence handling ───────────────────────────
def test_duplicate_overlapping_sequence_handling():
    # Packet 1: [1000, 1100), Packet 2: [1050, 1150) (partially overlaps)
    pkts = [
        _make_pkt(timestamp=1.0, tcp_seq=1000, payload_length=100),
        _make_pkt(timestamp=1.1, tcp_seq=1050, payload_length=100),
    ]
    feats = extract_packet_features(pkts)
    assert feats["tcp_retransmission_count"] == 1.0


# ── Test 9: Retransmission rate calculation ───────────────────────────────────
def test_retransmission_rate_calculation():
    # 4 TCP packets: 2 unique data packets, 2 retransmissions
    pkts = [
        _make_pkt(timestamp=1.0, tcp_seq=1000, payload_length=50),
        _make_pkt(timestamp=1.1, tcp_seq=1050, payload_length=50),
        _make_pkt(timestamp=1.2, tcp_seq=1000, payload_length=50),  # retrans
        _make_pkt(timestamp=1.3, tcp_seq=1050, payload_length=50),  # retrans
    ]
    feats = extract_packet_features(pkts)
    assert feats["tcp_retransmission_count"] == 2.0
    assert pytest.approx(feats["tcp_retransmission_rate"], abs=1e-4) == 0.5


# ── Test 10: ACK-only packet handling ─────────────────────────────────────────
def test_ack_only_packet_handling():
    # Pure ACKs (payload_length=0, flags=ACK) with identical sequence numbers
    # MUST NOT be counted as retransmissions
    pkts = [
        _make_pkt(timestamp=1.0 + i * 0.01, tcp_seq=5000, payload_length=0, tcp_flags=0x10)
        for i in range(10)
    ]
    feats = extract_packet_features(pkts)
    assert feats["tcp_retransmission_count"] == 0.0
    assert feats["tcp_retransmission_rate"] == 0.0


# ── Test 11: Insufficient TCP metadata handling ───────────────────────────────
def test_insufficient_tcp_metadata_handling():
    # Packets with zero sequence numbers and zero payload
    pkts = [
        _make_pkt(tcp_seq=0, payload_length=0, tcp_flags=0),
        _make_pkt(tcp_seq=0, payload_length=0, tcp_flags=0),
    ]
    feats = extract_packet_features(pkts)
    assert feats["tcp_retransmission_count"] == 0.0
    assert feats["tcp_retransmission_rate"] == 0.0


# ── Test 12: Inter-arrival mean ───────────────────────────────────────────────
def test_inter_arrival_mean():
    # Timestamps: 10.0, 10.2, 10.5 -> intervals: 0.2, 0.3 -> mean = 0.25
    pkts = [
        _make_pkt(timestamp=10.0),
        _make_pkt(timestamp=10.2),
        _make_pkt(timestamp=10.5),
    ]
    feats = extract_packet_features(pkts)
    assert pytest.approx(feats["packet_iat_mean"], abs=1e-5) == 0.25


# ── Test 13: Inter-arrival standard deviation ─────────────────────────────────
def test_inter_arrival_standard_deviation():
    # Intervals: [0.1, 0.1] -> std = 0.0
    pkts = [
        _make_pkt(timestamp=1.0),
        _make_pkt(timestamp=1.1),
        _make_pkt(timestamp=1.2),
    ]
    feats = extract_packet_features(pkts)
    assert feats["packet_iat_std"] == 0.0


# ── Test 14: IAT min/max ──────────────────────────────────────────────────────
def test_iat_min_max():
    # Intervals: [0.05, 0.30] -> min = 0.05, max = 0.30
    pkts = [
        _make_pkt(timestamp=10.0),
        _make_pkt(timestamp=10.05),
        _make_pkt(timestamp=10.35),
    ]
    feats = extract_packet_features(pkts)
    assert pytest.approx(feats["packet_iat_min"], abs=1e-5) == 0.05
    assert pytest.approx(feats["packet_iat_max"], abs=1e-5) == 0.30


# ── Test 15: IAT percentiles ──────────────────────────────────────────────────
def test_iat_percentiles():
    # 10 intervals from 0.01 to 0.10
    ts = [0.0]
    for i in range(1, 11):
        ts.append(ts[-1] + i * 0.01)
    pkts = [_make_pkt(timestamp=t) for t in ts]
    feats = extract_packet_features(pkts)
    iats = np.diff(ts)
    assert pytest.approx(feats["packet_iat_p50"], abs=1e-5) == float(np.percentile(iats, 50))
    assert pytest.approx(feats["packet_iat_p95"], abs=1e-5) == float(np.percentile(iats, 95))


# ── Test 16: IAT jitter ───────────────────────────────────────────────────────
def test_iat_jitter():
    # Timestamps: 0.0, 0.1, 0.3, 0.4
    # Intervals: Δt = [0.1, 0.2, 0.1]
    # Consecutive changes: |0.2 - 0.1| = 0.1, |0.1 - 0.2| = 0.1
    # Jitter = mean([0.1, 0.1]) = 0.1
    pkts = [
        _make_pkt(timestamp=0.0),
        _make_pkt(timestamp=0.1),
        _make_pkt(timestamp=0.3),
        _make_pkt(timestamp=0.4),
    ]
    feats = extract_packet_features(pkts)
    assert pytest.approx(feats["packet_iat_jitter"], abs=1e-5) == 0.1


# ── Test 17: Timestamp ordering ───────────────────────────────────────────────
def test_timestamp_ordering():
    # Unordered packet timestamps: [10.3, 10.0, 10.2, 10.1]
    # Should sort chronologically to [10.0, 10.1, 10.2, 10.3] -> Δt = [0.1, 0.1, 0.1]
    pkts = [
        _make_pkt(timestamp=10.3),
        _make_pkt(timestamp=10.0),
        _make_pkt(timestamp=10.2),
        _make_pkt(timestamp=10.1),
    ]
    feats = extract_packet_features(pkts)
    assert pytest.approx(feats["packet_iat_mean"], abs=1e-5) == 0.1
    assert feats["packet_iat_min"] >= 0.0
    assert feats["packet_iat_jitter"] == 0.0


# ── Test 18: Missing timestamp handling ───────────────────────────────────────
def test_missing_timestamp_handling():
    # Non-finite or missing timestamp
    pkts = [
        _make_pkt(timestamp=float("nan")),
        _make_pkt(timestamp=10.0),
        _make_pkt(timestamp=10.2),
    ]
    feats = extract_packet_features(pkts)
    # The nan is filtered, leaving [10.0, 10.2] -> single interval 0.2
    assert pytest.approx(feats["packet_iat_mean"], abs=1e-5) == 0.2
    assert feats["packet_iat_jitter"] == 0.0


# ── Test 19: Single-packet flow handling ──────────────────────────────────────
def test_single_packet_flow_handling():
    pkts = [_make_pkt(timestamp=10.0, ttl=64, tcp_seq=1000, payload_length=100)]
    feats = extract_packet_features(pkts)
    assert feats["packet_iat_mean"] == 0.0
    assert feats["packet_iat_std"] == 0.0
    assert feats["packet_iat_jitter"] == 0.0
    assert feats["tcp_retransmission_count"] == 0.0
    assert feats["tcp_retransmission_rate"] == 0.0
    assert feats["ttl_mean"] == 64.0


# ── Test 20: UDP flow handling ────────────────────────────────────────────────
def test_udp_flow_handling():
    pkts = [
        _make_pkt(timestamp=1.0, protocol=17, ttl=64, src_port=53001, dst_port=53),
        _make_pkt(timestamp=1.1, protocol=17, ttl=64, src_port=53001, dst_port=53),
    ]
    feats = extract_packet_features(pkts)
    assert feats["tcp_retransmission_count"] == 0.0
    assert feats["tcp_retransmission_rate"] == 0.0
    assert feats["ttl_mean"] == 64.0
    assert pytest.approx(feats["packet_iat_mean"], abs=1e-5) == 0.1


# ── Test 21: ICMP/non-TCP handling ────────────────────────────────────────────
def test_icmp_non_tcp_handling():
    pkts = [
        _make_pkt(timestamp=1.0, protocol=1, ttl=255, src_port=0, dst_port=0),
        _make_pkt(timestamp=1.05, protocol=1, ttl=255, src_port=0, dst_port=0),
    ]
    feats = extract_packet_features(pkts)
    assert feats["tcp_retransmission_count"] == 0.0
    assert feats["tcp_retransmission_rate"] == 0.0
    assert feats["ttl_mean"] == 255.0
    assert pytest.approx(feats["packet_iat_mean"], abs=1e-5) == 0.05


# ── Test 22: Mixed protocol handling ──────────────────────────────────────────
def test_mixed_protocol_handling():
    pkts = [
        _make_pkt(timestamp=1.0, protocol=6, ttl=64, tcp_seq=100, payload_length=50),
        _make_pkt(timestamp=1.1, protocol=17, ttl=128),
        _make_pkt(timestamp=1.2, protocol=1, ttl=32),
    ]
    feats = extract_packet_features(pkts)
    assert np.isfinite(feats["ttl_mean"])
    assert feats["ttl_min"] == 32.0
    assert feats["ttl_max"] == 128.0
    assert feats["tcp_retransmission_count"] == 0.0


# ── Test 23: Directional feature correctness ──────────────────────────────────
def test_directional_feature_correctness():
    key = FlowKey("192.168.1.10", "10.0.0.1", 54321, 80, 6)
    flow = FlowRecord(
        flow_id="test_flow_1",
        key=key,
        flow_start_time=1.0,
        flow_end_time=2.0,
        last_packet_time=2.0,
    )
    # Forward packets: TTL 64, with 1 retransmission
    fwd1 = _make_pkt(timestamp=1.0, ttl=64, tcp_seq=1000, payload_length=100)
    fwd2 = _make_pkt(timestamp=1.1, ttl=64, tcp_seq=1000, payload_length=100)  # duplicate
    # Backward packets: TTL 128, no retransmission
    bwd1 = _make_pkt(
        timestamp=1.2,
        src_ip="10.0.0.1",
        dst_ip="192.168.1.10",
        src_port=80,
        dst_port=54321,
        ttl=128,
        tcp_seq=5000,
        payload_length=200,
    )

    flow.add_packet(fwd1, is_forward=True)
    flow.add_packet(fwd2, is_forward=True)
    flow.add_packet(bwd1, is_forward=False)

    feats = extract_packet_features(flow)
    assert feats["fwd_ttl_mean"] == 64.0
    assert feats["bwd_ttl_mean"] == 128.0
    assert feats["fwd_tcp_retrans_count"] == 1.0
    assert feats["bwd_tcp_retrans_count"] == 0.0
    assert feats["tcp_retransmission_count"] == 1.0


# ── Test 24: Deterministic output ─────────────────────────────────────────────
def test_deterministic_output():
    pkts = [
        _make_pkt(timestamp=1.0 + i * 0.1, tcp_seq=1000 + i * 50, payload_length=50)
        for i in range(10)
    ]
    vec1, names1 = extract_packet_feature_vector(pkts)
    vec2, names2 = extract_packet_feature_vector(pkts)
    assert names1 == names2
    assert np.array_equal(vec1, vec2)


# ── Test 25: Feature schema ordering ──────────────────────────────────────────
def test_feature_schema_ordering():
    # Schema must be strictly sorted alphabetically
    assert list(PACKET_FEATURE_NAMES) == sorted(PACKET_FEATURE_NAMES)
    vec, names = extract_packet_feature_vector([])
    assert names == list(PACKET_FEATURE_NAMES)


# ── Test 26: Feature dimensionality ───────────────────────────────────────────
def test_feature_dimensionality():
    assert PACKET_FEATURE_DIM == 18
    assert len(PACKET_FEATURE_NAMES) == 18
    vec, names = extract_packet_feature_vector([])
    assert vec.shape == (18,)
    assert len(names) == 18


# ── Test 27: Existing 78-feature schema unchanged ─────────────────────────────
def test_existing_78_feature_schema_unchanged():
    assert CANONICAL_FEATURE_DIM == 78
    assert len(CANONICAL_FEATURES) == 78
    assert CANONICAL_FEATURES[0] == "ack_flag_cnt"
    assert CANONICAL_FEATURES[-1] == "urg_flag_cnt"


# ── Test 28: Existing 280-D state unchanged ───────────────────────────────────
def test_existing_280_d_state_unchanged():
    assert STATE_FEATURE_DIM == 280


# ── Test 29: Existing World Model input compatibility ─────────────────────────
def test_existing_world_model_input_compatibility():
    # The World Model expects [N, 280] state tensor, [N] horizon, [N] delta_t
    model = StateResidualDynamicsModel(state_dim=280)
    dummy_state = torch.randn(2, 280)
    dummy_k = torch.tensor([1, 3], dtype=torch.long)
    dummy_dt = torch.tensor([0.5, 1.5], dtype=torch.float32)

    predicted_state = model(dummy_state, dummy_k, dummy_dt)
    assert predicted_state.shape == (2, 280)
    assert not torch.isnan(predicted_state).any()


# ── Test 30: Malformed packet robustness ──────────────────────────────────────
def test_malformed_packet_robustness():
    pkts = [
        _make_pkt(payload_length=-10, tcp_seq=-500, ttl=999),  # invalid values
        _make_pkt(timestamp=float("inf"), ttl=-1),
    ]
    extractor = PacketFeatureExtractor()
    # Must not raise unhandled exception
    vec = extractor.extract_from_packets(pkts)
    assert isinstance(vec, PacketFeatureVector)
    # The out-of-range TTLs must be ignored or mapped cleanly
    assert np.isfinite(vec.packet_iat_mean)


# ── Test 31: Empty packet list handling ───────────────────────────────────────
def test_empty_packet_list_handling():
    feats = extract_packet_features([])
    assert feats["ttl_min"] == MISSING_TTL_SENTINEL
    assert feats["ttl_max"] == MISSING_TTL_SENTINEL
    assert feats["ttl_mean"] == MISSING_TTL_SENTINEL
    assert feats["tcp_retransmission_count"] == 0.0
    assert feats["tcp_retransmission_rate"] == 0.0
    assert feats["packet_iat_mean"] == 0.0
    assert feats["packet_iat_jitter"] == 0.0

    # Also test build_enriched_features with minimal empty flow
    key = FlowKey("1.1.1.1", "2.2.2.2", 1000, 80, 6)
    flow = FlowRecord("empty_flow", key, 1000.0, 1000.0, 1000.0)
    enriched = build_enriched_features(flow)
    assert "packet_features" in enriched
    assert len(enriched["packet_features"]) == 18
    assert "pkt_ttl_mean" in enriched


# ── Test 32: Large-flow bounded processing sanity test ────────────────────────
def test_large_flow_bounded_processing_sanity():
    # 5,000 packets with realistic alternating timestamps and sequence numbers
    n_pkts = 5000
    pkts = [
        _make_pkt(
            timestamp=1000.0 + i * 0.001,
            tcp_seq=1000 + i * 100,
            payload_length=100,
            ttl=64 if i % 2 == 0 else 128,
        )
        for i in range(n_pkts)
    ]
    t0 = time.perf_counter()
    vec, names = extract_packet_feature_vector(pkts)
    elapsed = time.perf_counter() - t0

    assert len(vec) == 18
    # 5,000 packets must be processed in under 1.0 second (typically < 50ms)
    assert elapsed < 1.0, f"Extraction took too long: {elapsed:.3f}s"
