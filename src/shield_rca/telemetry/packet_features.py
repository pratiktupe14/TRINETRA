"""Packet-Level Feature Extractor for TRINETRA / CyberHeal AI.

Extracts deterministic packet-level signals from packet streams or aggregated flows:
1. IPv4 TTL & IPv6 Hop-Limit statistics (min, max, mean, std, range, fwd/bwd mean).
2. TCP Retransmission detection via directional sequence-overlap tracking.
3. Packet Inter-Arrival Time (IAT) statistics and packet IAT jitter.

This module is additive and backward-compatible. It operates alongside the
existing 78-dimensional flow feature extraction pipeline without modifying
canonical flow feature dimensions or ordering.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .flow_aggregator import FlowRecord
from .flow_features import extract_flow_features
from .packet_feature_schema import (
    MISSING_TTL_SENTINEL,
    PACKET_FEATURE_NAMES,
    PacketFeatureVector,
    validate_packet_features,
)
from .pcap_parser import RawPacketRecord

logger = logging.getLogger(__name__)

# TCP flag bitmasks
_FLAG_FIN = 0x01
_FLAG_SYN = 0x02
_FLAG_RST = 0x04
_FLAG_PSH = 0x08
_FLAG_ACK = 0x10


def _merge_interval(
    intervals: list[tuple[int, int]], new_start: int, new_end: int
) -> list[tuple[int, int]]:
    """Merge [new_start, new_end) into a list of disjoint, sorted [start, end) intervals."""
    if not intervals:
        return [(new_start, new_end)]
    merged: list[tuple[int, int]] = []
    inserted = False
    cur_start, cur_end = new_start, new_end

    for start, end in intervals:
        if end < cur_start:
            merged.append((start, end))
        elif start > cur_end:
            if not inserted:
                merged.append((cur_start, cur_end))
                inserted = True
            merged.append((start, end))
        else:
            # Overlapping or contiguous intervals
            cur_start = min(cur_start, start)
            cur_end = max(cur_end, end)

    if not inserted:
        merged.append((cur_start, cur_end))

    return merged


def _overlaps(intervals: list[tuple[int, int]], s: int, e: int) -> bool:
    """Check if [s, e) overlaps any interval in disjoint sorted intervals."""
    for start, end in intervals:
        if max(start, s) < min(end, e):
            return True
        if start >= e:
            break
    return False


def _detect_direction_retransmissions(
    pkts: Sequence[RawPacketRecord],
) -> tuple[int, int]:
    """Detect TCP retransmissions in a single direction via sequence-overlap heuristic.

    A packet is flagged as a retransmission if:
    1. It is a TCP packet (protocol == 6).
    2. It carries payload (payload_length > 0) or sets SYN/FIN flags.
       (Pure zero-payload ACKs are excluded to prevent counting legitimate ACK progressions).
    3. Its sequence-number range [seq, seq + length) overlaps byte intervals
       previously observed in this direction.

    Args:
        pkts: Sequence of RawPacketRecord objects in one direction.

    Returns:
        (retransmission_count, total_tcp_packets)
    """
    tcp_pkts = [p for p in pkts if p.protocol == 6]
    if not tcp_pkts:
        return 0, 0

    # Ensure chronological order
    sorted_pkts = sorted(tcp_pkts, key=lambda p: p.timestamp)

    seen_intervals: list[tuple[int, int]] = []
    retrans_count = 0

    for p in sorted_pkts:
        payload_len = p.payload_length if p.payload_length is not None else 0
        is_syn = bool(p.tcp_flags & _FLAG_SYN)
        is_fin = bool(p.tcp_flags & _FLAG_FIN)

        # SYN and FIN flags consume 1 sequence space even if payload is 0
        effective_len = max(payload_len, 1) if (is_syn or is_fin) else payload_len

        # Legitimate zero-payload ACKs and keep-alives do not advance data sequence
        if effective_len <= 0:
            continue

        seq = p.tcp_seq if p.tcp_seq is not None else 0
        start = seq
        end = seq + effective_len

        # Check for overlap with previously transmitted data in this direction
        if _overlaps(seen_intervals, start, end):
            retrans_count += 1

        seen_intervals = _merge_interval(seen_intervals, start, end)

    return retrans_count, len(tcp_pkts)


class PacketFeatureExtractor:
    """Extractor for packet-level traffic signals.

    Computes:
    - IPv4 TTL & IPv6 Hop-Limit summary statistics
    - Deterministic TCP retransmission counts and rates
    - Packet Inter-Arrival Time (IAT) statistics and IAT jitter
    """

    def __init__(self, missing_ttl_sentinel: float = MISSING_TTL_SENTINEL) -> None:
        self.missing_ttl_sentinel = missing_ttl_sentinel

    def extract_from_flow(self, flow: FlowRecord) -> PacketFeatureVector:
        """Extract PacketFeatureVector from a completed FlowRecord."""
        all_pkts = flow.all_packets
        fwd_pkts = flow.fwd_packets
        bwd_pkts = flow.bwd_packets
        return self._extract(all_pkts, fwd_pkts, bwd_pkts)

    def extract_from_packets(
        self,
        packets: Sequence[RawPacketRecord],
        fwd_packets: Sequence[RawPacketRecord] | None = None,
        bwd_packets: Sequence[RawPacketRecord] | None = None,
    ) -> PacketFeatureVector:
        """Extract PacketFeatureVector from an arbitrary sequence of packets."""
        if fwd_packets is None and bwd_packets is None:
            fwd_pkts = list(packets)
            bwd_pkts = []
        else:
            fwd_pkts = list(fwd_packets or [])
            bwd_pkts = list(bwd_packets or [])
        return self._extract(packets, fwd_pkts, bwd_pkts)

    def _extract(
        self,
        all_pkts: Sequence[RawPacketRecord],
        fwd_pkts: Sequence[RawPacketRecord],
        bwd_pkts: Sequence[RawPacketRecord],
    ) -> PacketFeatureVector:
        # ── 1. TTL / Hop-Limit Features ───────────────────────────────────────
        ttl_dict = self._compute_ttl_stats(all_pkts, fwd_pkts, bwd_pkts)

        # ── 2. TCP Retransmission Features ────────────────────────────────────
        retrans_dict = self._compute_retransmission_stats(fwd_pkts, bwd_pkts)

        # ── 3. Inter-Arrival Timing & Jitter Features ─────────────────────────
        iat_dict = self._compute_iat_and_jitter(all_pkts)

        merged = {**ttl_dict, **retrans_dict, **iat_dict}
        vec = PacketFeatureVector.from_dict(merged)
        validate_packet_features(vec.to_dict())
        return vec

    def _compute_ttl_stats(
        self,
        all_pkts: Sequence[RawPacketRecord],
        fwd_pkts: Sequence[RawPacketRecord],
        bwd_pkts: Sequence[RawPacketRecord],
    ) -> dict[str, float]:
        valid_ttls = [
            float(p.ttl)
            for p in all_pkts
            if p.ttl is not None and 0 <= p.ttl <= 255
        ]
        if not valid_ttls:
            ttl_min = self.missing_ttl_sentinel
            ttl_max = self.missing_ttl_sentinel
            ttl_mean = self.missing_ttl_sentinel
            ttl_std = self.missing_ttl_sentinel
            ttl_range = self.missing_ttl_sentinel
        else:
            ttl_min = float(np.min(valid_ttls))
            ttl_max = float(np.max(valid_ttls))
            ttl_mean = float(np.mean(valid_ttls))
            ttl_std = float(np.std(valid_ttls))
            ttl_range = float(ttl_max - ttl_min)

        fwd_ttls = [
            float(p.ttl)
            for p in fwd_pkts
            if p.ttl is not None and 0 <= p.ttl <= 255
        ]
        fwd_ttl_mean = float(np.mean(fwd_ttls)) if fwd_ttls else self.missing_ttl_sentinel

        bwd_ttls = [
            float(p.ttl)
            for p in bwd_pkts
            if p.ttl is not None and 0 <= p.ttl <= 255
        ]
        bwd_ttl_mean = float(np.mean(bwd_ttls)) if bwd_ttls else self.missing_ttl_sentinel

        return {
            "ttl_min": ttl_min,
            "ttl_max": ttl_max,
            "ttl_mean": ttl_mean,
            "ttl_std": ttl_std,
            "ttl_range": ttl_range,
            "fwd_ttl_mean": fwd_ttl_mean,
            "bwd_ttl_mean": bwd_ttl_mean,
        }

    def _compute_retransmission_stats(
        self,
        fwd_pkts: Sequence[RawPacketRecord],
        bwd_pkts: Sequence[RawPacketRecord],
    ) -> dict[str, float]:
        fwd_retrans, fwd_tcp_count = _detect_direction_retransmissions(fwd_pkts)
        bwd_retrans, bwd_tcp_count = _detect_direction_retransmissions(bwd_pkts)

        tot_retrans = fwd_retrans + bwd_retrans
        tot_tcp = fwd_tcp_count + bwd_tcp_count

        rate = (float(tot_retrans) / float(tot_tcp)) if tot_tcp > 0 else 0.0

        return {
            "tcp_retransmission_count": float(tot_retrans),
            "tcp_retransmission_rate": float(rate),
            "fwd_tcp_retrans_count": float(fwd_retrans),
            "bwd_tcp_retrans_count": float(bwd_retrans),
        }

    def _compute_iat_and_jitter(
        self,
        all_pkts: Sequence[RawPacketRecord],
    ) -> dict[str, float]:
        valid_ts = [
            float(p.timestamp)
            for p in all_pkts
            if p.timestamp is not None and np.isfinite(p.timestamp)
        ]
        if len(valid_ts) < 2:
            return {
                "packet_iat_mean": 0.0,
                "packet_iat_std": 0.0,
                "packet_iat_min": 0.0,
                "packet_iat_max": 0.0,
                "packet_iat_p50": 0.0,
                "packet_iat_p95": 0.0,
                "packet_iat_jitter": 0.0,
            }

        valid_ts.sort()
        iats = np.diff(valid_ts)
        iats = np.maximum(iats, 0.0)  # eliminate negative epsilon from float rounding

        mean_val = float(np.mean(iats))
        std_val = float(np.std(iats))
        if std_val < 1e-12:
            std_val = 0.0
        min_val = float(np.min(iats))
        max_val = float(np.max(iats))
        p50_val = float(np.percentile(iats, 50))
        p95_val = float(np.percentile(iats, 95))

        if len(iats) < 2:
            jitter_val = 0.0
        else:
            # Deterministic packet inter-arrival jitter:
            # jitter = mean(|Δt_i - Δt_{i-1}|)
            jitter_diffs = np.abs(np.diff(iats))
            jitter_val = float(np.mean(jitter_diffs))
            if jitter_val < 1e-12:
                jitter_val = 0.0

        return {
            "packet_iat_mean": mean_val,
            "packet_iat_std": std_val,
            "packet_iat_min": min_val,
            "packet_iat_max": max_val,
            "packet_iat_p50": p50_val,
            "packet_iat_p95": p95_val,
            "packet_iat_jitter": jitter_val,
        }


# Module-level default singleton extractor
_DEFAULT_EXTRACTOR = PacketFeatureExtractor()


def extract_packet_features(
    flow_or_packets: FlowRecord | Sequence[RawPacketRecord],
) -> dict[str, float]:
    """Extract packet features as a dictionary keyed by canonical feature name.

    Args:
        flow_or_packets: Either a FlowRecord or a sequence of RawPacketRecord objects.

    Returns:
        Dictionary mapping all 18 packet feature names to float values.
    """
    if isinstance(flow_or_packets, FlowRecord):
        vec = _DEFAULT_EXTRACTOR.extract_from_flow(flow_or_packets)
    else:
        vec = _DEFAULT_EXTRACTOR.extract_from_packets(flow_or_packets)
    return vec.to_dict()


def extract_packet_feature_vector(
    flow_or_packets: FlowRecord | Sequence[RawPacketRecord],
) -> tuple[np.ndarray, list[str]]:
    """Extract numeric 1D numpy array of packet features and corresponding names.

    Args:
        flow_or_packets: Either a FlowRecord or a sequence of RawPacketRecord objects.

    Returns:
        Tuple of (1D float32 numpy array of shape (18,), list of 18 feature names)
    """
    if isinstance(flow_or_packets, FlowRecord):
        vec = _DEFAULT_EXTRACTOR.extract_from_flow(flow_or_packets)
    else:
        vec = _DEFAULT_EXTRACTOR.extract_from_packets(flow_or_packets)
    return vec.to_numpy(), list(PACKET_FEATURE_NAMES)


def build_enriched_features(flow: FlowRecord) -> dict[str, Any]:
    """Combine 78 canonical flow features with 18 packet-level features.

    Preserves 100% backward compatibility for canonical flow feature consumers
    while attaching packet-level enrichment under 'packet_features' and as
    prefixed 'pkt_<name>' keys.

    Args:
        flow: Completed FlowRecord.

    Returns:
        Dictionary containing:
        - 78 canonical flow features
        - Metadata fields (flow_id, src_ip, dst_ip, etc.)
        - 'packet_features': dict of 18 packet-level features
        - 'pkt_<feature_name>': flattened packet features for tabular ML
    """
    flow_dict = extract_flow_features(flow)
    packet_dict = extract_packet_features(flow)

    enriched = dict(flow_dict)
    enriched["packet_features"] = packet_dict

    # Provide prefixed keys to prevent any name collisions with 78 flow features
    for k, v in packet_dict.items():
        enriched[f"pkt_{k}"] = v

    return enriched
