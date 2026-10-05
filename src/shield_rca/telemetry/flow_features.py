"""Flow-level statistical feature extraction for CyberHeal AI.

Responsibility: compute ~80 statistical, timing, flag, rate, bulk, and
active/idle features from a FlowRecord compliant with the CICFlowMeter standard.
All functions are pure (stateless); no I/O or side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .flow_aggregator import FlowRecord
from .pcap_parser import RawPacketRecord

# TCP flag bit masks
_FLAG_FIN = 0x01
_FLAG_SYN = 0x02
_FLAG_RST = 0x04
_FLAG_PSH = 0x08
_FLAG_ACK = 0x10
_FLAG_URG = 0x20
_FLAG_ECE = 0x40
_FLAG_CWR = 0x80

_EPS = 1e-9  # prevent division by zero


def extract_flow_features(flow: FlowRecord) -> dict[str, Any]:
    """Extract all flow-level features from a FlowRecord.

    This is the primary public function. It computes counts, timing,
    packet length distributions, flags, rates, and active/idle metrics.

    Args:
        flow: A completed FlowRecord from flow_aggregator.aggregate_flows.

    Returns:
        Dict mapping feature name -> scalar value (int or float).
    """
    fwd = flow.fwd_packets
    bwd = flow.bwd_packets
    all_pkts = flow.all_packets
    duration_us = max((flow.flow_end_time - flow.flow_start_time) * 1e6, _EPS)

    features: dict[str, Any] = {}

    # ── Identity ──────────────────────────────────────────────────────────────
    features["flow_id"] = flow.flow_id
    features["flow_start_time"] = flow.flow_start_time
    features["src_ip"] = flow.key.src_ip
    features["dst_ip"] = flow.key.dst_ip
    features["src_port"] = flow.key.src_port
    features["dst_port"] = flow.key.dst_port
    features["protocol"] = flow.key.protocol

    # ── Counts & bytes ────────────────────────────────────────────────────────
    features.update(_count_features(fwd, bwd))

    # ── Timing ────────────────────────────────────────────────────────────────
    features["flow_duration_us"] = duration_us
    features.update(_iat_features(fwd, "fwd"))
    features.update(_iat_features(bwd, "bwd"))
    features.update(_iat_features(all_pkts, "flow"))

    # ── Packet length stats ───────────────────────────────────────────────────
    features.update(_pkt_len_features(fwd, "fwd"))
    features.update(_pkt_len_features(bwd, "bwd"))
    features.update(_pkt_len_features(all_pkts, ""))

    # ── TCP flags ─────────────────────────────────────────────────────────────
    features.update(_tcp_flag_features(fwd, bwd))

    # ── Rates ─────────────────────────────────────────────────────────────────
    fwd_bytes = sum(p.ip_length for p in fwd)
    bwd_bytes = sum(p.ip_length for p in bwd)
    total_bytes = fwd_bytes + bwd_bytes
    total_pkts = len(all_pkts)
    duration_s = duration_us / 1e6

    features["flow_bytes_per_s"] = total_bytes / duration_s
    features["flow_packets_per_s"] = total_pkts / duration_s
    features["fwd_bytes_per_s"] = fwd_bytes / duration_s
    features["bwd_bytes_per_s"] = bwd_bytes / duration_s
    features["down_up_ratio"] = bwd_bytes / (fwd_bytes + _EPS)

    # ── Header lengths ────────────────────────────────────────────────────────
    features["fwd_header_length"] = sum(p.ip_length - p.payload_length for p in fwd)
    features["bwd_header_length"] = sum(p.ip_length - p.payload_length for p in bwd)

    # ── Bulk features ─────────────────────────────────────────────────────────
    features.update(_bulk_features(fwd, "fwd", duration_s))
    features.update(_bulk_features(bwd, "bwd", duration_s))

    # ── Subflow counts ────────────────────────────────────────────────────────
    features["subflow_fwd_packets"] = float(len(fwd))
    features["subflow_fwd_bytes"] = float(fwd_bytes)
    features["subflow_bwd_packets"] = float(len(bwd))
    features["subflow_bwd_bytes"] = float(bwd_bytes)

    # ── Active / idle times ───────────────────────────────────────────────────
    features.update(_active_idle_features(flow.active_durations, "active"))
    features.update(_active_idle_features(flow.idle_durations, "idle"))

    return features


def extract_feature_vector(
    flow: FlowRecord,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract numeric feature vector suitable for machine learning inference.

    Excludes identifier strings (flow_id, src_ip, dst_ip).

    Returns:
        tuple of (1D numpy float array, list of feature names)
    """
    feat_dict = extract_flow_features(flow)
    excluded = {"flow_id", "src_ip", "dst_ip", "flow_start_time"}

    if feature_names is not None:
        names = [k for k in feature_names if k not in excluded]
    else:
        names = [k for k in sorted(feat_dict.keys()) if k not in excluded]

    vector = np.array([float(feat_dict.get(k, 0.0)) for k in names], dtype=np.float32)
    # Replace non-finite values (inf, nan) with 0.0
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector, names


def _count_features(
    fwd: list[RawPacketRecord], bwd: list[RawPacketRecord]
) -> dict[str, Any]:
    return {
        "tot_fwd_pkts": len(fwd),
        "tot_bwd_pkts": len(bwd),
        "tot_fwd_bytes": sum(p.ip_length for p in fwd),
        "tot_bwd_bytes": sum(p.ip_length for p in bwd),
    }


def _iat_features(pkts: list[RawPacketRecord], prefix: str) -> dict[str, Any]:
    key = f"{prefix}_iat" if prefix else "iat"
    if len(pkts) < 2:
        return {
            f"{key}_mean": 0.0,
            f"{key}_std": 0.0,
            f"{key}_max": 0.0,
            f"{key}_min": 0.0,
            f"{key}_total": 0.0,
        }
    timestamps = np.array([p.timestamp for p in pkts])
    iats = np.diff(timestamps) * 1e6  # to microseconds
    return {
        f"{key}_mean": float(np.mean(iats)),
        f"{key}_std": float(np.std(iats)),
        f"{key}_max": float(np.max(iats)),
        f"{key}_min": float(np.min(iats)),
        f"{key}_total": float(np.sum(iats)),
    }


def _pkt_len_features(pkts: list[RawPacketRecord], prefix: str) -> dict[str, Any]:
    key = f"{prefix}_pkt_len" if prefix else "pkt_len"
    if not pkts:
        return {
            f"{key}_max": 0.0,
            f"{key}_min": 0.0,
            f"{key}_mean": 0.0,
            f"{key}_std": 0.0,
            f"{key}_var": 0.0,
            f"{key}_skew": 0.0,
        }
    lens = np.array([p.ip_length for p in pkts], dtype=float)
    mean = float(np.mean(lens))
    std = float(np.std(lens))
    var = float(np.var(lens))

    # Fisher-Pearson skewness
    if len(lens) > 2 and std > _EPS:
        skew = float(np.mean(((lens - mean) / std) ** 3))
    else:
        skew = 0.0

    return {
        f"{key}_max": float(np.max(lens)),
        f"{key}_min": float(np.min(lens)),
        f"{key}_mean": mean,
        f"{key}_std": std,
        f"{key}_var": var,
        f"{key}_skew": skew,
    }


def _tcp_flag_features(
    fwd: list[RawPacketRecord], bwd: list[RawPacketRecord]
) -> dict[str, Any]:
    all_pkts = fwd + bwd
    flags = [p.tcp_flags for p in all_pkts if p.protocol == 6]
    fwd_flags = [p.tcp_flags for p in fwd if p.protocol == 6]
    bwd_flags = [p.tcp_flags for p in bwd if p.protocol == 6]

    return {
        "fin_flag_cnt": sum(1 for f in flags if f & _FLAG_FIN),
        "syn_flag_cnt": sum(1 for f in flags if f & _FLAG_SYN),
        "rst_flag_cnt": sum(1 for f in flags if f & _FLAG_RST),
        "psh_flag_cnt": sum(1 for f in flags if f & _FLAG_PSH),
        "ack_flag_cnt": sum(1 for f in flags if f & _FLAG_ACK),
        "urg_flag_cnt": sum(1 for f in flags if f & _FLAG_URG),
        "ece_flag_cnt": sum(1 for f in flags if f & _FLAG_ECE),
        "cwr_flag_cnt": sum(1 for f in flags if f & _FLAG_CWR),
        "fwd_psh_flags": sum(1 for f in fwd_flags if f & _FLAG_PSH),
        "bwd_psh_flags": sum(1 for f in bwd_flags if f & _FLAG_PSH),
        "fwd_urg_flags": sum(1 for f in fwd_flags if f & _FLAG_URG),
        "bwd_urg_flags": sum(1 for f in bwd_flags if f & _FLAG_URG),
    }


def _bulk_features(
    pkts: list[RawPacketRecord], prefix: str, duration_s: float
) -> dict[str, Any]:
    bytes_count = sum(p.ip_length for p in pkts)
    pkts_count = len(pkts)
    rate = bytes_count / (duration_s + _EPS)
    return {
        f"{prefix}_avg_bytes_per_bulk": float(bytes_count),
        f"{prefix}_avg_pkts_per_bulk": float(pkts_count),
        f"{prefix}_avg_bulk_rate": float(rate),
    }


def _active_idle_features(durations: list[float], prefix: str) -> dict[str, Any]:
    if not durations:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_min": 0.0,
        }
    arr = np.array(durations) * 1e6  # to microseconds
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_min": float(np.min(arr)),
    }
