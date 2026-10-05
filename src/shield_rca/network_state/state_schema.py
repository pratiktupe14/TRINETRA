"""Canonical Network State Schema for CyberHeal AI.

Defines the single source of truth for the Network State representation S_t in R^280.
Constructed deterministically from historical observation windows X_{t-T+1:t} in R^{T x 78}.

The state summarizes:
1. Continuous Traffic Dynamics (30 features x 5 stats = 150)
2. TCP Flag & Connection Signatures (12 features x 4 stats = 48)
3. Distribution Shape & Auxiliary Metrics (33 features x 2 stats = 66)
4. Network Endpoint & Protocol Diversity (6 structural features)
5. Macro-Network State Indicators (10 holistic features)

Total Dimension: S = 280.
Zero future leakage. Zero label inclusion. 100% deterministic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..telemetry.feature_schema import CANONICAL_FEATURES

STATE_SCHEMA_VERSION: str = "1.0.0"
STATE_FEATURE_DIM: int = 280

# ── Group 1: Continuous Traffic Metrics (30 features) ───────────────────────
# Features capturing rates, byte counts, IATs, and durations.
# Aggregations: mean, std, max, last, trend (5 stats)
GROUP_CONTINUOUS: tuple[str, ...] = (
    "active_max",
    "active_mean",
    "bwd_bytes_per_s",
    "bwd_iat_mean",
    "bwd_pkt_len_max",
    "bwd_pkt_len_mean",
    "down_up_ratio",
    "flow_bytes_per_s",
    "flow_duration_us",
    "flow_iat_max",
    "flow_iat_mean",
    "flow_iat_total",
    "flow_packets_per_s",
    "fwd_bytes_per_s",
    "fwd_iat_mean",
    "fwd_pkt_len_max",
    "fwd_pkt_len_mean",
    "idle_max",
    "idle_mean",
    "pkt_len_max",
    "pkt_len_mean",
    "pkt_len_std",
    "subflow_bwd_bytes",
    "subflow_bwd_packets",
    "subflow_fwd_bytes",
    "subflow_fwd_packets",
    "tot_bwd_bytes",
    "tot_bwd_pkts",
    "tot_fwd_bytes",
    "tot_fwd_pkts",
)

# ── Group 2: TCP Flags & Connection Counters (12 features) ───────────────────
# Discrete flag counters.
# Aggregations: mean, max, last, delta (4 stats)
GROUP_TCP_FLAGS: tuple[str, ...] = (
    "ack_flag_cnt",
    "bwd_psh_flags",
    "bwd_urg_flags",
    "cwr_flag_cnt",
    "ece_flag_cnt",
    "fin_flag_cnt",
    "fwd_psh_flags",
    "fwd_urg_flags",
    "psh_flag_cnt",
    "rst_flag_cnt",
    "syn_flag_cnt",
    "urg_flag_cnt",
)

# ── Group 3: Shape & Auxiliary Distribution Metrics (33 features) ────────────
# Skewness, variance, min lengths, header lengths, and bulk metrics.
# Aggregations: mean, last (2 stats)
GROUP_SHAPE: tuple[str, ...] = (
    "active_min",
    "active_std",
    "bwd_avg_bulk_rate",
    "bwd_avg_bytes_per_bulk",
    "bwd_avg_pkts_per_bulk",
    "bwd_header_length",
    "bwd_iat_max",
    "bwd_iat_min",
    "bwd_iat_std",
    "bwd_iat_total",
    "bwd_pkt_len_min",
    "bwd_pkt_len_skew",
    "bwd_pkt_len_std",
    "bwd_pkt_len_var",
    "flow_iat_min",
    "flow_iat_std",
    "fwd_avg_bulk_rate",
    "fwd_avg_bytes_per_bulk",
    "fwd_avg_pkts_per_bulk",
    "fwd_header_length",
    "fwd_iat_max",
    "fwd_iat_min",
    "fwd_iat_std",
    "fwd_iat_total",
    "fwd_pkt_len_min",
    "fwd_pkt_len_skew",
    "fwd_pkt_len_std",
    "fwd_pkt_len_var",
    "idle_min",
    "idle_std",
    "pkt_len_min",
    "pkt_len_skew",
    "pkt_len_var",
)

# ── Group 4: Network Endpoints (3 features: dst_port, src_port, protocol) ────
# Structural endpoint features (6 features)
GROUP_ENDPOINT_DERIVED: tuple[str, ...] = (
    "endpoint_last_dst_port",
    "endpoint_last_protocol",
    "endpoint_unique_dst_ports",
    "endpoint_unique_src_ports",
    "endpoint_tcp_ratio",
    "endpoint_udp_ratio",
)

# ── Group 5: Macro Network State Indicators (10 features) ─────────────────────
# Holistic cross-flow summaries (10 features)
GROUP_MACRO_INDICATORS: tuple[str, ...] = (
    "macro_total_bytes",
    "macro_total_packets",
    "macro_duration_sum_s",
    "macro_mean_byte_rate",
    "macro_mean_packet_rate",
    "macro_down_up_ratio",
    "macro_syn_ack_ratio",
    "macro_rst_ratio",
    "macro_port_dispersion",
    "macro_flow_density",
)


def _build_state_feature_names() -> tuple[str, ...]:
    """Construct the immutable list of 280 state feature names in strict order."""
    names: list[str] = []

    # 1. Continuous (30 x 5 = 150)
    for feat in GROUP_CONTINUOUS:
        for stat in ("mean", "std", "max", "last", "trend"):
            names.append(f"{feat}_{stat}")

    # 2. TCP Flags (12 x 4 = 48)
    for feat in GROUP_TCP_FLAGS:
        for stat in ("mean", "max", "last", "delta"):
            names.append(f"{feat}_{stat}")

    # 3. Shape (33 x 2 = 66)
    for feat in GROUP_SHAPE:
        for stat in ("mean", "last"):
            names.append(f"{feat}_{stat}")

    # 4. Endpoints (6)
    names.extend(GROUP_ENDPOINT_DERIVED)

    # 5. Macro (10)
    names.extend(GROUP_MACRO_INDICATORS)

    return tuple(names)


STATE_FEATURE_NAMES: tuple[str, ...] = _build_state_feature_names()
assert len(STATE_FEATURE_NAMES) == STATE_FEATURE_DIM, (
    f"Expected {STATE_FEATURE_DIM} state features, got {len(STATE_FEATURE_NAMES)}"
)


@dataclass(frozen=True, slots=True)
class StateSchemaMetadata:
    """Descriptor containing schema version, dimension, and feature names."""

    version: str
    dimension: int
    feature_names: list[str]
    input_feature_dim: int = 78
    dtype: str = "float32"


def get_state_schema_metadata() -> StateSchemaMetadata:
    """Return immutable state schema metadata."""
    return StateSchemaMetadata(
        version=STATE_SCHEMA_VERSION,
        dimension=STATE_FEATURE_DIM,
        feature_names=list(STATE_FEATURE_NAMES),
    )


def export_state_schema_json(dest_path: Path | str) -> Path:
    """Export the state schema metadata to a JSON file."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = get_state_schema_metadata()
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)
    return dest
