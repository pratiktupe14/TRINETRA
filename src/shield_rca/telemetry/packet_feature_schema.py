"""Packet-Level Feature Schema for TRINETRA / CyberHeal AI.

Defines the single source of truth for low-level packet features (F_pkt = 18)
enriching the flow-level pipeline in compliance with the SIH requirement:
"Use both flow-level AND packet-level features from open-source traffic data."

Features include:
1. IPv4 TTL & IPv6 Hop-Limit Statistics (min, max, mean, std, range, fwd/bwd mean)
2. TCP Retransmission Indicators (count, rate, fwd/bwd counts)
3. Packet Inter-Arrival Timing & Jitter (mean, std, min, max, p50, p95, jitter)

Note on Packet Count and Size:
Packet count and size distributions are already represented by the 78 canonical
flow features (tot_fwd_pkts, tot_bwd_pkts, pkt_len_mean, pkt_len_std, pkt_len_min,
pkt_len_max, fwd_pkt_len_*, bwd_pkt_len_*). Phase 8 adds missing packet-specific
signals without duplicate representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PACKET_SCHEMA_VERSION: str = "1.0.0"
PACKET_FEATURE_DIM: int = 18

MISSING_TTL_SENTINEL: float = -1.0

# Alphabetically sorted immutable tuple of canonical packet-level feature names
PACKET_FEATURE_NAMES: tuple[str, ...] = (
    "bwd_tcp_retrans_count",
    "bwd_ttl_mean",
    "fwd_tcp_retrans_count",
    "fwd_ttl_mean",
    "packet_iat_jitter",
    "packet_iat_max",
    "packet_iat_mean",
    "packet_iat_min",
    "packet_iat_p50",
    "packet_iat_p95",
    "packet_iat_std",
    "tcp_retransmission_count",
    "tcp_retransmission_rate",
    "ttl_max",
    "ttl_mean",
    "ttl_min",
    "ttl_range",
    "ttl_std",
)


@dataclass(slots=True, frozen=True)
class PacketFeatureVector:
    """Strongly typed container for the 18 packet-level features."""

    bwd_tcp_retrans_count: float
    bwd_ttl_mean: float
    fwd_tcp_retrans_count: float
    fwd_ttl_mean: float
    packet_iat_jitter: float
    packet_iat_max: float
    packet_iat_mean: float
    packet_iat_min: float
    packet_iat_p50: float
    packet_iat_p95: float
    packet_iat_std: float
    tcp_retransmission_count: float
    tcp_retransmission_rate: float
    ttl_max: float
    ttl_mean: float
    ttl_min: float
    ttl_range: float
    ttl_std: float

    def to_dict(self) -> dict[str, float]:
        """Return features as a dictionary keyed by canonical feature name."""
        return {name: float(getattr(self, name)) for name in PACKET_FEATURE_NAMES}

    def to_numpy(self) -> np.ndarray:
        """Return features as a 1D float32 numpy array in canonical order."""
        return np.array(
            [float(getattr(self, name)) for name in PACKET_FEATURE_NAMES],
            dtype=np.float32,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PacketFeatureVector:
        """Construct a PacketFeatureVector from a dictionary with missing defaults."""
        kwargs: dict[str, float] = {}
        for name in PACKET_FEATURE_NAMES:
            val = data.get(name)
            if val is None or not np.isfinite(val):
                if name.startswith("ttl") or name.endswith("ttl_mean"):
                    kwargs[name] = MISSING_TTL_SENTINEL
                else:
                    kwargs[name] = 0.0
            else:
                kwargs[name] = float(val)
        return cls(**kwargs)

    @classmethod
    def empty(cls) -> PacketFeatureVector:
        """Return a neutral vector representing missing / unobserved packet data."""
        return cls(
            bwd_tcp_retrans_count=0.0,
            bwd_ttl_mean=MISSING_TTL_SENTINEL,
            fwd_tcp_retrans_count=0.0,
            fwd_ttl_mean=MISSING_TTL_SENTINEL,
            packet_iat_jitter=0.0,
            packet_iat_max=0.0,
            packet_iat_mean=0.0,
            packet_iat_min=0.0,
            packet_iat_p50=0.0,
            packet_iat_p95=0.0,
            packet_iat_std=0.0,
            tcp_retransmission_count=0.0,
            tcp_retransmission_rate=0.0,
            ttl_max=MISSING_TTL_SENTINEL,
            ttl_mean=MISSING_TTL_SENTINEL,
            ttl_min=MISSING_TTL_SENTINEL,
            ttl_range=MISSING_TTL_SENTINEL,
            ttl_std=MISSING_TTL_SENTINEL,
        )


def validate_packet_features(features: dict[str, float] | np.ndarray) -> bool:
    """Validate dimensionality, finiteness, and physical boundaries.

    Args:
        features: Either a dict mapping feature name -> float or a 1D numpy array.

    Returns:
        True if valid.

    Raises:
        ValueError: If dimension mismatch, non-finite values, or physical bound violations occur.
    """
    if isinstance(features, np.ndarray):
        if features.ndim != 1 or len(features) != PACKET_FEATURE_DIM:
            raise ValueError(
                f"Packet feature vector dimension mismatch: expected ({PACKET_FEATURE_DIM},), "
                f"got {features.shape}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("Packet feature vector contains non-finite (NaN or Inf) values")
        feat_dict = {name: float(features[i]) for i, name in enumerate(PACKET_FEATURE_NAMES)}
    elif isinstance(features, dict):
        missing = set(PACKET_FEATURE_NAMES) - set(features.keys())
        if missing:
            raise ValueError(f"Packet feature dict missing required keys: {sorted(missing)}")
        feat_dict = features
    else:
        raise TypeError(f"Expected dict or np.ndarray, got {type(features).__name__}")

    # Check finiteness
    for k in PACKET_FEATURE_NAMES:
        val = feat_dict[k]
        if not np.isfinite(val):
            raise ValueError(f"Packet feature '{k}' is non-finite: {val}")

    # Physical boundary checks
    for k in (
        "tcp_retransmission_count",
        "fwd_tcp_retrans_count",
        "bwd_tcp_retrans_count",
    ):
        if feat_dict[k] < 0:
            raise ValueError(f"Packet feature '{k}' cannot be negative: {feat_dict[k]}")

    rate = feat_dict["tcp_retransmission_rate"]
    if not (0.0 <= rate <= 1.0 + 1e-6):
        raise ValueError(f"tcp_retransmission_rate must be in [0, 1], got {rate}")

    for k in (
        "packet_iat_mean",
        "packet_iat_std",
        "packet_iat_min",
        "packet_iat_max",
        "packet_iat_p50",
        "packet_iat_p95",
        "packet_iat_jitter",
    ):
        if feat_dict[k] < 0:
            raise ValueError(f"Timing feature '{k}' cannot be negative: {feat_dict[k]}")

    # TTL features: must be MISSING_TTL_SENTINEL (-1.0) or in [0, 255]
    for k in ("ttl_min", "ttl_max", "ttl_mean", "fwd_ttl_mean", "bwd_ttl_mean"):
        v = feat_dict[k]
        if abs(v - MISSING_TTL_SENTINEL) > 1e-5 and not (0.0 <= v <= 255.0):
            raise ValueError(f"TTL feature '{k}' out of valid range [-1, 255]: {v}")

    return True
