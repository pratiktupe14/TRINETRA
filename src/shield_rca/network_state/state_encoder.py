"""Deterministic Network State Encoder for CyberHeal AI.

Transforms historical temporal observation windows X_{t-T+1:t} in R^{T x 78}
into a standardized, compact, and mathematically principled Network State S_t in R^280.
Guarantees 100% determinism, zero data leakage, and zero inclusion of ground-truth labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..telemetry.dataset_builder import WindowBatch
from ..telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES
from .state_schema import (
    GROUP_CONTINUOUS,
    GROUP_ENDPOINT_DERIVED,
    GROUP_MACRO_INDICATORS,
    GROUP_SHAPE,
    GROUP_TCP_FLAGS,
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

# Precomputed index mappings for O(1) feature lookups
_FEAT_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_FEATURES)}
_CONTINUOUS_INDICES: list[int] = [_FEAT_TO_IDX[f] for f in GROUP_CONTINUOUS]
_FLAG_INDICES: list[int] = [_FEAT_TO_IDX[f] for f in GROUP_TCP_FLAGS]
_SHAPE_INDICES: list[int] = [_FEAT_TO_IDX[f] for f in GROUP_SHAPE]

_IDX_DST_PORT = _FEAT_TO_IDX["dst_port"]
_IDX_SRC_PORT = _FEAT_TO_IDX["src_port"]
_IDX_PROTOCOL = _FEAT_TO_IDX["protocol"]
_IDX_TOT_FWD_BYTES = _FEAT_TO_IDX["tot_fwd_bytes"]
_IDX_TOT_BWD_BYTES = _FEAT_TO_IDX["tot_bwd_bytes"]
_IDX_TOT_FWD_PKTS = _FEAT_TO_IDX["tot_fwd_pkts"]
_IDX_TOT_BWD_PKTS = _FEAT_TO_IDX["tot_bwd_pkts"]
_IDX_FLOW_DURATION = _FEAT_TO_IDX["flow_duration_us"]
_IDX_FLOW_BYTES_PER_S = _FEAT_TO_IDX["flow_bytes_per_s"]
_IDX_FLOW_PKTS_PER_S = _FEAT_TO_IDX["flow_packets_per_s"]
_IDX_SYN_FLAG = _FEAT_TO_IDX["syn_flag_cnt"]
_IDX_ACK_FLAG = _FEAT_TO_IDX["ack_flag_cnt"]
_IDX_RST_FLAG = _FEAT_TO_IDX["rst_flag_cnt"]

_EPS = 1e-6


@dataclass(slots=True)
class NetworkState:
    """Production container for a single deterministic Network State.

    Attributes:
        state_vector: 1D float32 array of dimension S=280.
        timestamp: Effective POSIX timestamp of the state (window end time t).
        window_start: Start timestamp of observation window (t - T + 1).
        window_end: End timestamp of observation window (t).
        source_id: Source host, interface, or scenario identifier.
        schema_version: Schema version string (e.g. '1.0.0').
        macro_metrics: Dictionary of macro network indicators for quick inspection.
    """

    state_vector: np.ndarray
    timestamp: float
    window_start: float
    window_end: float
    source_id: str
    schema_version: str
    macro_metrics: dict[str, float]

    @property
    def dimension(self) -> int:
        return len(self.state_vector)


@dataclass(slots=True)
class NetworkStateBatch:
    """Production container for a batch of Network States.

    Attributes:
        states: 2D float32 array of shape [N, 280].
        timestamps: 1D float array of shape [N] (window end times).
        window_starts: 1D float array of shape [N].
        window_ends: 1D float array of shape [N].
        source_ids: List of source identifiers of length N.
        feature_names: List of state feature names of length 280.
        schema_version: Schema version string.
    """

    states: np.ndarray
    timestamps: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    source_ids: list[str]
    feature_names: tuple[str, ...]
    schema_version: str
    labels: np.ndarray | None = None
    categories: list[str] | None = None

    @property
    def num_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])


class NetworkStateEncoder:
    """Encodes historical telemetry windows into deterministic Network State vectors."""

    def __init__(self, schema_version: str = STATE_SCHEMA_VERSION) -> None:
        self.schema_version = schema_version
        self.feature_names = STATE_FEATURE_NAMES

    def encode_window(
        self,
        window: np.ndarray,
        timestamp: float | None = None,
        window_start: float | None = None,
        window_end: float | None = None,
        source_id: str = "default",
    ) -> NetworkState:
        """Encode a single [T, 78] window matrix into a NetworkState.

        Args:
            window: 2D array of shape [T, 78], dtype float.
            timestamp: State timestamp (defaults to window_end or 0.0).
            window_start: Window start POSIX timestamp.
            window_end: Window end POSIX timestamp.
            source_id: Identifier string.

        Returns:
            NetworkState object containing state_vector in R^280.
        """
        self._validate_window_matrix(window)

        state_vec, macro_dict = self._compute_state_vector(window)

        w_end = window_end if window_end is not None else (timestamp if timestamp is not None else 0.0)
        w_start = window_start if window_start is not None else w_end
        effective_ts = timestamp if timestamp is not None else w_end

        return NetworkState(
            state_vector=state_vec,
            timestamp=float(effective_ts),
            window_start=float(w_start),
            window_end=float(w_end),
            source_id=str(source_id),
            schema_version=self.schema_version,
            macro_metrics=macro_dict,
        )

    def encode_batch(
        self,
        batch: WindowBatch | np.ndarray,
        labels: np.ndarray | None = None,
        categories: list[str] | None = None,
    ) -> NetworkStateBatch:
        """Encode a batch of windows into a NetworkStateBatch.

        Args:
            batch: WindowBatch object or 3D numpy array [N, T, 78].
            labels: Optional 1D array of ground truth labels [N].
            categories: Optional list of category names [N].

        Returns:
            NetworkStateBatch with states of shape [N, 280].
        """
        if isinstance(batch, WindowBatch):
            X = batch.X
            starts = batch.window_start_times
            ends = batch.window_end_times
            sources = batch.source_ids
            timestamps = ends
        elif isinstance(batch, np.ndarray):
            X = batch
            N = X.shape[0] if X.ndim == 3 else 0
            starts = np.zeros(N, dtype=float)
            ends = np.zeros(N, dtype=float)
            sources = ["default"] * N
            timestamps = np.zeros(N, dtype=float)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if X.ndim != 3:
            raise ValueError(f"Batch tensor must be 3D [N, T, F], got shape {X.shape}")

        N, T, F = X.shape
        if F != CANONICAL_FEATURE_DIM:
            raise ValueError(
                f"Feature dimension F must be {CANONICAL_FEATURE_DIM}, got {F}"
            )
        if T < 1:
            raise ValueError(f"Temporal dimension T must be >= 1, got {T}")
        if not np.all(np.isfinite(X)):
            raise ValueError("Input batch contains NaN or Infinite values")

        state_rows: list[np.ndarray] = []
        for i in range(N):
            s_vec, _ = self._compute_state_vector(X[i])
            state_rows.append(s_vec)

        states = np.stack(state_rows, axis=0) if state_rows else np.empty((0, STATE_FEATURE_DIM), dtype=np.float32)

        return NetworkStateBatch(
            states=states,
            timestamps=timestamps,
            window_starts=starts,
            window_ends=ends,
            source_ids=sources,
            feature_names=self.feature_names,
            schema_version=self.schema_version,
            labels=labels,
            categories=categories,
        )

    def _compute_state_vector(self, W: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Compute the deterministic 280-element state vector from a single [T, 78] window."""
        T = W.shape[0]

        # 1. Base Column Statistics across window
        means = np.mean(W, axis=0)
        stds = np.std(W, axis=0)
        maxs = np.max(W, axis=0)
        lasts = W[-1]
        firsts = W[0]
        deltas = lasts - firsts

        # Linear regression slope / trend across T
        if T >= 2:
            t = np.arange(T, dtype=float)
            t_centered = t - np.mean(t)
            denom = np.sum(t_centered**2) + _EPS
            slopes = np.dot(t_centered, W) / denom
            # Normalize slope by std to obtain dimensionless drift rate (z-score drift per step)
            trends = slopes / (stds + _EPS)
        else:
            trends = np.zeros(CANONICAL_FEATURE_DIM, dtype=float)

        state_values: list[float] = []

        # ── Group 1: Continuous (30 x 5 = 150) ──────────────────────────────
        for idx in _CONTINUOUS_INDICES:
            state_values.append(float(means[idx]))
            state_values.append(float(stds[idx]))
            state_values.append(float(maxs[idx]))
            state_values.append(float(lasts[idx]))
            state_values.append(float(trends[idx]))

        # ── Group 2: TCP Flags (12 x 4 = 48) ────────────────────────────────
        for idx in _FLAG_INDICES:
            state_values.append(float(means[idx]))
            state_values.append(float(maxs[idx]))
            state_values.append(float(lasts[idx]))
            state_values.append(float(deltas[idx]))

        # ── Group 3: Shape Metrics (33 x 2 = 66) ────────────────────────────
        for idx in _SHAPE_INDICES:
            state_values.append(float(means[idx]))
            state_values.append(float(lasts[idx]))

        # ── Group 4: Endpoints (6 derived metrics) ──────────────────────────
        last_dst_port = float(lasts[_IDX_DST_PORT])
        last_proto = float(lasts[_IDX_PROTOCOL])
        unique_dst = float(len(np.unique(W[:, _IDX_DST_PORT])))
        unique_src = float(len(np.unique(W[:, _IDX_SRC_PORT])))
        tcp_ratio = float(np.mean(W[:, _IDX_PROTOCOL] == 6))
        udp_ratio = float(np.mean(W[:, _IDX_PROTOCOL] == 17))

        state_values.extend([
            last_dst_port,
            last_proto,
            unique_dst,
            unique_src,
            tcp_ratio,
            udp_ratio,
        ])

        # ── Group 5: Macro Network Indicators (10 derived metrics) ──────────
        total_fwd_bytes = float(np.sum(W[:, _IDX_TOT_FWD_BYTES]))
        total_bwd_bytes = float(np.sum(W[:, _IDX_TOT_BWD_BYTES]))
        total_bytes = total_fwd_bytes + total_bwd_bytes

        total_fwd_pkts = float(np.sum(W[:, _IDX_TOT_FWD_PKTS]))
        total_bwd_pkts = float(np.sum(W[:, _IDX_TOT_BWD_PKTS]))
        total_packets = total_fwd_pkts + total_bwd_pkts

        dur_sum_s = float(np.sum(W[:, _IDX_FLOW_DURATION])) / 1e6
        mean_byte_rate = float(np.mean(W[:, _IDX_FLOW_BYTES_PER_S]))
        mean_pkt_rate = float(np.mean(W[:, _IDX_FLOW_PKTS_PER_S]))

        down_up_ratio = total_bwd_bytes / (total_fwd_bytes + _EPS)

        syn_sum = float(np.sum(W[:, _IDX_SYN_FLAG]))
        ack_sum = float(np.sum(W[:, _IDX_ACK_FLAG]))
        syn_ack_ratio = syn_sum / (ack_sum + _EPS)

        rst_sum = float(np.sum(W[:, _IDX_RST_FLAG]))
        rst_ratio = rst_sum / (total_packets + _EPS)

        port_dispersion = unique_dst / float(T)
        flow_density = float(T) / (dur_sum_s + _EPS)

        macro_dict: dict[str, float] = {
            "macro_total_bytes": total_bytes,
            "macro_total_packets": total_packets,
            "macro_duration_sum_s": dur_sum_s,
            "macro_mean_byte_rate": mean_byte_rate,
            "macro_mean_packet_rate": mean_pkt_rate,
            "macro_down_up_ratio": down_up_ratio,
            "macro_syn_ack_ratio": syn_ack_ratio,
            "macro_rst_ratio": rst_ratio,
            "macro_port_dispersion": port_dispersion,
            "macro_flow_density": flow_density,
        }

        for m_name in GROUP_MACRO_INDICATORS:
            state_values.append(macro_dict[m_name])

        state_arr = np.array(state_values, dtype=np.float32)
        # Ensure exact finiteness
        state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return state_arr, macro_dict

    def _validate_window_matrix(self, W: np.ndarray) -> None:
        """Strict validation of a single window matrix."""
        if not isinstance(W, np.ndarray):
            raise TypeError(f"Window must be a numpy ndarray, got {type(W)}")
        if W.ndim != 2:
            raise ValueError(f"Window matrix must be 2D [T, 78], got shape {W.shape}")
        T, F = W.shape
        if T < 1:
            raise ValueError(f"Window temporal length T must be >= 1, got {T}")
        if F != CANONICAL_FEATURE_DIM:
            raise ValueError(
                f"Window feature dimension F must be {CANONICAL_FEATURE_DIM}, got {F}"
            )
        if not np.all(np.isfinite(W)):
            raise ValueError("Window matrix contains NaN or Infinite values")
