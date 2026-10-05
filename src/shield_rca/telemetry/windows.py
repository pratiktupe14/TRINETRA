"""Multivariate time-series window builder and rolling buffer for CyberHeal AI.

Responsibility:
- Construct fixed-length sliding windows [N, T, F] from flow DataFrames.
- Provide WindowRingBuffer for real-time, in-memory rolling window aggregation.
- Support one-hot encoding for categorical variables.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WindowDataset:
    """In-memory container for sliding window tensors."""

    X: np.ndarray  # shape: (N, T, F) float32
    y: np.ndarray  # shape: (N,) or (N, T) int64
    feature_names: list[str]
    group_ids: list[str]


def encode_categorical_cols(
    df: pd.DataFrame,
    onehot_specs: list[dict],
) -> pd.DataFrame:
    """One-hot encode categorical integer columns into binary flag columns.

    Args:
        df: Flow-level DataFrame.
        onehot_specs: List of encoding specs. Each spec is a dict with:
            - "col": str — name of the source column.
            - "known_values": list[int] — category values to expand.
            - "value_names": list[str] | None — friendly suffixes.

    Returns:
        A copy of df with the new binary indicator columns appended.
    """
    df = df.copy()
    for spec in onehot_specs:
        col: str = spec["col"]
        if col not in df.columns:
            raise ValueError(f"encode_categorical_cols: column '{col}' not in DataFrame")
        known: list = spec["known_values"]
        names: list[str] = spec.get("value_names") or [str(v) for v in known]
        if len(names) != len(known):
            raise ValueError(
                f"encode_categorical_cols: value_names length {len(names)} "
                f"!= known_values length {len(known)} for column '{col}'"
            )
        for v, name in zip(known, names, strict=True):
            df[f"{col}_{name}"] = (df[col] == v).astype("int8")
        df[f"{col}_other"] = (~df[col].isin(known)).astype("int8")
    return df


_DEFAULT_BLOCKED_FEATURE_COLS: frozenset[str] = frozenset(
    {
        "flow_id",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "attack_class",
        "scenario_id",
        "flow_start_time",
    }
)


def build_timeseries_windows(
    flows_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    blocked_feature_cols: list[str] | None = None,
    label_col: str = "label_int",
    group_key: str = "scenario_id",
    time_col: str = "flow_start_time",
    window_size: int = 20,
    stride: int = 10,
    min_points_per_window: int = 1,
    sequence_label_rule: str = "majority",
    filter_benign_from_attack_groups: bool = False,
    log_transform_cols: list[str] | None = None,
    seq2seq: bool = False,
    add_dt_feature: bool = False,
) -> WindowDataset:
    """Construct multivariate sliding windows [N, T, F] from a flow DataFrame.

    Args:
        flows_df: DataFrame of flows.
        feature_cols: Feature columns to include. If None, auto-detected.
        blocked_feature_cols: Columns to exclude to prevent label/identity leakage.
        label_col: Integer label column.
        group_key: Column used to group consecutive flows (e.g. scenario_id or src_ip).
        time_col: Timestamp column used for intra-group sorting.
        window_size: Number of flows per window (T).
        stride: Step between window start positions.
        min_points_per_window: Minimum flows required to emit a window (zero-padded).
        sequence_label_rule: "majority", "last", or "any_attack" (for non-seq2seq mode).
        filter_benign_from_attack_groups: If True, drop pure BENIGN windows from attack groups.
        log_transform_cols: Columns to apply log1p transform.
        seq2seq: If True, y has shape (N, T); if False, y has shape (N,).
        add_dt_feature: If True, appends delta-time between flows as an extra feature.

    Returns:
        WindowDataset with X (N, T, F), y, feature_names, and group_ids.
    """
    df = flows_df.copy()

    # Auto-detect features if not specified
    if feature_cols is None:
        blocked = (
            frozenset(blocked_feature_cols)
            if blocked_feature_cols is not None
            else _DEFAULT_BLOCKED_FEATURE_COLS
        )
        feature_cols = [
            c
            for c in df.columns
            if c not in blocked
            and c != label_col
            and c != group_key
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    # Optional log1p transformation for heavy-tailed rate/byte features
    if log_transform_cols:
        for c in log_transform_cols:
            if c in df.columns:
                df[c] = np.log1p(np.clip(df[c].fillna(0).astype(float), a_min=0, a_max=None))

    # Fallback group key if not present
    if group_key not in df.columns:
        df[group_key] = "default_group"

    # Fill NaNs in features
    df[feature_cols] = df[feature_cols].fillna(0.0)

    has_labels = label_col in df.columns
    if not has_labels:
        df[label_col] = 0

    all_windows_X: list[np.ndarray] = []
    all_windows_y: list[Any] = []
    all_group_ids: list[str] = []

    f_dim = len(feature_cols) + (1 if add_dt_feature else 0)

    for gid, group_df in df.groupby(group_key, sort=False):
        if time_col in group_df.columns:
            group_df = group_df.sort_values(time_col)

        feats = group_df[feature_cols].to_numpy(dtype=np.float32)
        lbls = group_df[label_col].to_numpy(dtype=np.int64)

        dts = None
        if add_dt_feature and time_col in group_df.columns:
            ts = group_df[time_col].to_numpy(dtype=np.float64)
            dts = np.diff(ts, prepend=ts[0]).astype(np.float32)

        n_flows = len(group_df)
        if n_flows < min_points_per_window:
            continue

        for start_idx in range(0, n_flows, stride):
            end_idx = start_idx + window_size
            sub_feats = feats[start_idx:end_idx]
            sub_lbls = lbls[start_idx:end_idx]
            actual_len = len(sub_feats)

            if actual_len < min_points_per_window:
                break

            # Zero-padding if window is shorter than window_size
            if actual_len < window_size:
                pad_len = window_size - actual_len
                sub_feats = np.pad(sub_feats, ((0, pad_len), (0, 0)), mode="constant")
                if add_dt_feature:
                    sub_dts = np.pad(dts[start_idx:end_idx], (0, pad_len), mode="constant")
                if seq2seq:
                    sub_lbls = np.pad(sub_lbls, (0, pad_len), mode="constant", constant_values=-100)
            else:
                if add_dt_feature:
                    sub_dts = dts[start_idx:end_idx]

            if add_dt_feature:
                sub_feats = np.column_stack([sub_feats, sub_dts])

            # Label assignment
            if seq2seq:
                window_label = sub_lbls
            else:
                valid_lbls = sub_lbls[:actual_len]
                if sequence_label_rule == "majority":
                    counts = np.bincount(valid_lbls[valid_lbls >= 0], minlength=1)
                    window_label = int(np.argmax(counts))
                elif sequence_label_rule == "any_attack":
                    attack_lbls = valid_lbls[valid_lbls > 0]
                    window_label = int(attack_lbls[0]) if len(attack_lbls) > 0 else 0
                elif sequence_label_rule == "last":
                    window_label = int(valid_lbls[-1])
                else:
                    window_label = int(valid_lbls[0])

            all_windows_X.append(sub_feats)
            all_windows_y.append(window_label)
            all_group_ids.append(str(gid))

    if all_windows_X:
        X = np.stack(all_windows_X, axis=0).astype(np.float32)
        y = np.array(all_windows_y, dtype=np.int64)
    else:
        X = np.zeros((0, window_size, f_dim), dtype=np.float32)
        y = np.zeros((0, window_size) if seq2seq else (0,), dtype=np.int64)

    output_features = list(feature_cols)
    if add_dt_feature:
        output_features.append("dt")

    return WindowDataset(
        X=X,
        y=y,
        feature_names=output_features,
        group_ids=all_group_ids,
    )


class WindowRingBuffer:
    """Real-time rolling window ring buffer for live streaming ingestion.

    Maintains the most recent window_size flow vectors in memory. Once full,
    each new incoming flow produces an updated window tensor of shape (1, window_size, F)
    for real-time temporal anomaly scoring.
    """

    def __init__(self, window_size: int = 20, feature_names: list[str] | None = None):
        self.window_size = window_size
        self.feature_names = feature_names or []
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=window_size)

    def append(self, feature_vector: np.ndarray) -> None:
        """Add a single flow's numeric feature vector to the rolling buffer."""
        self._buffer.append(np.asarray(feature_vector, dtype=np.float32))

    @property
    def is_ready(self) -> bool:
        """Check if buffer has reached the required window size."""
        return len(self._buffer) == self.window_size

    def get_window(self) -> np.ndarray | None:
        """Return the current window tensor of shape (1, window_size, F), or None if not ready."""
        if not self.is_ready:
            return None
        return np.expand_dims(np.array(self._buffer, dtype=np.float32), axis=0)

    def reset(self) -> None:
        """Clear the ring buffer."""
        self._buffer.clear()
