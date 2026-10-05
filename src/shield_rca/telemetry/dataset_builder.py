"""Dataset Preparation Pipeline for CyberHeal AI.

Converts real network data (raw PCAP files, packet byte streams, or CIC-IDS CSVs)
into validated, chronologically-ordered multivariate temporal window batches [N, T, F]
ready for downstream Network State representations and World Model rollouts.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models.label_maps import assign_labels
from .feature_schema import (
    CANONICAL_FEATURE_DIM,
    CANONICAL_FEATURES,
    SCHEMA_VERSION,
    align_dataframe_to_schema,
)
from .flow_aggregator import aggregate_flows
from .flow_features import extract_flow_features
from .pcap_parser import iter_packets, iter_packets_from_bytes
from .validator import DataValidator, TelemetryValidationError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WindowBatch:
    """Production container for temporal window tensors and aligned metadata.

    Attributes:
        X: Feature tensor of shape [N, T, F], dtype float32.
           Represents historical observation windows [t - T + 1, ..., t].
        y: Target label array of shape [N] (window-level) or [N, T] (step-level).
        window_start_times: Array of starting POSIX timestamps for each window (length N).
        window_end_times: Array of ending POSIX timestamps for each window (length N).
        source_ids: Identifier strings (e.g. source IP or scenario) for each window (length N).
        feature_names: List of canonical feature names corresponding to dimension F (length F).
        schema_version: Version identifier string of the feature schema.
        target_horizon_indices: Reserved metadata slice/index mapping for future K-step prediction targets [t + 1, ..., t + K].
    """

    X: np.ndarray
    y: np.ndarray | None
    window_start_times: np.ndarray
    window_end_times: np.ndarray
    source_ids: list[str]
    feature_names: list[str]
    schema_version: str
    target_horizon_indices: list[tuple[int, int]] | None = None

    @property
    def num_windows(self) -> int:
        return int(self.X.shape[0])

    @property
    def window_size(self) -> int:
        return int(self.X.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.X.shape[2])


class DatasetBuilder:
    """Unified pipeline builder for PCAP and CSV telemetry data."""

    def __init__(
        self,
        window_size: int = 20,
        stride: int = 10,
        min_points_per_window: int = 1,
        sequence_label_rule: str = "majority",
        validator: DataValidator | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.min_points_per_window = min_points_per_window
        self.sequence_label_rule = sequence_label_rule
        self.validator = validator or DataValidator()
        self.schema_version = schema_version

    def from_pcap(
        self,
        pcap_source: Path | str | bytes,
        scenario_id: str = "live_pcap",
        default_label: str = "BENIGN",
    ) -> pd.DataFrame:
        """Parse raw PCAP file or byte buffer into a standardized flow DataFrame.

        Steps:
            PCAP -> RawPacketRecords -> 5-tuple Flows -> 80 Flow Features -> Canonical Schema DataFrame.
        """
        if isinstance(pcap_source, (bytes, io.BytesIO)):
            raw_bytes = pcap_source if isinstance(pcap_source, bytes) else pcap_source.getvalue()
            packets = iter_packets_from_bytes(raw_bytes)
        else:
            pcap_path = Path(pcap_source)
            if not pcap_path.exists():
                raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
            packets = iter_packets(pcap_path)

        flows = aggregate_flows(packets, mode="bidirectional")

        records: list[dict[str, Any]] = []
        for flow in flows:
            feat = extract_flow_features(flow)
            feat["scenario_id"] = scenario_id
            feat["attack_class"] = default_label
            records.append(feat)

        if not records:
            raise TelemetryValidationError("No completed flows could be extracted from PCAP source")

        raw_df = pd.DataFrame(records)

        # Map labels to integer index
        df_labeled = assign_labels(
            raw_df,
            raw_label_col="attack_class",
            int_label_col="label_int",
            unknown_strategy="drop",
        )

        # Align to canonical schema (F=78)
        aligned = align_dataframe_to_schema(df_labeled, fill_missing_features=True)

        # Validate
        validated = self.validator.validate_flow_dataframe(aligned, time_col="flow_start_time")

        # Strict chronological ordering
        sorted_df = validated.sort_values(by="flow_start_time", ascending=True).reset_index(drop=True)
        return sorted_df

    def from_csv(
        self,
        csv_source: Path | str | pd.DataFrame,
        time_col: str = "flow_start_time",
        label_col: str = "attack_class",
    ) -> pd.DataFrame:
        """Ingest and standardize a flow-level CSV file or DataFrame (e.g. CIC-IDS2017/2018).

        Steps:
            CSV -> Alias Normalization -> Label Assignment -> Canonical Schema -> Validation -> Sort.
        """
        if isinstance(csv_source, (Path, str)):
            path = Path(csv_source)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            df = pd.read_csv(path, low_memory=False)
        elif isinstance(csv_source, pd.DataFrame):
            df = csv_source.copy()
        else:
            raise TypeError(f"Unsupported csv_source type: {type(csv_source)}")

        # Align and map column aliases to canonical names
        aligned = align_dataframe_to_schema(df, fill_missing_features=True)

        # Label assignment if attack_class or label is present
        if "attack_class" in aligned.columns:
            aligned = assign_labels(
                aligned,
                raw_label_col="attack_class",
                int_label_col="label_int",
                unknown_strategy="drop",
            )
        elif "label_int" not in aligned.columns:
            aligned["label_int"] = 0  # Default benign if completely unlabelled

        if "scenario_id" not in aligned.columns:
            aligned["scenario_id"] = "csv_import"

        # Validate
        validated = self.validator.validate_flow_dataframe(aligned, time_col=time_col)

        # Strict chronological ordering
        sorted_df = validated.sort_values(by=time_col, ascending=True).reset_index(drop=True)
        return sorted_df

    def build_windows(
        self,
        df: pd.DataFrame,
        group_key: str = "scenario_id",
        time_col: str = "flow_start_time",
        label_col: str = "label_int",
        future_horizon_k: int = 0,
    ) -> WindowBatch:
        """Construct chronologically ordered sliding windows [N, T, F] and metadata.

        Semantics:
            Each window slice X[i] represents historical observation [t - T + 1, ..., t].
            The optional target_horizon_indices records the exact index range [t + 1, ..., t + K]
            in the underlying sequence for future prediction heads.

        Args:
            df: Validated, chronologically ordered flow DataFrame.
            group_key: Column name for session/scenario boundaries.
            time_col: Column name for timestamps.
            label_col: Column name for integer labels.
            future_horizon_k: Number of future time-steps (K) reserved for horizon metadata.

        Returns:
            WindowBatch dataclass with X [N, T, F], y, timestamps, and schema version.
        """
        # Ensure canonical sorting
        df_sorted = df.sort_values(by=[group_key, time_col], ascending=True).reset_index(drop=True)

        feature_matrix = df_sorted[list(CANONICAL_FEATURES)].to_numpy(dtype=np.float32)
        timestamps = df_sorted[time_col].to_numpy(dtype=float)
        labels = df_sorted[label_col].to_numpy(dtype=int) if label_col in df_sorted.columns else None
        sources = df_sorted[group_key].astype(str).tolist() if group_key in df_sorted.columns else ["default"] * len(df)

        window_X_list: list[np.ndarray] = []
        window_y_list: list[int] = []
        start_times_list: list[float] = []
        end_times_list: list[float] = []
        source_id_list: list[str] = []
        horizon_indices_list: list[tuple[int, int]] = []

        total_rows = len(df_sorted)
        T = self.window_size
        stride = self.stride

        # If grouping by scenario, slide within each scenario
        groups = df_sorted.groupby(group_key, sort=False)

        for gid, grp in groups:
            grp_indices = grp.index.to_numpy()
            n_grp = len(grp_indices)

            if n_grp < self.min_points_per_window:
                continue

            for start in range(0, n_grp, stride):
                end = start + T
                if end > n_grp:
                    # Partial tail window
                    tail_len = n_grp - start
                    if tail_len < self.min_points_per_window:
                        break
                    # Zero pad at start so observation aligns to end
                    pad_len = T - tail_len
                    win_feat = np.zeros((T, CANONICAL_FEATURE_DIM), dtype=np.float32)
                    win_feat[pad_len:] = feature_matrix[grp_indices[start:n_grp]]
                    w_start_ts = timestamps[grp_indices[start]]
                    w_end_ts = timestamps[grp_indices[n_grp - 1]]
                    win_labels = labels[grp_indices[start:n_grp]] if labels is not None else None
                    future_slice = (grp_indices[n_grp - 1] + 1, min(grp_indices[n_grp - 1] + 1 + future_horizon_k, total_rows))
                else:
                    win_feat = feature_matrix[grp_indices[start:end]]
                    w_start_ts = timestamps[grp_indices[start]]
                    w_end_ts = timestamps[grp_indices[end - 1]]
                    win_labels = labels[grp_indices[start:end]] if labels is not None else None
                    future_slice = (grp_indices[end - 1] + 1, min(grp_indices[end - 1] + 1 + future_horizon_k, total_rows))

                # Compute aggregate sequence label
                if win_labels is not None:
                    if self.sequence_label_rule == "any_attack":
                        attacks = win_labels[win_labels > 0]
                        lbl = int(attacks[0]) if len(attacks) > 0 else 0
                    elif self.sequence_label_rule == "last":
                        lbl = int(win_labels[-1])
                    else:  # majority
                        vals, counts = np.unique(win_labels, return_counts=True)
                        lbl = int(vals[np.argmax(counts)])
                    window_y_list.append(lbl)

                window_X_list.append(win_feat)
                start_times_list.append(w_start_ts)
                end_times_list.append(w_end_ts)
                source_id_list.append(str(gid))
                horizon_indices_list.append(future_slice)

        if not window_X_list:
            raise TelemetryValidationError(
                f"No windows could be formed. (Total rows={total_rows}, window_size={T})"
            )

        X = np.stack(window_X_list, axis=0)  # [N, T, F]
        y = np.array(window_y_list, dtype=np.int64) if window_y_list else None

        # Validate window batch
        self.validator.validate_window_batch(X, y, expected_T=T, expected_F=CANONICAL_FEATURE_DIM)

        return WindowBatch(
            X=X,
            y=y,
            window_start_times=np.array(start_times_list, dtype=float),
            window_end_times=np.array(end_times_list, dtype=float),
            source_ids=source_id_list,
            feature_names=list(CANONICAL_FEATURES),
            schema_version=self.schema_version,
            target_horizon_indices=horizon_indices_list if future_horizon_k > 0 else None,
        )
