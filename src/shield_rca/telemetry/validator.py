"""Data validation layer for CyberHeal AI.

Enforces strict physical constraints, schema compliance, temporal consistency,
and data integrity on raw and processed network telemetry.
Fails loudly on invalid production data to prevent silent model poisoning.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .feature_schema import (
    CANONICAL_FEATURE_DIM,
    CANONICAL_FEATURES,
    NON_NEGATIVE_FEATURES,
)

logger = logging.getLogger(__name__)


class TelemetryValidationError(ValueError):
    """Raised when telemetry data fails scientific or integrity validation checks."""

    pass


class DataValidator:
    """Production validator for network flow telemetry datasets."""

    def __init__(
        self,
        strict_nan: bool = True,
        strict_negative: bool = True,
        strict_time: bool = True,
        drop_duplicates: bool = True,
    ) -> None:
        self.strict_nan = strict_nan
        self.strict_negative = strict_negative
        self.strict_time = strict_time
        self.drop_duplicates = drop_duplicates

    def validate_flow_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = "flow_start_time",
        min_rows: int = 1,
    ) -> pd.DataFrame:
        """Validate a flow DataFrame against canonical requirements.

        Args:
            df: DataFrame containing flow records.
            time_col: Timestamp column name.
            min_rows: Minimum acceptable rows.

        Returns:
            Validated (and potentially deduplicated) DataFrame.

        Raises:
            TelemetryValidationError: If any critical validation check fails.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise TelemetryValidationError("Input data must be a non-null pandas DataFrame")

        # 1. Empty dataset check
        if len(df) == 0:
            raise TelemetryValidationError("Telemetry dataset is empty (0 rows)")

        if len(df) < min_rows:
            raise TelemetryValidationError(
                f"Insufficient flow records: found {len(df)}, minimum required is {min_rows}"
            )

        # 2. Timestamp column check
        if time_col not in df.columns:
            raise TelemetryValidationError(
                f"Required timestamp column '{time_col}' is missing from DataFrame"
            )

        ts = pd.to_numeric(df[time_col], errors="coerce")
        if ts.isna().any():
            invalid_cnt = int(ts.isna().sum())
            raise TelemetryValidationError(
                f"Found {invalid_cnt} invalid/unparseable timestamps in column '{time_col}'"
            )

        if self.strict_time:
            if (ts <= 0).any():
                non_pos = int((ts <= 0).sum())
                raise TelemetryValidationError(
                    f"Found {non_pos} non-positive timestamp values in column '{time_col}'"
                )

        # 3. Canonical feature presence check
        missing_features = [f for f in CANONICAL_FEATURES if f not in df.columns]
        if missing_features:
            raise TelemetryValidationError(
                f"Missing {len(missing_features)} canonical features. First missing: {missing_features[:5]}"
            )

        # 4. Inconsistent feature dimensions check
        numeric_cols = [c for c in CANONICAL_FEATURES if c in df.columns]
        if len(numeric_cols) != CANONICAL_FEATURE_DIM:
            raise TelemetryValidationError(
                f"Inconsistent feature dimension: expected {CANONICAL_FEATURE_DIM}, got {len(numeric_cols)}"
            )

        # 5. NaN / Infinity checks
        feature_data = df[list(CANONICAL_FEATURES)]
        nan_mask = feature_data.isna().to_numpy()
        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            bad_cols = [col for col in CANONICAL_FEATURES if df[col].isna().any()]
            if self.strict_nan:
                raise TelemetryValidationError(
                    f"Found {nan_count} NaN values across columns: {bad_cols[:5]}"
                )
            logger.warning("Replacing %d NaN values with 0.0 in columns: %s", nan_count, bad_cols[:5])
            df = df.copy()
            df[list(CANONICAL_FEATURES)] = df[list(CANONICAL_FEATURES)].fillna(0.0)

        inf_mask = np.isinf(feature_data.to_numpy(dtype=float))
        if inf_mask.any():
            inf_count = int(inf_mask.sum())
            if self.strict_nan:
                raise TelemetryValidationError(
                    f"Found {inf_count} Infinite values in numerical features"
                )
            logger.warning("Capping %d Infinite values with 0.0", inf_count)
            df = df.copy()
            df[list(CANONICAL_FEATURES)] = df[list(CANONICAL_FEATURES)].replace(
                [np.inf, -np.inf], 0.0
            )

        # 6. Physical impossibility checks (negative counts, bytes, durations)
        if self.strict_negative:
            for feat in NON_NEGATIVE_FEATURES:
                if feat in df.columns:
                    neg_mask = df[feat] < 0
                    if neg_mask.any():
                        neg_cnt = int(neg_mask.sum())
                        min_val = float(df[feat].min())
                        raise TelemetryValidationError(
                            f"Physically impossible negative values in '{feat}': {neg_cnt} rows (min={min_val})"
                        )

        # 7. Deduplication
        if self.drop_duplicates:
            subset = ["flow_id"] if "flow_id" in df.columns else [time_col, "src_port", "dst_port", "protocol"]
            subset_existing = [c for c in subset if c in df.columns]
            if subset_existing:
                orig_len = len(df)
                df = df.drop_duplicates(subset=subset_existing)
                dropped = orig_len - len(df)
                if dropped > 0:
                    logger.info("Deduplicated %d duplicate flow records (from %d to %d)", dropped, orig_len, len(df))

        return df

    def validate_window_batch(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        expected_T: int,
        expected_F: int = CANONICAL_FEATURE_DIM,
    ) -> None:
        """Validate an emitted [N, T, F] window tensor."""
        if X.ndim != 3:
            raise TelemetryValidationError(
                f"Window tensor must have 3 dimensions [N, T, F], got shape {X.shape}"
            )
        N, T, F = X.shape
        if T != expected_T:
            raise TelemetryValidationError(
                f"Window temporal length mismatch: expected T={expected_T}, got T={T}"
            )
        if F != expected_F:
            raise TelemetryValidationError(
                f"Window feature dimension mismatch: expected F={expected_F}, got F={F}"
            )
        if not np.all(np.isfinite(X)):
            raise TelemetryValidationError("Window tensor X contains non-finite (NaN or Inf) values")
        if y is not None:
            if y.shape[0] != N:
                raise TelemetryValidationError(
                    f"Label array length mismatch: X has N={N} windows, but y has {y.shape[0]}"
                )
