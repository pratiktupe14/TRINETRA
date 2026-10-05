"""Leakage-safe temporal and scenario dataset splitting for CyberHeal AI.

Enforces strict chronological partitioning:
- TRAIN: earlier time period
- VALIDATION: middle time period
- TEST: latest time period
Random shuffling of time-series data is strictly prohibited to prevent lookahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Container for chronologically partitioned DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_time_range: tuple[float, float]
    val_time_range: tuple[float, float]
    test_time_range: tuple[float, float]


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "flow_start_time",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetSplits:
    """Partition a DataFrame strictly by timestamp into train, val, and test.

    Guarantees:
        max(train[time_col]) <= min(val[time_col]) <= max(val[time_col]) <= min(test[time_col])

    Args:
        df: Input DataFrame.
        time_col: Timestamp column name.
        train_ratio: Proportion for training (e.g. 0.70).
        val_ratio: Proportion for validation (e.g. 0.15).
        test_ratio: Proportion for testing (e.g. 0.15).

    Returns:
        DatasetSplits object with train, val, test partitions and time ranges.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in DataFrame")

    if len(df) < 3:
        raise ValueError(f"Dataset too small to split into 3 partitions (length={len(df)})")

    # Strictly sort by timestamp ascending if not already sorted
    if not df[time_col].is_monotonic_increasing:
        sorted_df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
    else:
        sorted_df = df

    n = len(sorted_df)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    # Ensure at least 1 sample in each split if possible
    train_idx = max(1, train_idx)
    val_idx = max(train_idx + 1, min(val_idx, n - 1))

    train_df = sorted_df.iloc[:train_idx]
    val_df = sorted_df.iloc[train_idx:val_idx]
    test_df = sorted_df.iloc[val_idx:]

    train_range = (float(train_df[time_col].min()), float(train_df[time_col].max()))
    val_range = (float(val_df[time_col].min()), float(val_df[time_col].max()))
    test_range = (float(test_df[time_col].min()), float(test_df[time_col].max()))

    # Verify no temporal backward leakage
    if train_range[1] > val_range[0]:
        logger.warning(
            "Boundary overlap at train/val boundary (equal timestamps): max_train=%s, min_val=%s",
            train_range[1],
            val_range[0],
        )
    if val_range[1] > test_range[0]:
        logger.warning(
            "Boundary overlap at val/test boundary (equal timestamps): max_val=%s, min_test=%s",
            val_range[1],
            test_range[0],
        )

    logger.info(
        "Chronological split: train=%d (%.1f%%, %s to %s), val=%d (%.1f%%, %s to %s), test=%d (%.1f%%, %s to %s)",
        len(train_df),
        (len(train_df) / n) * 100,
        train_range[0],
        train_range[1],
        len(val_df),
        (len(val_df) / n) * 100,
        val_range[0],
        val_range[1],
        len(test_df),
        (len(test_df) / n) * 100,
        test_range[0],
        test_range[1],
    )

    return DatasetSplits(
        train=train_df,
        val=val_df,
        test=test_df,
        train_time_range=train_range,
        val_time_range=val_range,
        test_time_range=test_range,
    )


def scenario_aware_split(
    df: pd.DataFrame,
    scenario_col: str = "scenario_id",
    time_col: str = "flow_start_time",
    train_scenarios: list[str] | None = None,
    val_scenarios: list[str] | None = None,
    test_scenarios: list[str] | None = None,
) -> DatasetSplits:
    """Split dataset by scenario/session IDs to ensure complete session isolation."""
    if scenario_col not in df.columns:
        raise KeyError(f"Scenario column '{scenario_col}' not found in DataFrame")

    available_scenarios = df[scenario_col].dropna().unique().tolist()

    if train_scenarios is None:
        # Sort scenarios chronologically by their earliest timestamp
        scenario_starts = df.groupby(scenario_col)[time_col].min().sort_values()
        sorted_scenarios = scenario_starts.index.tolist()
        num_scenarios = len(sorted_scenarios)
        if num_scenarios < 3:
            raise ValueError(
                f"At least 3 distinct scenarios required for scenario split, found {num_scenarios}"
            )
        train_end = int(num_scenarios * 0.70)
        val_end = int(num_scenarios * 0.85)
        train_scenarios = sorted_scenarios[: max(1, train_end)]
        val_scenarios = sorted_scenarios[max(1, train_end) : max(2, val_end)]
        test_scenarios = sorted_scenarios[max(2, val_end) :]

    train_df = df[df[scenario_col].isin(train_scenarios)].copy()
    val_df = df[df[scenario_col].isin(val_scenarios)].copy()
    test_df = df[df[scenario_col].isin(test_scenarios)].copy()

    train_range = (
        (float(train_df[time_col].min()), float(train_df[time_col].max()))
        if len(train_df) > 0
        else (0.0, 0.0)
    )
    val_range = (
        (float(val_df[time_col].min()), float(val_df[time_col].max()))
        if len(val_df) > 0
        else (0.0, 0.0)
    )
    test_range = (
        (float(test_df[time_col].min()), float(test_df[time_col].max()))
        if len(test_df) > 0
        else (0.0, 0.0)
    )

    return DatasetSplits(
        train=train_df,
        val=val_df,
        test=test_df,
        train_time_range=train_range,
        val_time_range=val_range,
        test_time_range=test_range,
    )
