"""Tests for chronological dataset splitting in CyberHeal AI."""

import numpy as np
import pandas as pd
import pytest

from shield_rca.telemetry.split import chronological_split, scenario_aware_split


@pytest.fixture
def chronological_df() -> pd.DataFrame:
    timestamps = np.linspace(1_000_000, 2_000_000, 100)
    return pd.DataFrame({
        "flow_start_time": timestamps,
        "tot_fwd_pkts": np.arange(100),
        "scenario_id": ["scen_A"] * 50 + ["scen_B"] * 30 + ["scen_C"] * 20,
    })


def test_chronological_split_zero_overlap(chronological_df):
    splits = chronological_split(
        chronological_df,
        time_col="flow_start_time",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    assert len(splits.train) == 70
    assert len(splits.val) == 15
    assert len(splits.test) == 15

    # Strict zero temporal overlap
    assert splits.train["flow_start_time"].max() <= splits.val["flow_start_time"].min()
    assert splits.val["flow_start_time"].max() <= splits.test["flow_start_time"].min()


def test_scenario_aware_split(chronological_df):
    splits = scenario_aware_split(
        chronological_df,
        scenario_col="scenario_id",
        time_col="flow_start_time",
    )

    train_scenarios = set(splits.train["scenario_id"].unique())
    val_scenarios = set(splits.val["scenario_id"].unique())
    test_scenarios = set(splits.test["scenario_id"].unique())

    # Ensure no scenario appears in multiple splits
    assert train_scenarios.isdisjoint(val_scenarios)
    assert val_scenarios.isdisjoint(test_scenarios)
    assert train_scenarios.isdisjoint(test_scenarios)


def test_split_invalid_ratios(chronological_df):
    with pytest.raises(ValueError, match="must sum to 1.0"):
        chronological_split(chronological_df, train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
