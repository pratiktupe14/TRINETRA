"""Tests for DatasetBuilder and WindowBatch in CyberHeal AI."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from shield_rca.telemetry.dataset_builder import DatasetBuilder, WindowBatch
from shield_rca.telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES
from tests.fixtures.generate_test_pcap import create_synthetic_pcap


@pytest.fixture(scope="module")
def synthetic_pcap(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("dataset_builder_pcaps") / "test_builder.pcap"
    return create_synthetic_pcap(tmp)


def test_dataset_builder_from_pcap(synthetic_pcap: Path):
    builder = DatasetBuilder(window_size=2, stride=1, min_points_per_window=1)
    df = builder.from_pcap(synthetic_pcap, scenario_id="test_scen")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 4 flows in synthetic pcap
    assert list(df.columns[:78]) == list(CANONICAL_FEATURES)
    assert "flow_start_time" in df.columns
    assert "label_int" in df.columns

    # Build windows
    batch = builder.build_windows(df, group_key="scenario_id", future_horizon_k=2)
    assert isinstance(batch, WindowBatch)
    assert batch.X.ndim == 3
    assert batch.X.shape[1] == 2  # T=2
    assert batch.X.shape[2] == 78  # F=78
    assert batch.schema_version == "1.0.0"
    assert len(batch.window_start_times) == batch.num_windows
    assert len(batch.window_end_times) == batch.num_windows
    assert batch.target_horizon_indices is not None
    assert len(batch.target_horizon_indices) == batch.num_windows


def test_dataset_builder_from_csv(tmp_path: Path):
    builder = DatasetBuilder(window_size=5, stride=2, min_points_per_window=2)

    # Create dummy CIC-IDS style CSV
    csv_path = tmp_path / "cicids_sample.csv"
    rows = []
    for i in range(15):
        row = {
            "Timestamp": 1499162400.0 + i * 10,
            "Flow Duration": 1000.0 * (i + 1),
            "Total Fwd Packets": 10 + i,
            "Total Backward Packets": 5 + i,
            "Destination Port": 80,
            "Protocol": 6,
            "Label": "BENIGN" if i < 10 else "DDoS",
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = builder.from_csv(csv_path)
    assert len(df) == 15
    assert list(df.columns[:78]) == list(CANONICAL_FEATURES)
    assert df["label_int"].iloc[12] == 2  # DDoS mapped to 2

    # Chronologically sorted
    assert df["flow_start_time"].is_monotonic_increasing

    # Build windows
    batch = builder.build_windows(df, group_key="scenario_id")
    assert batch.X.shape[1] == 5  # T=5
    assert batch.X.shape[2] == 78  # F=78
    assert batch.num_windows >= 5
    assert batch.y is not None
