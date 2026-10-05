"""Tests for canonical feature schema in CyberHeal AI."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from shield_rca.telemetry.feature_schema import (
    CANONICAL_FEATURE_DIM,
    CANONICAL_FEATURES,
    CIC_COLUMN_ALIASES,
    SCHEMA_VERSION,
    align_dataframe_to_schema,
    export_schema_json,
    get_schema_metadata,
    validate_feature_vector,
)


def test_schema_dimension_and_ordering():
    assert len(CANONICAL_FEATURES) == 78
    assert CANONICAL_FEATURE_DIM == 78
    # Features must be strictly alphabetically sorted
    assert list(CANONICAL_FEATURES) == sorted(CANONICAL_FEATURES)
    meta = get_schema_metadata()
    assert meta.version == SCHEMA_VERSION
    assert meta.feature_count == 78


def test_align_dataframe_aliases():
    # Simulate a raw CIC-IDS2017 CSV format with human-readable headers
    raw_data = {
        "Flow Duration": [1000.0, 2000.0],
        "Total Fwd Packets": [5, 10],
        "Total Backward Packets": [3, 8],
        "Flow Bytes/s": [500.0, 1000.0],
        "Timestamp": [1499162400.0, 1499162410.0],
        "Label": ["BENIGN", "DDoS"],
    }
    raw_df = pd.DataFrame(raw_data)

    aligned = align_dataframe_to_schema(raw_df, fill_missing_features=True)

    # First 78 columns must be exactly CANONICAL_FEATURES
    assert list(aligned.columns[:78]) == list(CANONICAL_FEATURES)
    assert aligned["flow_duration_us"].tolist() == [1000.0, 2000.0]
    assert aligned["tot_fwd_pkts"].tolist() == [5.0, 10.0]
    assert aligned["flow_start_time"].tolist() == [1499162400.0, 1499162410.0]
    assert aligned["attack_class"].tolist() == ["BENIGN", "DDoS"]


def test_validate_feature_vector():
    good_vec = np.zeros(78, dtype=np.float32)
    assert validate_feature_vector(good_vec)

    # Wrong dimension
    bad_dim = np.zeros(70, dtype=np.float32)
    with pytest.raises(ValueError, match="does not match canonical dimension"):
        validate_feature_vector(bad_dim)

    # Non-finite value
    bad_nan = np.zeros(78, dtype=np.float32)
    bad_nan[5] = np.nan
    with pytest.raises(ValueError, match="contains NaN or Infinite"):
        validate_feature_vector(bad_nan)


def test_export_schema_json(tmp_path: Path):
    dest = tmp_path / "schema.json"
    exported = export_schema_json(dest)
    assert exported.exists()
    content = exported.read_text()
    assert '"feature_count": 78' in content
    assert '"version": "1.0.0"' in content
