"""Tests for telemetry validator in CyberHeal AI."""

import numpy as np
import pandas as pd
import pytest

from shield_rca.telemetry.feature_schema import CANONICAL_FEATURES
from shield_rca.telemetry.validator import DataValidator, TelemetryValidationError


@pytest.fixture
def valid_flow_df() -> pd.DataFrame:
    data = {f: [1.0, 2.0, 3.0] for f in CANONICAL_FEATURES}
    data["flow_start_time"] = [100.0, 101.0, 102.0]
    data["flow_id"] = ["f1", "f2", "f3"]
    return pd.DataFrame(data)


def test_validator_passes_on_valid_data(valid_flow_df):
    validator = DataValidator()
    res = validator.validate_flow_dataframe(valid_flow_df)
    assert len(res) == 3


def test_validator_rejects_empty_data():
    validator = DataValidator()
    empty_df = pd.DataFrame()
    with pytest.raises(TelemetryValidationError, match="dataset is empty"):
        validator.validate_flow_dataframe(empty_df)


def test_validator_missing_timestamp(valid_flow_df):
    validator = DataValidator()
    df = valid_flow_df.drop(columns=["flow_start_time"])
    with pytest.raises(TelemetryValidationError, match="Required timestamp column"):
        validator.validate_flow_dataframe(df)


def test_validator_non_positive_timestamp(valid_flow_df):
    validator = DataValidator(strict_time=True)
    df = valid_flow_df.copy()
    df.loc[0, "flow_start_time"] = -5.0
    with pytest.raises(TelemetryValidationError, match="non-positive timestamp"):
        validator.validate_flow_dataframe(df)


def test_validator_rejects_nan_in_strict_mode(valid_flow_df):
    validator = DataValidator(strict_nan=True)
    df = valid_flow_df.copy()
    df.loc[0, "tot_fwd_pkts"] = np.nan
    with pytest.raises(TelemetryValidationError, match="Found 1 NaN"):
        validator.validate_flow_dataframe(df)


def test_validator_rejects_negative_physical_values(valid_flow_df):
    validator = DataValidator(strict_negative=True)
    df = valid_flow_df.copy()
    df.loc[0, "tot_fwd_pkts"] = -10.0
    with pytest.raises(TelemetryValidationError, match="Physically impossible negative values"):
        validator.validate_flow_dataframe(df)


def test_validator_window_batch_shape():
    validator = DataValidator()
    X = np.zeros((5, 20, 78), dtype=np.float32)
    y = np.zeros(5, dtype=np.int64)
    # Valid
    validator.validate_window_batch(X, y, expected_T=20, expected_F=78)

    # Wrong T
    with pytest.raises(TelemetryValidationError, match="temporal length mismatch"):
        validator.validate_window_batch(X, y, expected_T=15, expected_F=78)

    # Wrong F
    X_bad_F = np.zeros((5, 20, 50), dtype=np.float32)
    with pytest.raises(TelemetryValidationError, match="feature dimension mismatch"):
        validator.validate_window_batch(X_bad_F, y, expected_T=20, expected_F=78)
