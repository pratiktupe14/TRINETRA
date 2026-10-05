"""Tests for telemetry preprocessor and scaling in CyberHeal AI."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from shield_rca.telemetry.feature_schema import CANONICAL_FEATURES
from shield_rca.telemetry.preprocessor import TelemetryPreprocessor


@pytest.fixture
def train_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    train_dict = {f: np.random.uniform(10.0, 100.0, size=50) for f in CANONICAL_FEATURES}
    train_dict["label_int"] = np.random.randint(0, 2, size=50)
    train_dict["flow_start_time"] = np.linspace(100, 200, 50)

    test_dict = {f: np.random.uniform(10.0, 100.0, size=20) for f in CANONICAL_FEATURES}
    test_dict["label_int"] = np.random.randint(0, 2, size=20)
    test_dict["flow_start_time"] = np.linspace(201, 250, 20)

    return pd.DataFrame(train_dict), pd.DataFrame(test_dict)


def test_fit_only_on_train(train_test_data):
    train_df, test_df = train_test_data
    preprocessor = TelemetryPreprocessor(scaler_type="standard")

    assert not preprocessor.is_fitted
    with pytest.raises(RuntimeError, match="must be fit on training data"):
        preprocessor.transform(test_df)

    preprocessor.fit(train_df)
    assert preprocessor.is_fitted

    scaled_train = preprocessor.transform(train_df)
    scaled_test = preprocessor.transform(test_df)

    assert scaled_train.shape == (50, 78)
    assert scaled_test.shape == (20, 78)
    # Train mean should be close to 0
    assert np.allclose(scaled_train.mean(axis=0), 0.0, atol=1e-5)


def test_transform_3d_tensor(train_test_data):
    train_df, _ = train_test_data
    preprocessor = TelemetryPreprocessor(scaler_type="standard")
    preprocessor.fit(train_df)

    tensor = np.random.uniform(10.0, 100.0, size=(10, 5, 78)).astype(np.float32)
    scaled_tensor = preprocessor.transform(tensor)

    assert scaled_tensor.shape == (10, 5, 78)
    assert scaled_tensor.dtype == np.float32


def test_save_and_load_preprocessor(train_test_data, tmp_path: Path):
    train_df, test_df = train_test_data
    preprocessor = TelemetryPreprocessor(scaler_type="robust")
    preprocessor.fit(train_df)

    save_dir = tmp_path / "scaler_artifacts"
    preprocessor.save(save_dir)

    loaded = TelemetryPreprocessor.load(save_dir)
    assert loaded.is_fitted
    assert loaded.scaler_type == "robust"

    scaled_orig = preprocessor.transform(test_df)
    scaled_loaded = loaded.transform(test_df)
    assert np.allclose(scaled_orig, scaled_loaded)
