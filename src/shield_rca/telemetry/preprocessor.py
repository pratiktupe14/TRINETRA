"""Normalization and scaling preprocessor for CyberHeal AI.

Enforces strict leak-free scaling:
- Scaler is fit EXCLUSIVELY on training data.
- Validation, test, and live inference partitions are transformed using the fitted scaler.
- Labels and non-feature metadata are strictly protected from scaling.
- Scaler is serialized alongside the canonical feature schema version.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .feature_schema import (
    CANONICAL_FEATURE_DIM,
    CANONICAL_FEATURES,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


class TelemetryPreprocessor:
    """Preprocesses and normalizes canonical telemetry feature matrices and tensors."""

    def __init__(
        self,
        scaler_type: str = "standard",
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        """Initialize preprocessor.

        Args:
            scaler_type: "standard" (StandardScaler) or "robust" (RobustScaler).
            schema_version: Schema version string to lock compatibility.
        """
        self.scaler_type = scaler_type.lower()
        self.schema_version = schema_version
        self.feature_names: list[str] = list(CANONICAL_FEATURES)
        self.is_fitted: bool = False

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type: '{scaler_type}'. Choose 'standard' or 'robust'.")

    def fit(self, X: pd.DataFrame | np.ndarray) -> TelemetryPreprocessor:
        """Fit scaler strictly on training feature data.

        Args:
            X: 2D array or DataFrame of shape (N, F) where F=78.
        """
        mat = self._extract_feature_matrix(X)
        self.scaler.fit(mat)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform feature data using the fitted training scaler.

        Args:
            X: 2D array or DataFrame (N, F) or 3D tensor (N, T, F).

        Returns:
            Normalized numpy array of matching shape.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit on training data before calling transform()")

        if isinstance(X, np.ndarray) and X.ndim == 3:
            N, T, F = X.shape
            if F != CANONICAL_FEATURE_DIM:
                raise ValueError(f"Expected F={CANONICAL_FEATURE_DIM}, got {F}")
            reshaped = X.reshape(-1, F)
            scaled = self.scaler.transform(reshaped)
            return scaled.reshape(N, T, F).astype(np.float32)

        mat = self._extract_feature_matrix(X)
        scaled = self.scaler.transform(mat)
        return scaled.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit on training data and transform."""
        self.fit(X)
        return self.transform(X)

    def save(self, dest_dir: Path | str) -> Path:
        """Persist fitted scaler and schema metadata to directory."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor")

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        scaler_path = dest / "scaler.joblib"
        meta_path = dest / "preprocessor_meta.json"

        joblib.dump(self.scaler, scaler_path)

        meta = {
            "schema_version": self.schema_version,
            "scaler_type": self.scaler_type,
            "feature_dim": CANONICAL_FEATURE_DIM,
            "feature_names": self.feature_names,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved preprocessor to %s", dest)
        return dest

    @classmethod
    def load(cls, src_dir: Path | str) -> TelemetryPreprocessor:
        """Load fitted preprocessor and schema metadata from directory."""
        src = Path(src_dir)
        scaler_path = src / "scaler.joblib"
        meta_path = src / "preprocessor_meta.json"

        if not scaler_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing preprocessor files in {src}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                "Preprocessor schema version mismatch: loaded %s, current %s",
                meta.get("schema_version"),
                SCHEMA_VERSION,
            )

        inst = cls(scaler_type=meta["scaler_type"], schema_version=meta["schema_version"])
        inst.scaler = joblib.load(scaler_path)
        inst.feature_names = meta.get("feature_names", list(CANONICAL_FEATURES))
        inst.is_fitted = True
        return inst

    def _extract_feature_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Safely extract (N, F) feature matrix in canonical order."""
        if isinstance(X, pd.DataFrame):
            missing = [f for f in CANONICAL_FEATURES if f not in X.columns]
            if missing:
                raise ValueError(f"DataFrame is missing canonical features: {missing[:5]}")
            arr = X[list(CANONICAL_FEATURES)].to_numpy(dtype=np.float32)
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != CANONICAL_FEATURE_DIM:
                raise ValueError(
                    f"Feature array must be 2D with shape (N, {CANONICAL_FEATURE_DIM}), got {X.shape}"
                )
            arr = X.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type for preprocessor: {type(X)}")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Feature matrix contains non-finite (NaN or Inf) values")
        return arr
