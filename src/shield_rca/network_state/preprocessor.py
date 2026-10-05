"""State-Level Normalization and Preprocessor for CyberHeal AI.

Enforces strict leak-free scaling on the 280-dimensional Network State representation S_t:
- Fitted EXCLUSIVELY on training states.
- Normalizes both current states S_t and future target states S_{t+k} identically.
- Serialized alongside the state schema version.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from .state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class NetworkStatePreprocessor:
    """Preprocesses and normalizes 280-dimensional Network State vectors."""

    def __init__(
        self,
        scaler_type: str = "standard",
        schema_version: str = STATE_SCHEMA_VERSION,
        use_log1p: bool = False,
    ) -> None:
        raw_type = scaler_type.lower()
        if raw_type.startswith("log_"):
            self.use_log1p = True
            self.scaler_type = raw_type.replace("log_", "")
        else:
            self.use_log1p = use_log1p
            self.scaler_type = raw_type

        self.schema_version = schema_version
        self.is_fitted: bool = False
        self.non_neg_mask: np.ndarray | None = None

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(
                f"Unsupported scaler_type: '{scaler_type}'. Choose 'standard', 'robust', 'log_standard', or 'log_robust'."
            )

    def fit(self, states: np.ndarray) -> NetworkStatePreprocessor:
        """Fit scaler strictly on training state vectors.

        Args:
            states: 2D array of shape [N, 280].
        """
        arr = self._validate_state_matrix(states)
        if self.use_log1p:
            self.non_neg_mask = np.all(arr >= -1e-5, axis=0)
            arr_tf = arr.copy()
            arr_tf[:, self.non_neg_mask] = np.log1p(np.maximum(arr[:, self.non_neg_mask], 0.0))
            self.scaler.fit(arr_tf)
        else:
            self.scaler.fit(arr)
        self.is_fitted = True
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        """Transform state vectors using the fitted training scaler.

        Args:
            states: 1D array [280] or 2D array [N, 280].

        Returns:
            Normalized float32 array of matching shape.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "NetworkStatePreprocessor must be fit on training states before calling transform()"
            )

        if states.ndim == 3:
            N, T, D = states.shape
            flat = states.reshape(-1, D)
            scaled = self.transform(flat)
            return scaled.reshape(N, T, D)

        is_1d = states.ndim == 1
        arr = np.atleast_2d(states)
        arr = self._validate_state_matrix(arr)

        if self.use_log1p and self.non_neg_mask is not None:
            arr_tf = arr.copy()
            arr_tf[:, self.non_neg_mask] = np.log1p(np.maximum(arr[:, self.non_neg_mask], 0.0))
            scaled = self.scaler.transform(arr_tf).astype(np.float32)
        else:
            scaled = self.scaler.transform(arr).astype(np.float32)

        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return scaled[0] if is_1d else scaled

    def fit_transform(self, states: np.ndarray) -> np.ndarray:
        """Fit on training states and transform."""
        self.fit(states)
        return self.transform(states)

    def save(self, dest_dir: Path | str) -> Path:
        """Persist fitted state scaler and schema metadata."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted state preprocessor")

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        scaler_path = dest / "state_scaler.joblib"
        meta_path = dest / "state_preprocessor_meta.json"

        joblib.dump(self.scaler, scaler_path)

        meta = {
            "schema_version": self.schema_version,
            "scaler_type": self.scaler_type,
            "state_dim": STATE_FEATURE_DIM,
            "use_log1p": self.use_log1p,
            "non_neg_mask": self.non_neg_mask.tolist() if self.non_neg_mask is not None else None,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved state preprocessor to %s", dest)
        return dest

    @classmethod
    def load(cls, src_dir: Path | str) -> NetworkStatePreprocessor:
        """Load fitted state scaler and schema metadata from directory."""
        src = Path(src_dir)
        scaler_path = src / "state_scaler.joblib"
        meta_path = src / "state_preprocessor_meta.json"

        if not scaler_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing state preprocessor files in {src}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("schema_version") != STATE_SCHEMA_VERSION:
            logger.warning(
                "State preprocessor schema version mismatch: loaded %s, current %s",
                meta.get("schema_version"),
                STATE_SCHEMA_VERSION,
            )

        inst = cls(
            scaler_type=meta.get("scaler_type", "standard"),
            schema_version=meta.get("schema_version", STATE_SCHEMA_VERSION),
            use_log1p=meta.get("use_log1p", False),
        )
        inst.scaler = joblib.load(scaler_path)
        if meta.get("non_neg_mask") is not None:
            inst.non_neg_mask = np.array(meta["non_neg_mask"], dtype=bool)
        inst.is_fitted = True
        return inst

    def _validate_state_matrix(self, arr: np.ndarray) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"State data must be a numpy ndarray, got {type(arr)}")
        if arr.ndim != 2:
            raise ValueError(f"State matrix must be 2D [N, 280], got ndim={arr.ndim}")
        if arr.shape[1] != STATE_FEATURE_DIM:
            raise ValueError(
                f"State dimension mismatch: expected {STATE_FEATURE_DIM}, got {arr.shape[1]}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("State matrix contains non-finite (NaN or Inf) values")
        return arr.astype(np.float32)
