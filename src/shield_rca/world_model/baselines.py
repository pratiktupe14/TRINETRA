"""Baseline Models for Future-State Regression in CyberHeal AI.

Implements:
1. Persistence Baseline: S_hat_{t+k} = S_t
2. Linear Dynamics Baseline: Ridge regression on (S_t, k, Delta tau) -> S_{t+k}
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from ..network_state.future_dataset import FutureStatePairBatch
from .loss import (
    SLICE_CONTINUOUS,
    SLICE_ENDPOINTS,
    SLICE_FLAGS,
    SLICE_MACRO,
    SLICE_SHAPE,
)


class PersistenceBaseline:
    """Predicts no state change: S_hat_{t+k} = S_t."""

    def predict(self, current_states: np.ndarray) -> np.ndarray:
        """Return current states as the predicted future state."""
        if current_states.ndim == 3:
            return np.array(current_states[:, -1, :], copy=True, dtype=np.float32)
        return np.array(current_states, copy=True, dtype=np.float32)

    def evaluate(self, batch: FutureStatePairBatch) -> dict[str, float]:
        """Compute baseline error metrics against target states."""
        pred = self.predict(batch.current_states)
        targ = batch.target_states
        diff = np.abs(pred - targ)
        sq_diff = (pred - targ) ** 2

        return {
            "mae": float(np.mean(diff)),
            "mse": float(np.mean(sq_diff)),
            "rmse": float(np.sqrt(np.mean(sq_diff))),
            "mae_continuous": float(np.mean(diff[:, SLICE_CONTINUOUS])),
            "mae_flags": float(np.mean(diff[:, SLICE_FLAGS])),
            "mae_shape": float(np.mean(diff[:, SLICE_SHAPE])),
            "mae_endpoints": float(np.mean(diff[:, SLICE_ENDPOINTS])),
            "mae_macro": float(np.mean(diff[:, SLICE_MACRO])),
        }


class LinearDynamicsBaseline:
    """Linear / Ridge dynamics baseline estimating Delta S = W [S_t, k, log(1 + dt)] + b."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.is_fitted = False

    def fit(self, train_batch: FutureStatePairBatch) -> LinearDynamicsBaseline:
        """Fit linear dynamics on training delta: Delta S = S_{t+k} - S_t."""
        s_t = (
            train_batch.current_states[:, -1, :]
            if train_batch.current_states.ndim == 3
            else train_batch.current_states
        )
        X_in = self._build_features(
            s_t,
            train_batch.horizon,
            train_batch.delta_times,
        )
        delta_targets = train_batch.target_states - s_t
        self.model.fit(X_in, delta_targets)
        self.is_fitted = True
        return self

    def predict(
        self,
        current_states: np.ndarray,
        horizon: int,
        delta_times: np.ndarray,
    ) -> np.ndarray:
        """Predict S_hat_{t+k} = S_t + Delta S_hat."""
        if not self.is_fitted:
            raise RuntimeError("LinearDynamicsBaseline must be fit before calling predict()")
        s_t = current_states[:, -1, :] if current_states.ndim == 3 else current_states
        X_in = self._build_features(s_t, horizon, delta_times)
        pred_deltas = self.model.predict(X_in).astype(np.float32)
        return s_t + pred_deltas

    def evaluate(self, batch: FutureStatePairBatch) -> dict[str, float]:
        """Compute evaluation metrics on a pair batch."""
        pred = self.predict(batch.current_states, batch.horizon, batch.delta_times)
        targ = batch.target_states
        diff = np.abs(pred - targ)
        sq_diff = (pred - targ) ** 2

        return {
            "mae": float(np.mean(diff)),
            "mse": float(np.mean(sq_diff)),
            "rmse": float(np.sqrt(np.mean(sq_diff))),
            "mae_continuous": float(np.mean(diff[:, SLICE_CONTINUOUS])),
            "mae_flags": float(np.mean(diff[:, SLICE_FLAGS])),
            "mae_shape": float(np.mean(diff[:, SLICE_SHAPE])),
            "mae_endpoints": float(np.mean(diff[:, SLICE_ENDPOINTS])),
            "mae_macro": float(np.mean(diff[:, SLICE_MACRO])),
        }

    def _build_features(
        self,
        current_states: np.ndarray,
        horizon: int,
        delta_times: np.ndarray,
    ) -> np.ndarray:
        s_t = current_states[:, -1, :] if current_states.ndim == 3 else current_states
        N = len(s_t)
        h_col = np.full((N, 1), float(horizon), dtype=np.float32)
        dt_col = np.log1p(np.maximum(delta_times.reshape(N, 1), 0.0)).astype(np.float32)
        return np.hstack([s_t, h_col, dt_col])
