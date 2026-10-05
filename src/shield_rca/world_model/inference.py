"""Inference API for the CyberHeal World Model.

Predicts future continuous Network State vectors without triggering agent actions or risk scores:
    predict(current_state, horizon, delta_t) -> predicted_future_state
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from ..network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from .model import StateResidualDynamicsModel

logger = logging.getLogger(__name__)


class WorldModelInference:
    """Inference wrapper for the trained World Model."""

    def __init__(
        self,
        model: StateResidualDynamicsModel,
        device: str = "cpu",
        schema_version: str = STATE_SCHEMA_VERSION,
    ) -> None:
        self.model = model
        self.device = device
        self.schema_version = schema_version
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        current_state: np.ndarray,
        horizon: int = 1,
        delta_t: float = 1.0,
    ) -> dict[str, Any]:
        """Predict future continuous state S_hat_{t+k}.

        Args:
            current_state: 1D array [280] or 2D array [N, 280].
            horizon: Integer future step k >= 1.
            delta_t: Elapsed wall-clock time in seconds >= 0.0.

        Returns:
            Dictionary containing 'predicted_state' (np.ndarray) and metadata.
        """
        if horizon < 1:
            raise ValueError(f"Horizon k must be >= 1, got {horizon}")
        if delta_t < 0.0:
            raise ValueError(f"Elapsed time delta_t must be >= 0.0, got {delta_t}")

        is_1d = current_state.ndim == 1
        arr = np.atleast_2d(current_state).astype(np.float32)

        if arr.shape[1] != STATE_FEATURE_DIM:
            raise ValueError(
                f"State dimension mismatch: expected {STATE_FEATURE_DIM}, got {arr.shape[1]}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input state contains non-finite values (NaN or Inf)")

        N = len(arr)
        x_tensor = torch.from_numpy(arr).float().to(self.device)
        k_tensor = torch.full((N,), horizon, dtype=torch.long, device=self.device)
        dt_tensor = torch.full((N,), delta_t, dtype=torch.float, device=self.device)

        pred_tensor, delta_tensor = self.model(x_tensor, k_tensor, dt_tensor, return_delta=True)

        pred_np = pred_tensor.cpu().numpy().astype(np.float32)
        delta_np = delta_tensor.cpu().numpy().astype(np.float32)

        output_state = pred_np[0] if is_1d else pred_np
        output_delta = delta_np[0] if is_1d else delta_np

        return {
            "predicted_state": output_state,
            "predicted_delta": output_delta,
            "horizon": horizon,
            "delta_t": delta_t,
            "schema_version": self.schema_version,
            "state_dim": STATE_FEATURE_DIM,
        }
