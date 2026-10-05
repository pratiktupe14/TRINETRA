"""Loss and Evaluation Metrics for the CyberHeal World Model.

Computes:
- Huber Loss (Smooth L1) as the robust regression objective.
- Holistic error metrics (MAE, MSE, RMSE).
- Error breakdowns across the 5 Network State groups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Group slice boundaries in the 280-dimensional state vector:
# 1. Continuous: [0:150] (150 features)
# 2. TCP Flags:  [150:198] (48 features)
# 3. Shape:      [198:264] (66 features)
# 4. Endpoints:  [264:270] (6 features)
# 5. Macro:      [270:280] (10 features)
SLICE_CONTINUOUS = slice(0, 150)
SLICE_FLAGS = slice(150, 198)
SLICE_SHAPE = slice(198, 264)
SLICE_ENDPOINTS = slice(264, 270)
SLICE_MACRO = slice(270, 280)


class WorldModelLoss(nn.Module):
    """Huber regression loss with per-group error breakdown."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar Huber loss for backward pass."""
        return self.huber(predicted, target)

    @torch.no_grad()
    def compute_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute MAE, MSE, RMSE, and per-group MAEs."""
        diff = predicted - target
        abs_err = torch.abs(diff)
        sq_err = diff ** 2

        mae = float(torch.mean(abs_err).item())
        mse = float(torch.mean(sq_err).item())
        rmse = float(torch.sqrt(torch.mean(sq_err)).item())
        huber_val = float(self.huber(predicted, target).item())

        # Group-specific MAE
        mae_continuous = float(torch.mean(abs_err[:, SLICE_CONTINUOUS]).item())
        mae_flags = float(torch.mean(abs_err[:, SLICE_FLAGS]).item())
        mae_shape = float(torch.mean(abs_err[:, SLICE_SHAPE]).item())
        mae_endpoints = float(torch.mean(abs_err[:, SLICE_ENDPOINTS]).item())
        mae_macro = float(torch.mean(abs_err[:, SLICE_MACRO]).item())

        return {
            "huber_loss": huber_val,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mae_continuous": mae_continuous,
            "mae_flags": mae_flags,
            "mae_shape": mae_shape,
            "mae_endpoints": mae_endpoints,
            "mae_macro": mae_macro,
        }
