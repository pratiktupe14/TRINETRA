"""Time and Horizon Conditioning Layer for the CyberHeal World Model.

Embeds discrete future window horizons k in {1, 3, 5, ...} and continuous
elapsed wall-clock seconds Delta tau into a joint conditioning representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeHorizonConditioner(nn.Module):
    """Encodes discrete horizon step k and continuous elapsed time Delta tau."""

    def __init__(
        self,
        max_horizon: int = 16,
        horizon_dim: int = 16,
        time_dim: int = 16,
    ) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.horizon_dim = horizon_dim
        self.time_dim = time_dim
        self.output_dim = horizon_dim + time_dim

        # Horizon integer embedding (clamp to [0, max_horizon - 1])
        self.horizon_embed = nn.Embedding(max_horizon, horizon_dim)

        # Continuous delta_t MLP projection
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, horizon: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            horizon: LongTensor of shape [N] (e.g. 1, 3, 5).
            delta_t: FloatTensor of shape [N] (elapsed seconds).

        Returns:
            Conditioning tensor of shape [N, 32].
        """
        # Ensure correct shape and range
        h_clamped = torch.clamp(horizon.long(), min=0, max=self.max_horizon - 1)
        h_emb = self.horizon_embed(h_clamped)  # [N, horizon_dim]

        # Log transform continuous time to stabilize wide dynamic ranges: log1p(max(0, dt))
        dt_safe = torch.log1p(torch.clamp(delta_t.float().unsqueeze(-1), min=0.0))
        t_emb = self.time_mlp(dt_safe)  # [N, time_dim]

        return torch.cat([h_emb, t_emb], dim=-1)  # [N, 32]
