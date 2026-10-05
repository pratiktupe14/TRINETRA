"""Residual Dynamics World Model for CyberHeal AI.

Predicts future continuous Network State trajectories:
    S_hat_{t+k} = S_t + Delta_S_hat(S_t, k, Delta tau) in R^280
using a Residual MLP architecture conditioned on horizon k and elapsed time Delta tau.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..network_state.state_schema import STATE_FEATURE_DIM
from .time_conditioning import TimeHorizonConditioner


class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block with GELU non-linearity."""

    def __init__(self, hidden_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return residual + h


class StateResidualDynamicsModel(nn.Module):
    """World Model implementing continuous state transition S_hat_{t+k} = W_theta(S_t, k, Delta tau)."""

    def __init__(
        self,
        state_dim: int = STATE_FEATURE_DIM,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.05,
        max_horizon: int = 16,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # 1. Conditioning layer for (k, Delta tau) -> R^32
        self.conditioner = TimeHorizonConditioner(max_horizon=max_horizon)
        cond_dim = self.conditioner.output_dim

        # 2. Input projection [state_dim + cond_dim] -> [hidden_dim]
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3. Residual dynamics backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        # 4. Output head predicting the state transition delta Delta S_hat
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        current_state: torch.Tensor,
        horizon: torch.Tensor,
        delta_t: torch.Tensor,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting future state S_hat_{t+k}.

        Args:
            current_state: FloatTensor of shape [N, 280] (S_t).
            horizon: LongTensor of shape [N] (k).
            delta_t: FloatTensor of shape [N] (Delta tau).
            return_delta: If True, returns (S_hat, Delta_S_hat).

        Returns:
            predicted_state: FloatTensor of shape [N, 280] (S_hat_{t+k}).
        """
        # Embed time and horizon
        cond = self.conditioner(horizon, delta_t)  # [N, 32]

        # Handle 2D [N, 280] or 3D [N, T, 280] inputs (extract S_t if 3D)
        ref_state = current_state[:, -1, :] if current_state.ndim == 3 else current_state

        # Concatenate state and conditioning
        x = torch.cat([ref_state, cond], dim=-1)  # [N, 280 + 32]
        h = self.input_proj(x)  # [N, hidden_dim]

        for block in self.blocks:
            h = block(h)

        h = self.out_norm(h)
        delta_s = self.delta_head(h)  # [N, 280]

        # Residual dynamics formulation: S_hat_{t+k} = S_t + Delta_S_hat
        predicted_state = ref_state + delta_s

        if return_delta:
            return predicted_state, delta_s
        return predicted_state
