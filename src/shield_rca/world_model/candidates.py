"""TRINETRA World Model Candidate Architectures for Phase 10 Model Improvement.

Provides candidate models for temporal Network State forecasting:
- Candidate A: StateResidualDynamicsModel (Residual MLP) with proper state normalization.
- Candidate B: TemporalContextResidualModel (Group-Gated Multi-Scale Residual Dynamics).
- Candidate C: GRUDynamicsWorldModel (Recurrent GRU Temporal Dynamics).

All candidates:
1. Strictly implement the standard continuous transition interface:
       forward(current_state, horizon, delta_t, return_delta=False)
2. Predict future state via residual dynamics:
       S_hat_{t+k} = S_t + Delta_S_hat(S_t, k, Delta tau)
3. Maintain full compatibility with WorldModelEvaluator, PredictiveRiskEstimator,
   AttackStageMapper, and Explainability engines.
4. Are isolated under artifacts/world_model/candidates/ without overwriting production weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..network_state.state_schema import (
    STATE_FEATURE_DIM,
)
from .loss import (
    SLICE_CONTINUOUS,
    SLICE_ENDPOINTS,
    SLICE_FLAGS,
    SLICE_MACRO,
    SLICE_SHAPE,
)
from .model import ResidualBlock, StateResidualDynamicsModel
from .time_conditioning import TimeHorizonConditioner


class GroupFeatureProjection(nn.Module):
    """Group-specific linear projection with LayerNorm and GELU non-linearity."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TemporalContextResidualModel(nn.Module):
    """Candidate B: Residual dynamics model with group-wise temporal gating.

    Disentangles the 5 network state feature groups (continuous, flags, shape,
    endpoints, macro) into specialized subspace embeddings, applies channel-wise
    gating, and predicts residual state evolution conditioned on horizon k and elapsed time Delta tau.
    """

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

        # 2. Group-specific feature projectors (total output = 64 * 3 + 32 * 2 = 256)
        # Continuous (150 -> 64)
        self.proj_cont = GroupFeatureProjection(150, 64)
        # TCP Flags (48 -> 64)
        self.proj_flags = GroupFeatureProjection(48, 64)
        # Shape (66 -> 64)
        self.proj_shape = GroupFeatureProjection(66, 64)
        # Endpoints (6 -> 32)
        self.proj_endpoints = GroupFeatureProjection(6, 32)
        # Macro (10 -> 32)
        self.proj_macro = GroupFeatureProjection(10, 32)

        total_proj_dim = 64 + 64 + 64 + 32 + 32  # 256

        # 3. Context fusion: fuses projected state groups + conditioning
        self.fusion = nn.Sequential(
            nn.Linear(total_proj_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 4. Gating mechanism: adaptive feature importance per group
        self.gate = nn.Sequential(
            nn.Linear(total_proj_dim + cond_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # 5. Residual dynamics backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        # 6. Output projection head predicting state transition Delta S_hat
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        current_state: torch.Tensor,
        horizon: torch.Tensor,
        delta_t: torch.Tensor,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting future state S_hat_{t+k}."""
        # Embed time and horizon
        cond = self.conditioner(horizon, delta_t)  # [N, 32]

        # Extract state feature groups
        c_cont = current_state[:, SLICE_CONTINUOUS]
        c_flags = current_state[:, SLICE_FLAGS]
        c_shape = current_state[:, SLICE_SHAPE]
        c_endp = current_state[:, SLICE_ENDPOINTS]
        c_macro = current_state[:, SLICE_MACRO]

        # Project groups
        h_cont = self.proj_cont(c_cont)
        h_flags = self.proj_flags(c_flags)
        h_shape = self.proj_shape(c_shape)
        h_endp = self.proj_endpoints(c_endp)
        h_macro = self.proj_macro(c_macro)

        h_grouped = torch.cat([h_cont, h_flags, h_shape, h_endp, h_macro], dim=-1)  # [N, 256]

        # Context fusion with temporal conditioning
        h_in = torch.cat([h_grouped, cond], dim=-1)
        fused = self.fusion(h_in)
        gates = self.gate(h_in)
        h = fused * gates

        for block in self.blocks:
            h = block(h)

        h = self.out_norm(h)
        delta_s = self.delta_head(h)  # [N, 280]

        predicted_state = current_state + delta_s

        if return_delta:
            return predicted_state, delta_s
        return predicted_state


class GRUDynamicsWorldModel(nn.Module):
    """Candidate C: Recurrent GRU dynamics model for temporal network evolution.

    Models continuous state evolution using a multi-layer Gated Recurrent Unit (GRU)
    backbone with horizon and elapsed-time conditioning.
    """

    def __init__(
        self,
        state_dim: int = STATE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        max_horizon: int = 16,
        bidirectional: bool = False,
        require_temporal_sequence: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_blocks = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.require_temporal_sequence = require_temporal_sequence

        # 1. Conditioning layer for (k, Delta tau) -> R^32
        self.conditioner = TimeHorizonConditioner(max_horizon=max_horizon)
        cond_dim = self.conditioner.output_dim

        # 2. Input projection into GRU hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 3. Recurrent backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 4. Output head combining GRU representation + conditioning
        self.delta_head = nn.Sequential(
            nn.Linear(gru_out_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        horizon: torch.Tensor,
        delta_t: torch.Tensor,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting future state S_hat_{t+k}."""
        # Embed time and horizon
        cond = self.conditioner(horizon, delta_t)  # [N, 32]

        # Handle 2D [N, 280] or 3D [N, T, 280] inputs
        if current_state.ndim == 2:
            if self.require_temporal_sequence:
                raise ValueError(
                    f"GRUDynamicsWorldModel configured with require_temporal_sequence=True "
                    f"requires 3D temporal sequence [N, T, 280] with T > 1, got 2D shape {current_state.shape}"
                )
            x_seq = current_state.unsqueeze(1)  # [N, 1, 280]
        elif current_state.ndim == 3:
            if self.require_temporal_sequence and current_state.shape[1] <= 1:
                raise ValueError(
                    f"GRUDynamicsWorldModel requires sequence length T > 1, got T={current_state.shape[1]}"
                )
            x_seq = current_state
        else:
            raise ValueError(f"Expected current_state to have 2 or 3 dimensions, got {current_state.ndim}")

        h_proj = self.input_proj(x_seq)  # [N, seq_len, hidden_dim]
        gru_out, _ = self.gru(h_proj)
        h_last = gru_out[:, -1, :]  # [N, hidden_dim]

        # Combine with conditioning
        combined = torch.cat([h_last, cond], dim=-1)  # [N, hidden_dim + 32]
        delta_s = self.delta_head(combined)  # [N, 280]

        # S_hat = S_t + delta_s
        ref_state = current_state if current_state.ndim == 2 else current_state[:, -1, :]
        predicted_state = ref_state + delta_s

        if return_delta:
            return predicted_state, delta_s
        return predicted_state


class TemporalAttention(nn.Module):
    """Temporal attention pooling over history steps T."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = 1.0 / (hidden_dim ** 0.5)

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        # gru_out: [N, T, hidden_dim]
        q = self.query(gru_out[:, -1:, :])  # [N, 1, hidden_dim]
        k = self.key(gru_out)               # [N, T, hidden_dim]
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [N, 1, T]
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, gru_out).squeeze(1)  # [N, hidden_dim]
        return attended


class GRUAttentionDynamicsWorldModel(nn.Module):
    """Candidate F: Recurrent GRU dynamics model with temporal attention pooling."""

    def __init__(
        self,
        state_dim: int = STATE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        max_horizon: int = 16,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_blocks = num_layers
        self.dropout = dropout

        # 1. Conditioning layer for (k, Delta tau) -> R^32
        self.conditioner = TimeHorizonConditioner(max_horizon=max_horizon)
        cond_dim = self.conditioner.output_dim

        # 2. Input projection into GRU hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3. Recurrent backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Temporal Attention Pooling
        self.attention = TemporalAttention(hidden_dim)

        # 5. Output head combining Attended representation + conditioning
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        horizon: torch.Tensor,
        delta_t: torch.Tensor,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting future state S_hat_{t+k}."""
        cond = self.conditioner(horizon, delta_t)

        if current_state.ndim == 2:
            x_seq = current_state.unsqueeze(1)
        elif current_state.ndim == 3:
            x_seq = current_state
        else:
            raise ValueError(f"Expected current_state 2D or 3D, got {current_state.ndim}")

        h_proj = self.input_proj(x_seq)
        gru_out, _ = self.gru(h_proj)
        h_att = self.attention(gru_out)

        combined = torch.cat([h_att, cond], dim=-1)
        delta_s = self.delta_head(combined)

        ref_state = current_state if current_state.ndim == 2 else current_state[:, -1, :]
        predicted_state = ref_state + delta_s

        if return_delta:
            return predicted_state, delta_s
        return predicted_state

