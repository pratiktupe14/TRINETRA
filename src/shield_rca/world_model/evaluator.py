"""Evaluation Framework for the CyberHeal World Model.

Computes separate performance metrics for each horizon k in {1, 3, 5}:
- World Model vs Persistence Baseline vs Linear Dynamics Baseline
- Relative percentage improvements
- Detailed error breakdown across the 5 Network State groups
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..network_state.future_dataset import FutureStatePairBatch, MultiHorizonFutureDataset
from .baselines import LinearDynamicsBaseline, PersistenceBaseline
from .loss import (
    SLICE_CONTINUOUS,
    SLICE_ENDPOINTS,
    SLICE_FLAGS,
    SLICE_MACRO,
    SLICE_SHAPE,
    WorldModelLoss,
)
from .model import StateResidualDynamicsModel


class WorldModelEvaluator:
    """Evaluates World Model predictions across horizons and state feature groups."""

    def __init__(
        self,
        model: StateResidualDynamicsModel,
        device: str = "cpu",
        delta_huber: float = 1.0,
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.criterion = WorldModelLoss(delta=delta_huber)
        self.persistence = PersistenceBaseline()

    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: FutureStatePairBatch,
        linear_baseline: LinearDynamicsBaseline | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single horizon pair batch."""
        if batch.num_pairs == 0:
            return {"num_pairs": 0}

        # 1. World Model predictions
        X = torch.from_numpy(batch.current_states).float().to(self.device)
        k_tensor = torch.full((batch.num_pairs,), batch.horizon, dtype=torch.long, device=self.device)
        dt_tensor = torch.from_numpy(batch.delta_times).float().to(self.device)
        Y = torch.from_numpy(batch.target_states).float().to(self.device)

        pred = self.model(X, k_tensor, dt_tensor)
        wm_metrics = self.criterion.compute_metrics(pred, Y)

        # 2. Persistence baseline metrics
        pers_metrics = self.persistence.evaluate(batch)

        # 3. Linear baseline metrics (if provided)
        linear_metrics = linear_baseline.evaluate(batch) if linear_baseline is not None else None

        # 4. Relative improvement: (1 - MAE_model / MAE_baseline) * 100%
        eps = 1e-7
        imp_over_pers = (1.0 - (wm_metrics["mae"] / (pers_metrics["mae"] + eps))) * 100.0

        res: dict[str, Any] = {
            "horizon": batch.horizon,
            "num_pairs": batch.num_pairs,
            "world_model": wm_metrics,
            "persistence_baseline": pers_metrics,
            "improvement_over_persistence_pct": imp_over_pers,
            "group_errors": {
                "continuous_mae": wm_metrics["mae_continuous"],
                "flags_mae": wm_metrics["mae_flags"],
                "shape_mae": wm_metrics["mae_shape"],
                "endpoints_mae": wm_metrics["mae_endpoints"],
                "macro_mae": wm_metrics["mae_macro"],
            },
        }

        if linear_metrics is not None:
            res["linear_baseline"] = linear_metrics
            imp_over_linear = (1.0 - (wm_metrics["mae"] / (linear_metrics["mae"] + eps))) * 100.0
            res["improvement_over_linear_pct"] = imp_over_linear

        return res

    def evaluate_multi_horizon(
        self,
        dataset: MultiHorizonFutureDataset,
        linear_baselines: dict[int, LinearDynamicsBaseline] | None = None,
    ) -> dict[str, Any]:
        """Evaluate performance separately for each horizon k in the dataset."""
        report: dict[str, Any] = {
            "total_usable_pairs": dataset.total_usable_pairs,
            "horizons": dataset.horizons,
            "results_by_horizon": {},
        }

        for k in dataset.horizons:
            batch = dataset.batches[k]
            lin = linear_baselines.get(k) if linear_baselines else None
            report["results_by_horizon"][f"k_{k}"] = self.evaluate_batch(batch, linear_baseline=lin)

        return report
