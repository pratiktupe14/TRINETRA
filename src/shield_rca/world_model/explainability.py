"""Predictive Explainability and Attribution for CyberHeal AI / TRINETRA.

Implements deterministic Forecast Driver Analysis and State Delta Attribution:
    explain_forecast(current_state, predicted_state, horizon, top_k) -> ForecastExplanation

Honest Explainable AI (XAI) Disclosure:
- Method: "State Delta Attribution" (Exact mathematical feature and group decomposition).
- NOT SHAP: This module does NOT compute Shapley values (SHAP). It computes deterministic
  first-order state-space trajectory deltas across the 280 Network State dimensions and 5
  architectural groups.
- Answers:
  1. What changed between the current state S_t and predicted state S_hat_{t+k}?
  2. Which specific features experienced the largest surges or drops?
  3. Which feature groups (Flags, Macro, Endpoints, Continuous, Shape) dominate the deviation?
  4. What is the signed direction of change (INCREASE / DECREASE / UNCHANGED)?
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..network_state.state_schema import (
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
)
from .loss import (
    SLICE_CONTINUOUS,
    SLICE_ENDPOINTS,
    SLICE_FLAGS,
    SLICE_MACRO,
    SLICE_SHAPE,
)

logger = logging.getLogger(__name__)

# Map group names to their respective slice boundaries in the 280-d state vector
GROUP_SLICES: dict[str, slice] = {
    "continuous": SLICE_CONTINUOUS,
    "flags": SLICE_FLAGS,
    "shape": SLICE_SHAPE,
    "endpoints": SLICE_ENDPOINTS,
    "macro": SLICE_MACRO,
}


def _get_feature_group(index: int) -> str:
    """Determine group name for a given feature index in [0, 279]."""
    if index < 150:
        return "continuous"
    elif index < 198:
        return "flags"
    elif index < 264:
        return "shape"
    elif index < 270:
        return "endpoints"
    elif index < 280:
        return "macro"
    raise IndexError(f"Feature index out of range: {index}")


@dataclass(frozen=True, slots=True)
class FeatureChange:
    """Detailed attribution of an individual state feature's predicted change."""

    feature_name: str
    group: str
    current_value: float
    predicted_value: float
    delta: float  # Signed difference: predicted - current
    magnitude: float  # Absolute difference: |delta|
    direction: str  # "INCREASE", "DECREASE", or "UNCHANGED"
    relative_share: float  # Fraction of total state drift in [0.0, 1.0]


@dataclass(frozen=True, slots=True)
class GroupAttribution:
    """Attribution aggregated across an architectural state group."""

    group_name: str
    absolute_delta_sum: float
    relative_share: float  # Fraction of total state drift in [0.0, 1.0]
    dominant_features: list[str]  # Top features within this specific group


@dataclass(frozen=True, slots=True)
class ForecastExplanation:
    """Comprehensive explainability report for a predicted Network State forecast."""

    explanation_method: str  # "State Delta Attribution"
    horizon: int
    total_drift_magnitude: float
    top_features: list[FeatureChange]
    group_attributions: list[GroupAttribution]
    summary_text: str

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to a JSON-serializable dictionary."""
        return asdict(self)


class ForecastExplainabilityEngine:
    """Explains World Model predictions by identifying top feature drivers and group contributions."""

    def __init__(self, default_top_k: int = 5) -> None:
        """Initialize the explainability engine.

        Args:
            default_top_k: Number of top feature drivers to extract by default.
        """
        if default_top_k < 1:
            raise ValueError(f"default_top_k must be >= 1, got {default_top_k}")
        self.default_top_k = default_top_k

    def explain(
        self,
        current_state: np.ndarray,
        predicted_state: np.ndarray,
        horizon: int = 1,
        top_k: int | None = None,
    ) -> ForecastExplanation:
        """Generate deterministic feature and group attribution for a predicted state.

        Args:
            current_state: 1D array of shape (280,) representing S_t.
            predicted_state: 1D array of shape (280,) representing S_hat_{t+k}.
            horizon: Discrete future horizon k >= 1.
            top_k: Number of top driving features to return (defaults to self.default_top_k).

        Returns:
            ForecastExplanation containing top feature drivers and group attribution breakdown.
        """
        k_val = top_k if top_k is not None else self.default_top_k
        if k_val < 1:
            raise ValueError(f"top_k must be >= 1, got {k_val}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        cur = np.asarray(current_state, dtype=np.float32)
        pred = np.asarray(predicted_state, dtype=np.float32)

        if cur.shape != (STATE_FEATURE_DIM,) or pred.shape != (STATE_FEATURE_DIM,):
            raise ValueError(
                f"State dimension mismatch: expected ({STATE_FEATURE_DIM},), "
                f"got cur={cur.shape}, pred={pred.shape}"
            )
        if not np.all(np.isfinite(cur)) or not np.all(np.isfinite(pred)):
            raise ValueError("Input state contains non-finite values (NaN or Inf)")

        # 1. Compute signed and absolute deltas
        deltas = pred - cur
        abs_deltas = np.abs(deltas)
        total_drift = float(np.sum(abs_deltas))
        eps = 1e-9
        denominator = total_drift if total_drift > 0.0 else eps

        # 2. Extract Top-K Features
        # Stable deterministic sorting: sort primarily by magnitude descending,
        # break ties by feature index ascending
        indices = np.arange(STATE_FEATURE_DIM)
        # Using lexsort: primary key is -abs_deltas, secondary key is indices
        sort_order = np.lexsort((indices, -abs_deltas))
        top_indices = sort_order[: min(k_val, STATE_FEATURE_DIM)]

        top_features: list[FeatureChange] = []
        for idx in top_indices:
            val_cur = float(cur[idx])
            val_pred = float(pred[idx])
            val_delta = float(deltas[idx])
            val_mag = float(abs_deltas[idx])

            if val_delta > 1e-6:
                direction = "INCREASE"
            elif val_delta < -1e-6:
                direction = "DECREASE"
            else:
                direction = "UNCHANGED"

            rel_share = float(val_mag / denominator) if total_drift > 0.0 else 0.0

            top_features.append(
                FeatureChange(
                    feature_name=STATE_FEATURE_NAMES[idx],
                    group=_get_feature_group(int(idx)),
                    current_value=val_cur,
                    predicted_value=val_pred,
                    delta=val_delta,
                    magnitude=val_mag,
                    direction=direction,
                    relative_share=rel_share,
                )
            )

        # 3. Group Attribution Breakdown
        group_attributions: list[GroupAttribution] = []
        for g_name, sl in GROUP_SLICES.items():
            g_abs = abs_deltas[sl]
            g_sum = float(np.sum(g_abs))
            g_share = float(g_sum / denominator) if total_drift > 0.0 else 0.0

            # Find top 3 dominant features inside this group
            g_indices = np.arange(sl.start, sl.stop)
            g_sort = np.lexsort((g_indices, -abs_deltas[sl]))
            dominant_feats = [STATE_FEATURE_NAMES[g_indices[i]] for i in g_sort[:3]]

            group_attributions.append(
                GroupAttribution(
                    group_name=g_name,
                    absolute_delta_sum=g_sum,
                    relative_share=g_share,
                    dominant_features=dominant_feats,
                )
            )

        # Sort group attributions descending by absolute delta sum (with name tie-breaker)
        group_attributions.sort(key=lambda g: (-g.absolute_delta_sum, g.group_name))

        # 4. Generate Human-Readable Summary
        if total_drift < 1e-5:
            summary = f"Horizon k={horizon}: Forecast indicates a quiescent, steady-state network with negligible drift."
        else:
            top_g = group_attributions[0]
            top_f = top_features[0]
            summary = (
                f"Horizon k={horizon}: Predicted trajectory deviation is dominated by '{top_g.group_name}' "
                f"({top_g.relative_share * 100:.1f}% of total drift). Primary feature driver is "
                f"'{top_f.feature_name}' ({top_f.direction} of {top_f.magnitude:.4f})."
            )

        return ForecastExplanation(
            explanation_method="State Delta Attribution",
            horizon=horizon,
            total_drift_magnitude=total_drift,
            top_features=top_features,
            group_attributions=group_attributions,
            summary_text=summary,
        )


def explain_forecast(
    current_state: np.ndarray,
    predicted_state: np.ndarray,
    horizon: int = 1,
    top_k: int | None = None,
) -> ForecastExplanation:
    """Convenience function to generate ForecastExplanation."""
    engine = ForecastExplainabilityEngine(default_top_k=top_k or 5)
    return engine.explain(current_state, predicted_state, horizon=horizon, top_k=top_k)
