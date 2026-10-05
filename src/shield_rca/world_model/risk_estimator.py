"""Predictive Risk and Infiltration Estimator for CyberHeal AI / TRINETRA.

Translates future Network State forecasts into mathematically bounded,
deterministic risk indices and infiltration scores:
    estimate_risk(current_state, predicted_state, horizon, delta_t) -> HorizonRiskEstimate
    estimate_trajectory_risk(current_state, predicted_trajectory) -> RiskTrajectory

Mathematical Properties:
- Normalized risk score: risk_score in [0.0, 1.0].
- Infiltration score: infiltration_score in [0.0, 1.0] (uncalibrated proxy for infiltration dynamics).
- Zero state change yields risk_score = 0.0 and infiltration_score = 0.0.
- Larger anomalous deviations yield monotonically non-decreasing risk up to 1.0.
- Rejects non-finite values (NaN, Inf) and dimension mismatches strictly.
- Strict determinism: identical inputs produce identical outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

# Configurable group importance weights for aggregate risk computation.
# Reflects security domain importance: flags, endpoints, and macro volume indicators
# carry higher diagnostic weight than general background distribution shape.
DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "continuous": 0.20,
    "flags": 0.30,
    "shape": 0.10,
    "endpoints": 0.20,
    "macro": 0.20,
}

# Group slices mapped by group name for fast vectorised indexing
GROUP_SLICES: dict[str, slice] = {
    "continuous": SLICE_CONTINUOUS,
    "flags": SLICE_FLAGS,
    "shape": SLICE_SHAPE,
    "endpoints": SLICE_ENDPOINTS,
    "macro": SLICE_MACRO,
}


@dataclass(frozen=True, slots=True)
class StateChangeSummary:
    """Summary of state vector deviations between current and predicted state."""

    total_l1_drift: float
    total_l2_drift: float
    group_drifts: dict[str, float]
    max_drift_feature: str
    max_drift_value: float


@dataclass(frozen=True, slots=True)
class HorizonRiskEstimate:
    """Predictive risk assessment for a specific observation horizon k."""

    horizon: int
    delta_t: float
    risk_score: float  # Bounded [0.0, 1.0]
    infiltration_score: float  # Bounded [0.0, 1.0] (uncalibrated heuristic score)
    confidence: float | None  # Documented confidence proxy or None if uncomputed
    state_change: StateChangeSummary
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskTrajectory:
    """Comprehensive multi-horizon predictive risk trajectory."""

    aggregate_risk: float  # Overall maximum or trajectory-weighted risk in [0.0, 1.0]
    aggregate_infiltration: float  # Trajectory-weighted infiltration score in [0.0, 1.0]
    horizon_estimates: dict[int, HorizonRiskEstimate]
    peak_horizon: int
    is_escalating: bool


class PredictiveRiskEstimator:
    """Computes bounded predictive risk and infiltration scores from Network State forecasts."""

    def __init__(
        self,
        group_weights: dict[str, float] | None = None,
        risk_sensitivity: float = 1.5,
        infiltration_sensitivity: float = 2.0,
    ) -> None:
        """Initialize the risk estimator with configurable sensitivity parameters.

        Args:
            group_weights: Relative weights for the 5 state groups (must sum to ~1.0).
            risk_sensitivity: Scaling factor lambda in the smooth saturation curve:
                risk = 1.0 - exp(-lambda * weighted_drift). Higher = more sensitive.
            infiltration_sensitivity: Scaling factor for infiltration-specific metrics.
        """
        weights = group_weights if group_weights is not None else DEFAULT_GROUP_WEIGHTS
        total_w = sum(weights.values())
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"Group weights must sum to 1.0, got {total_w}")

        self.group_weights = dict(weights)
        self.risk_sensitivity = float(risk_sensitivity)
        self.infiltration_sensitivity = float(infiltration_sensitivity)

    def _validate_state(self, state: np.ndarray, name: str) -> np.ndarray:
        """Validate state vector dimensionality and finiteness."""
        if state is None:
            raise ValueError(f"{name} cannot be None")

        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D vector, got ndim={arr.ndim} with shape {arr.shape}")
        if arr.shape[0] != STATE_FEATURE_DIM:
            raise ValueError(
                f"{name} dimension mismatch: expected {STATE_FEATURE_DIM}, got {arr.shape[0]}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

        return arr

    def compute_state_change(
        self,
        current_state: np.ndarray,
        predicted_state: np.ndarray,
    ) -> tuple[np.ndarray, StateChangeSummary]:
        """Compute absolute and group-level differences between states."""
        cur = self._validate_state(current_state, "current_state")
        pred = self._validate_state(predicted_state, "predicted_state")

        diff = pred - cur
        abs_diff = np.abs(diff)

        total_l1 = float(np.sum(abs_diff))
        total_l2 = float(np.linalg.norm(diff))

        # Per-group mean absolute deviation
        group_drifts: dict[str, float] = {}
        for g_name, sl in GROUP_SLICES.items():
            g_slice_diff = abs_diff[sl]
            group_drifts[g_name] = float(np.mean(g_slice_diff)) if len(g_slice_diff) > 0 else 0.0

        # Maximum drifting individual feature
        max_idx = int(np.argmax(abs_diff))
        max_feature_name = STATE_FEATURE_NAMES[max_idx]
        max_val = float(abs_diff[max_idx])

        summary = StateChangeSummary(
            total_l1_drift=total_l1,
            total_l2_drift=total_l2,
            group_drifts=group_drifts,
            max_drift_feature=max_feature_name,
            max_drift_value=max_val,
        )

        return diff, summary

    def estimate_horizon_risk(
        self,
        current_state: np.ndarray,
        predicted_state: np.ndarray,
        horizon: int = 1,
        delta_t: float = 1.0,
        model_confidence: float | None = None,
    ) -> HorizonRiskEstimate:
        """Estimate bounded risk for a single future prediction horizon.

        Args:
            current_state: Current 280-d Network State S_t.
            predicted_state: Model-predicted future 280-d Network State S_hat_{t+k}.
            horizon: Discrete future horizon k >= 1.
            delta_t: Continuous elapsed wall-clock seconds >= 0.0.
            model_confidence: Optional external model evaluation confidence in [0.0, 1.0].
                If None, confidence is recorded as None (no fabricated confidence).

        Returns:
            HorizonRiskEstimate containing bounded risk_score, infiltration_score,
            and state change summary.
        """
        if horizon < 1:
            raise ValueError(f"Horizon k must be >= 1, got {horizon}")
        if delta_t < 0.0:
            raise ValueError(f"Elapsed time delta_t must be >= 0.0, got {delta_t}")
        if model_confidence is not None and not (0.0 <= model_confidence <= 1.0):
            raise ValueError(f"model_confidence must be in [0.0, 1.0], got {model_confidence}")

        diff, summary = self.compute_state_change(current_state, predicted_state)

        # 1. Bounded Risk Score Formulation:
        # Weighted mean absolute deviation across groups:
        weighted_drift = sum(
            self.group_weights[g] * summary.group_drifts[g] for g in self.group_weights
        )

        # Smooth asymptotic mapping: risk_score = 1.0 - exp(-lambda * weighted_drift)
        # Guarantees:
        # - drift == 0.0 -> risk_score == 0.0
        # - drift > 0.0 -> monotonically increasing, bounded strictly in [0.0, 1.0]
        # - Never outputs negative numbers or exceeds 1.0
        risk_score = float(1.0 - np.exp(-self.risk_sensitivity * weighted_drift))
        risk_score = min(max(risk_score, 0.0), 1.0)

        # 2. Infiltration Score Formulation (NOT calibrated probability!):
        # Infiltration attacks (lateral movement, payload delivery, slow data exfiltration)
        # disproportionately distort endpoint diversity, macro byte asymmetry, and TCP reset rates.
        # We extract specific infiltration-sensitive group signals:
        endpoint_drift = summary.group_drifts["endpoints"]
        macro_drift = summary.group_drifts["macro"]
        flags_drift = summary.group_drifts["flags"]

        # Composite infiltration indicator
        infil_indicator = (0.45 * endpoint_drift) + (0.35 * macro_drift) + (0.20 * flags_drift)
        infiltration_score = float(1.0 - np.exp(-self.infiltration_sensitivity * infil_indicator))
        infiltration_score = min(max(infiltration_score, 0.0), 1.0)

        return HorizonRiskEstimate(
            horizon=horizon,
            delta_t=delta_t,
            risk_score=risk_score,
            infiltration_score=infiltration_score,
            confidence=model_confidence,
            state_change=summary,
            details={
                "weighted_drift": float(weighted_drift),
                "infil_indicator": float(infil_indicator),
                "is_zero_drift": bool(summary.total_l1_drift == 0.0),
            },
        )

    def estimate_trajectory_risk(
        self,
        current_state: np.ndarray,
        predicted_states: dict[int, np.ndarray],
        delta_times: dict[int, float] | None = None,
        model_confidences: dict[int, float] | None = None,
    ) -> RiskTrajectory:
        """Estimate composite risk across multi-horizon trajectory (e.g. k=1, 3, 5).

        Args:
            current_state: Current 280-d Network State S_t.
            predicted_states: Dictionary mapping horizon k -> predicted state S_hat_{t+k}.
            delta_times: Optional mapping of horizon k -> elapsed time delta_t.
            model_confidences: Optional mapping of horizon k -> confidence.

        Returns:
            RiskTrajectory aggregating all horizon estimates.
        """
        if not predicted_states:
            raise ValueError("predicted_states cannot be empty")

        horizons = sorted(predicted_states.keys())
        estimates: dict[int, HorizonRiskEstimate] = {}

        for k in horizons:
            dt = delta_times.get(k, float(k)) if delta_times is not None else float(k)
            conf = model_confidences.get(k) if model_confidences is not None else None
            est = self.estimate_horizon_risk(
                current_state=current_state,
                predicted_state=predicted_states[k],
                horizon=k,
                delta_t=dt,
                model_confidence=conf,
            )
            estimates[k] = est

        # Aggregate risk is taken as the maximum observed risk across the trajectory,
        # with escalation detection if risk strictly increases with horizon k.
        risk_values = [estimates[k].risk_score for k in horizons]
        infil_values = [estimates[k].infiltration_score for k in horizons]

        peak_idx = int(np.argmax(risk_values))
        peak_horizon = horizons[peak_idx]
        aggregate_risk = max(risk_values)
        aggregate_infil = max(infil_values)

        # Escalation check: does risk increase as horizon advances?
        is_escalating = False
        if len(risk_values) >= 2:
            is_escalating = bool(risk_values[-1] > risk_values[0] and aggregate_risk > 0.05)

        return RiskTrajectory(
            aggregate_risk=aggregate_risk,
            aggregate_infiltration=aggregate_infil,
            horizon_estimates=estimates,
            peak_horizon=peak_horizon,
            is_escalating=is_escalating,
        )
