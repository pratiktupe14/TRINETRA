"""Predictive Trigger Gate for CyberHeal AI / TRINETRA.

Serves as the safety-first decision bridge between the Predictive Intelligence
Layer (World Model forecast, risk estimator, stage mapper, explainability) and
downstream response orchestration:

    Network State S_t
            ↓
    World Model Forecast S_hat_{t+k}
            ↓
    Risk Estimator & Stage Mapper & Explainability
            ↓
    PREDICTIVE TRIGGER GATE
            ↓
    ┌─────────────────────────────┐
    │ High confidence + safe      │
    │       → AUTO_TRIGGER        │
    └─────────────────────────────┘
                  OR
    ┌─────────────────────────────┐
    │ Uncertain / high impact     │
    │       → HUMAN_REVIEW        │
    └─────────────────────────────┘
                  OR
    ┌─────────────────────────────┐
    │ Low risk / quiescent        │
    │       → NO_TRIGGER          │
    └─────────────────────────────┘

Guiding Safety Principles:
1. Conservative by Design: When in doubt, escalate to HUMAN_REVIEW or suppress (NO_TRIGGER).
2. Multi-Dimensional Decision: Evaluates Risk, Confidence, Evidence, Asset Criticality,
   and Action Safety. Never triggers based on risk score alone.
3. Anti-Fabrication & Honesty: Risk and confidence are explicit policy signals, not
   unverified calibrated probabilities. UNKNOWN stages and UNKNOWN action safety always
   prevent automated triggering.
4. Offline & Deterministic: Zero external API/LLM calls; 100% deterministic local policy.
5. Explainable: Every decision emits explicit positive reasons and blocking reasons.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .explainability import ForecastExplanation
from .risk_estimator import HorizonRiskEstimate, RiskTrajectory
from .stage_mapper import AttackStage, StageMappingResult

logger = logging.getLogger(__name__)


class TriggerDecisionType(str, Enum):
    """Possible outcomes produced by the Predictive Trigger Gate."""

    NO_TRIGGER = "NO_TRIGGER"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    AUTO_TRIGGER = "AUTO_TRIGGER"


class AssetCriticality(str, Enum):
    """Categorical operational criticality of a target network asset."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Numeric ordering for inequality comparisons: LOW < MEDIUM < HIGH < CRITICAL."""
        _ranks = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return _ranks[self.value]

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, AssetCriticality):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, AssetCriticality):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, AssetCriticality):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, AssetCriticality):
            return NotImplemented
        return self.rank >= other.rank


class ActionSafety(str, Enum):
    """Safety classification of proposed downstream remediation action."""

    SAFE = "SAFE"
    RESTRICTED = "RESTRICTED"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TriggerPolicy:
    """Configurable safety thresholds and gating constraints for automated response."""

    risk_auto_threshold: float = 0.75
    risk_review_threshold: float = 0.35
    confidence_auto_threshold: float = 0.75
    confidence_review_threshold: float = 0.40
    evidence_auto_threshold: float = 0.60
    maximum_auto_asset_criticality: AssetCriticality = AssetCriticality.MEDIUM
    require_safe_action: bool = True
    allowed_auto_stages: frozenset[AttackStage] = frozenset({
        AttackStage.RECONNAISSANCE,
        AttackStage.INITIAL_ACCESS,
        AttackStage.INFILTRATION,
        AttackStage.IMPACT,
    })
    blocked_auto_stages: frozenset[AttackStage] = frozenset({
        AttackStage.NORMAL,
        AttackStage.UNKNOWN,
    })
    require_trajectory_escalation_for_borderline: bool = False
    policy_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate policy parameters upon construction."""
        for name in (
            "risk_auto_threshold",
            "risk_review_threshold",
            "confidence_auto_threshold",
            "confidence_review_threshold",
            "evidence_auto_threshold",
        ):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Policy threshold {name} must be a finite float, got {val}")
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Policy threshold {name} must be in [0.0, 1.0], got {val}")

        if self.risk_review_threshold > self.risk_auto_threshold:
            raise ValueError(
                f"risk_review_threshold ({self.risk_review_threshold}) cannot exceed "
                f"risk_auto_threshold ({self.risk_auto_threshold})"
            )
        if self.confidence_review_threshold > self.confidence_auto_threshold:
            raise ValueError(
                f"confidence_review_threshold ({self.confidence_review_threshold}) cannot exceed "
                f"confidence_auto_threshold ({self.confidence_auto_threshold})"
            )


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Immutable, explainable decision record from the Predictive Trigger Gate."""

    decision: TriggerDecisionType
    risk_score: float
    confidence: float | None
    evidence_strength: float
    asset_criticality: AssetCriticality
    action_safety: ActionSafety
    predicted_stage: AttackStage
    reasons: list[str]
    blocking_reasons: list[str]
    policy_version: str
    risk_trajectory_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to a pure JSON-serializable dictionary."""
        return {
            "decision": self.decision.value,
            "risk_score": float(self.risk_score),
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "evidence_strength": float(self.evidence_strength),
            "asset_criticality": self.asset_criticality.value,
            "action_safety": self.action_safety.value,
            "predicted_stage": self.predicted_stage.value,
            "reasons": list(self.reasons),
            "blocking_reasons": list(self.blocking_reasons),
            "policy_version": self.policy_version,
            "risk_trajectory_summary": self.risk_trajectory_summary,
            "metadata": dict(self.metadata),
        }


class PredictiveTriggerGate:
    """Evaluates predictive intelligence against safety policies to govern response escalation."""

    def __init__(self, policy: TriggerPolicy | None = None) -> None:
        """Initialize the trigger gate with a specific or default safety policy."""
        self.policy = policy if policy is not None else TriggerPolicy()

    def evaluate(
        self,
        risk_score: float | HorizonRiskEstimate | RiskTrajectory,
        confidence: float | None = None,
        evidence_strength: float | None = None,
        asset_criticality: AssetCriticality | str = AssetCriticality.MEDIUM,
        action_safety: ActionSafety | str = ActionSafety.UNKNOWN,
        predicted_stage: AttackStage | StageMappingResult | str = AttackStage.UNKNOWN,
        explanation: ForecastExplanation | None = None,
        risk_trajectory: RiskTrajectory | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TriggerDecision:
        """Evaluate multi-dimensional predictive threat signals and return a TriggerDecision.

        Args:
            risk_score: Predictive risk score in [0.0, 1.0] or HorizonRiskEstimate / RiskTrajectory.
            confidence: Estimated model/forecast confidence in [0.0, 1.0] or None.
            evidence_strength: Strength of supporting telemetry evidence in [0.0, 1.0] or None.
            asset_criticality: Operational criticality of the target asset (LOW, MEDIUM, HIGH, CRITICAL).
            action_safety: Safety classification of proposed remediation action (SAFE, RESTRICTED, UNSAFE, UNKNOWN).
            predicted_stage: Predicted attack stage or StageMappingResult.
            explanation: Optional Phase 1 ForecastExplanation to extract evidence if not supplied.
            risk_trajectory: Optional Phase 1 multi-horizon RiskTrajectory.
            metadata: Optional dictionary with incident/context identifiers (not mutated).

        Returns:
            TriggerDecision containing the gating outcome, scores, reasons, and blocking reasons.
        """
        # 1. Parse and extract risk_score and trajectory
        raw_risk: float
        if isinstance(risk_score, RiskTrajectory):
            if risk_trajectory is None:
                risk_trajectory = risk_score
            raw_risk = risk_score.aggregate_risk
            if confidence is None:
                peak_est = risk_score.horizon_estimates.get(risk_score.peak_horizon)
                if peak_est is not None and peak_est.confidence is not None:
                    confidence = peak_est.confidence
        elif isinstance(risk_score, HorizonRiskEstimate):
            raw_risk = risk_score.risk_score
            if confidence is None and risk_score.confidence is not None:
                confidence = risk_score.confidence
        elif isinstance(risk_score, (int, float)):
            raw_risk = float(risk_score)
        else:
            raise ValueError(f"Unsupported risk_score type: {type(risk_score)}")

        if math.isnan(raw_risk) or math.isinf(raw_risk):
            raise ValueError(f"risk_score must be a finite float, got {raw_risk}")
        if not 0.0 <= raw_risk <= 1.0:
            raise ValueError(f"risk_score must be in [0.0, 1.0], got {raw_risk}")

        # 2. Parse and validate confidence
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
                raise ValueError(f"confidence must be a finite float or None, got {confidence}")
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")

        # 3. Parse and validate predicted_stage
        stage_enum: AttackStage
        if isinstance(predicted_stage, StageMappingResult):
            stage_enum = predicted_stage.stage
            if confidence is None and predicted_stage.confidence is not None:
                confidence = predicted_stage.confidence
        elif isinstance(predicted_stage, str):
            try:
                stage_enum = AttackStage(predicted_stage)
            except ValueError:
                raise ValueError(f"Invalid predicted_stage string: {predicted_stage}")
        elif isinstance(predicted_stage, AttackStage):
            stage_enum = predicted_stage
        else:
            raise ValueError(f"Unsupported predicted_stage type: {type(predicted_stage)}")

        # 4. Parse and validate asset_criticality
        asset_enum: AssetCriticality
        if isinstance(asset_criticality, str):
            try:
                asset_enum = AssetCriticality(asset_criticality)
            except ValueError:
                raise ValueError(f"Invalid asset_criticality string: {asset_criticality}")
        elif isinstance(asset_criticality, AssetCriticality):
            asset_enum = asset_criticality
        else:
            raise ValueError(f"Unsupported asset_criticality type: {type(asset_criticality)}")

        # 5. Parse and validate action_safety
        action_enum: ActionSafety
        if isinstance(action_safety, str):
            try:
                action_enum = ActionSafety(action_safety)
            except ValueError:
                raise ValueError(f"Invalid action_safety string: {action_safety}")
        elif isinstance(action_safety, ActionSafety):
            action_enum = action_safety
        else:
            raise ValueError(f"Unsupported action_safety type: {type(action_safety)}")

        # 6. Parse or derive evidence_strength
        ev_strength: float
        if evidence_strength is not None:
            if not isinstance(evidence_strength, (int, float)) or math.isnan(evidence_strength) or math.isinf(evidence_strength):
                raise ValueError(f"evidence_strength must be a finite float, got {evidence_strength}")
            ev_strength = float(evidence_strength)
            if not 0.0 <= ev_strength <= 1.0:
                raise ValueError(f"evidence_strength must be in [0.0, 1.0], got {evidence_strength}")
        else:
            if explanation is not None:
                ev_strength = self._derive_evidence_from_explanation(explanation)
            elif risk_trajectory is not None:
                ev_strength = self._derive_evidence_from_trajectory(risk_trajectory)
            elif isinstance(risk_score, HorizonRiskEstimate):
                ev_strength = self._derive_evidence_from_estimate(risk_score)
            else:
                ev_strength = 0.0

        # 7. Extract trajectory summary if available
        traj_summary: dict[str, Any] | None = None
        if risk_trajectory is not None:
            traj_summary = {
                "aggregate_risk": float(risk_trajectory.aggregate_risk),
                "aggregate_infiltration": float(risk_trajectory.aggregate_infiltration),
                "peak_horizon": int(risk_trajectory.peak_horizon),
                "is_escalating": bool(risk_trajectory.is_escalating),
                "horizons": sorted(list(risk_trajectory.horizon_estimates.keys())),
            }

        # 8. Evaluate policy criteria for AUTO_TRIGGER
        is_risk_auto = raw_risk >= self.policy.risk_auto_threshold
        is_conf_auto = (confidence is not None) and (confidence >= self.policy.confidence_auto_threshold)
        is_ev_auto = ev_strength >= self.policy.evidence_auto_threshold
        is_action_auto = (action_enum == ActionSafety.SAFE) if self.policy.require_safe_action else (action_enum != ActionSafety.UNSAFE)
        is_asset_auto = asset_enum <= self.policy.maximum_auto_asset_criticality
        is_stage_auto = (stage_enum in self.policy.allowed_auto_stages) and (stage_enum not in self.policy.blocked_auto_stages)

        reasons: list[str] = []
        blocking_reasons: list[str] = []

        # Trajectory escalation check if enabled
        trajectory_blocked = False
        if self.policy.require_trajectory_escalation_for_borderline and risk_trajectory is not None:
            if raw_risk < (self.policy.risk_auto_threshold + 0.10) and not risk_trajectory.is_escalating:
                trajectory_blocked = True
                blocking_reasons.append("Multi-horizon risk trajectory is not escalating for borderline risk")

        if not is_risk_auto:
            blocking_reasons.append(
                f"Risk score ({raw_risk:.3f}) below auto threshold ({self.policy.risk_auto_threshold:.3f})"
            )
        if not is_conf_auto:
            if confidence is None:
                blocking_reasons.append("Prediction confidence is unavailable (None)")
            else:
                blocking_reasons.append(
                    f"Confidence ({confidence:.3f}) below auto threshold ({self.policy.confidence_auto_threshold:.3f})"
                )
        if not is_ev_auto:
            blocking_reasons.append(
                f"Evidence strength ({ev_strength:.3f}) below auto threshold ({self.policy.evidence_auto_threshold:.3f})"
            )
        if not is_action_auto:
            if action_enum == ActionSafety.UNKNOWN:
                blocking_reasons.append("Action safety is UNKNOWN (conservative policy requires SAFE for automated action)")
            elif action_enum == ActionSafety.RESTRICTED:
                blocking_reasons.append("Action safety is RESTRICTED (policy requires human operator approval)")
            elif action_enum == ActionSafety.UNSAFE:
                blocking_reasons.append("Action safety is UNSAFE (automated execution strictly prohibited)")
        if not is_asset_auto:
            blocking_reasons.append(
                f"Asset criticality ({asset_enum.value}) exceeds auto threshold ({self.policy.maximum_auto_asset_criticality.value}); requires human sign-off"
            )
        if not is_stage_auto:
            if stage_enum == AttackStage.UNKNOWN:
                blocking_reasons.append("Predicted attack stage is UNKNOWN (ambiguous signature requires human review)")
            elif stage_enum == AttackStage.NORMAL:
                blocking_reasons.append("Predicted attack stage is NORMAL (no automated remediation permitted)")
            else:
                blocking_reasons.append(f"Predicted attack stage ({stage_enum.value}) is not in allowed auto stages")

        # 9. Determine final decision
        decision_type: TriggerDecisionType
        if is_risk_auto and is_conf_auto and is_ev_auto and is_action_auto and is_asset_auto and is_stage_auto and not trajectory_blocked:
            decision_type = TriggerDecisionType.AUTO_TRIGGER
            reasons.append(f"Risk score ({raw_risk:.3f}) meets or exceeds auto threshold ({self.policy.risk_auto_threshold:.3f})")
            reasons.append(f"Confidence ({confidence:.3f}) meets or exceeds auto threshold ({self.policy.confidence_auto_threshold:.3f})")
            reasons.append(f"Evidence strength ({ev_strength:.3f}) meets or exceeds auto threshold ({self.policy.evidence_auto_threshold:.3f})")
            reasons.append(f"Action safety ({action_enum.value}) is approved for automated remediation")
            reasons.append(f"Asset criticality ({asset_enum.value}) permits automated action (<= {self.policy.maximum_auto_asset_criticality.value})")
            reasons.append(f"Predicted stage ({stage_enum.value}) is approved for automated triggering")
            if risk_trajectory is not None and risk_trajectory.is_escalating:
                reasons.append("Multi-horizon trajectory confirms escalating threat trend")
            blocking_reasons.clear()
        else:
            if raw_risk < self.policy.risk_review_threshold:
                decision_type = TriggerDecisionType.NO_TRIGGER
                reasons.append(
                    f"Risk score ({raw_risk:.3f}) is below review threshold ({self.policy.risk_review_threshold:.3f}); response suppressed"
                )
                if stage_enum == AttackStage.NORMAL:
                    reasons.append("Predicted stage is NORMAL; quiescent baseline network state")
            elif stage_enum == AttackStage.NORMAL and raw_risk < self.policy.risk_auto_threshold:
                decision_type = TriggerDecisionType.NO_TRIGGER
                reasons.append(
                    f"Predicted stage is NORMAL and risk score ({raw_risk:.3f}) does not reach emergency auto threshold ({self.policy.risk_auto_threshold:.3f}); response suppressed"
                )
            else:
                decision_type = TriggerDecisionType.HUMAN_REVIEW
                reasons.append(
                    f"Predictive risk score ({raw_risk:.3f}) meets or exceeds review threshold ({self.policy.risk_review_threshold:.3f})"
                )
                reasons.append(
                    f"Automated triggering withheld due to {len(blocking_reasons)} safety gate check(s); escalating to human review"
                )
                if stage_enum != AttackStage.UNKNOWN and stage_enum != AttackStage.NORMAL:
                    reasons.append(f"Predicted threat stage is {stage_enum.value}")

        return TriggerDecision(
            decision=decision_type,
            risk_score=raw_risk,
            confidence=confidence,
            evidence_strength=ev_strength,
            asset_criticality=asset_enum,
            action_safety=action_enum,
            predicted_stage=stage_enum,
            reasons=reasons,
            blocking_reasons=blocking_reasons,
            policy_version=self.policy.policy_version,
            risk_trajectory_summary=traj_summary,
            metadata=dict(metadata) if metadata is not None else {},
        )

    @staticmethod
    def _derive_evidence_from_explanation(explanation: ForecastExplanation) -> float:
        """Deterministic mathematical evidence score from ForecastExplanation in [0.0, 1.0]."""
        # Magnitude saturation
        drift_saturation = 1.0 - math.exp(-1.5 * explanation.total_drift_magnitude)
        # Top-3 feature concentration
        top_share = sum(f.relative_share for f in explanation.top_features[:3]) if explanation.top_features else 0.0
        # Active groups count
        active_groups = sum(1 for g in explanation.group_attributions if g.absolute_delta_sum > 1e-4)
        group_factor = min(1.0, active_groups / 3.0)

        score = 0.50 * drift_saturation + 0.30 * min(1.0, top_share) + 0.20 * group_factor
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _derive_evidence_from_trajectory(trajectory: RiskTrajectory) -> float:
        """Deterministic evidence score from RiskTrajectory in [0.0, 1.0]."""
        score = 0.60 * trajectory.aggregate_risk + 0.40 * (0.80 if trajectory.is_escalating else 0.40)
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _derive_evidence_from_estimate(estimate: HorizonRiskEstimate) -> float:
        """Deterministic evidence score from HorizonRiskEstimate in [0.0, 1.0]."""
        drift_saturation = 1.0 - math.exp(-0.5 * estimate.state_change.total_l1_drift)
        return max(0.0, min(1.0, float(0.70 * drift_saturation + 0.30 * estimate.risk_score)))
