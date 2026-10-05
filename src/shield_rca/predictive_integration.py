"""TRINETRA Predictive Intelligence → CyberHeal Response Integration.

Provides the secure, deterministic, and backward-compatible integration adapter
connecting TRINETRA's Predictive Trigger Gate (Phase 2) to CyberHeal's multi-agent
remediation workflow (AgentState, Sentinel, Causor, Guardian, Planner, Executor):

    TRINETRA TriggerDecision
               ↓
    PredictiveIncidentAdapter
               ↓
    ┌───────────────────────────────────────────────┐
    │ AUTO_TRIGGER                                  │
    │   → Validated AgentState (incident_type=PRED) │
    │   → Enters CyberHeal Workflow                 │
    │   → Guardian retains full veto power          │
    ├───────────────────────────────────────────────┤
    │ HUMAN_REVIEW                                  │
    │   → HumanReviewPayload created                │
    │   → Remediation NOT initiated                 │
    │   → Escalated to SOC operator review          │
    ├───────────────────────────────────────────────┤
    │ NO_TRIGGER                                    │
    │   → Suppressed (No incident / No execution)   │
    └───────────────────────────────────────────────┘

Non-Negotiable Architecture Rules:
1. TRINETRA NEVER directly executes remediation commands (no iptables/netsh).
2. CyberHeal's Guardian is NEVER bypassed; AUTO_TRIGGER merely certifies the event
   as eligible for CyberHeal intake.
3. Original predictive evidence (risk, confidence, stage, features, trajectory, reasons)
   is fully preserved in AgentState and audit payloads.
4. Malformed inputs fail safely to REJECTED_INVALID.
5. 100% offline, deterministic, and backward-compatible.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .langgraph_core import AgentState
from .world_model.explainability import ForecastExplanation
from .world_model.trigger_gate import (
    ActionSafety,
    AssetCriticality,
    TriggerDecision,
    TriggerDecisionType,
)

logger = logging.getLogger(__name__)


class IncidentType(str, Enum):
    """Categorical classification of incident origins in CyberHeal."""

    REACTIVE = "REACTIVE"
    PREDICTIVE = "PREDICTIVE"


class PredictiveHandoffStatus(str, Enum):
    """Status outcomes from the predictive integration boundary."""

    AUTO_TRIGGERED = "AUTO_TRIGGERED"
    ESCALATED_HUMAN_REVIEW = "ESCALATED_HUMAN_REVIEW"
    SUPPRESSED = "SUPPRESSED"
    REJECTED_INVALID = "REJECTED_INVALID"


@dataclass(frozen=True, slots=True)
class HumanReviewPayload:
    """Audit and escalation bundle for predictive events requiring human authorization."""

    incident_id: str
    source: str
    trigger_decision: str  # "HUMAN_REVIEW"
    risk_score: float
    confidence: float | None
    evidence_strength: float
    predicted_stage: str
    asset_criticality: str
    action_safety: str
    reasons: list[str]
    blocking_reasons: list[str]
    target_ip: str
    explanation_summary: str | None = None
    risk_trajectory: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to a JSON-serializable dictionary."""
        return {
            "incident_id": self.incident_id,
            "source": self.source,
            "trigger_decision": self.trigger_decision,
            "risk_score": float(self.risk_score),
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "evidence_strength": float(self.evidence_strength),
            "predicted_stage": self.predicted_stage,
            "asset_criticality": self.asset_criticality,
            "action_safety": self.action_safety,
            "reasons": list(self.reasons),
            "blocking_reasons": list(self.blocking_reasons),
            "target_ip": self.target_ip,
            "explanation_summary": self.explanation_summary,
            "risk_trajectory": self.risk_trajectory,
            "timestamp": float(self.timestamp),
        }


@dataclass(frozen=True, slots=True)
class PredictiveIntegrationResult:
    """Immutable result of processing a TriggerDecision through the CyberHeal adapter."""

    status: PredictiveHandoffStatus
    incident_id: str | None
    agent_state: AgentState | None
    review_payload: HumanReviewPayload | None
    reason: str
    trace_id: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        state_dict = None
        if self.agent_state is not None:
            if hasattr(self.agent_state, "model_dump"):
                state_dict = self.agent_state.model_dump()
            else:
                state_dict = self.agent_state.dict()
        return {
            "status": self.status.value,
            "incident_id": self.incident_id,
            "agent_state": state_dict,
            "review_payload": self.review_payload.to_dict() if self.review_payload is not None else None,
            "reason": self.reason,
            "trace_id": self.trace_id,
            "timestamp": float(self.timestamp),
        }


class PredictiveIncidentAdapter:
    """Adapts Phase 2 TriggerDecision events into CyberHeal AgentState and workflows."""

    def __init__(self, default_target_ip: str = "192.168.1.100") -> None:
        """Initialize the predictive incident adapter.

        Args:
            default_target_ip: Fallback IP address when none is explicitly provided.
        """
        self.default_target_ip = default_target_ip

    def process_trigger_decision(
        self,
        decision: TriggerDecision | dict[str, Any],
        target_ip: str | None = None,
        incident_id: str | None = None,
        explanation: ForecastExplanation | None = None,
        forecasted_state: list[float] | np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PredictiveIntegrationResult:
        """Convert a TriggerDecision into a validated CyberHeal handoff result.

        Args:
            decision: TriggerDecision instance or valid dictionary representation.
            target_ip: Optional target IP address for the predictive threat.
            incident_id: Optional incident ID. If None, auto-generated with "INC-PRED-" prefix.
            explanation: Optional Phase 1 ForecastExplanation containing top feature drivers.
            forecasted_state: Optional raw forecasted Network State vector (280-d).
            metadata: Optional dictionary with caller-supplied context (preserved immutably).

        Returns:
            PredictiveIntegrationResult containing handoff status, AgentState (if AUTO_TRIGGER),
            HumanReviewPayload (if HUMAN_REVIEW), or suppression confirmation (if NO_TRIGGER).
        """
        resolved_ip = target_ip or self.default_target_ip

        # 1. Validate and normalize TriggerDecision input
        normalized_decision, validation_err = self._validate_and_normalize_decision(decision)
        if validation_err is not None or normalized_decision is None:
            err_trace_id = f"TRACE-ERR-{uuid.uuid4().hex[:8].upper()}"
            return PredictiveIntegrationResult(
                status=PredictiveHandoffStatus.REJECTED_INVALID,
                incident_id=None,
                agent_state=None,
                review_payload=None,
                reason=f"Integration rejected malformed input: {validation_err}",
                trace_id=err_trace_id,
            )

        # 2. Determine or generate traceable IDs
        if incident_id is None:
            resolved_incident_id = f"INC-PRED-{uuid.uuid4().hex[:8].upper()}"
        else:
            resolved_incident_id = str(incident_id)

        trace_id = f"TRACE-{resolved_incident_id}"

        # 3. Branch on TriggerDecisionType
        dec_type = normalized_decision["decision"]

        # Case A: NO_TRIGGER -> Suppress completely (do not invoke CyberHeal)
        if dec_type == TriggerDecisionType.NO_TRIGGER.value:
            logger.info(f"[{trace_id}] Predictive event suppressed by Trigger Gate (NO_TRIGGER).")
            return PredictiveIntegrationResult(
                status=PredictiveHandoffStatus.SUPPRESSED,
                incident_id=None,
                agent_state=None,
                review_payload=None,
                reason="Predictive event suppressed: risk score or evidence is below triggering threshold (NO_TRIGGER)",
                trace_id=trace_id,
            )

        # Case B: HUMAN_REVIEW -> Escalate to human review without automated remediation
        elif dec_type == TriggerDecisionType.HUMAN_REVIEW.value:
            logger.warning(
                f"[{trace_id}] Predictive event escalated to HUMAN_REVIEW (Risk: {normalized_decision['risk_score']:.2f}, "
                f"Stage: {normalized_decision['predicted_stage']})."
            )
            review_payload = HumanReviewPayload(
                incident_id=resolved_incident_id,
                source="TRINETRA_PREDICTIVE",
                trigger_decision=TriggerDecisionType.HUMAN_REVIEW.value,
                risk_score=float(normalized_decision["risk_score"]),
                confidence=float(normalized_decision["confidence"]) if normalized_decision["confidence"] is not None else None,
                evidence_strength=float(normalized_decision["evidence_strength"]),
                predicted_stage=str(normalized_decision["predicted_stage"]),
                asset_criticality=str(normalized_decision["asset_criticality"]),
                action_safety=str(normalized_decision["action_safety"]),
                reasons=list(normalized_decision["reasons"]),
                blocking_reasons=list(normalized_decision["blocking_reasons"]),
                target_ip=resolved_ip,
                explanation_summary=explanation.summary_text if explanation is not None else None,
                risk_trajectory=normalized_decision.get("risk_trajectory_summary"),
            )
            return PredictiveIntegrationResult(
                status=PredictiveHandoffStatus.ESCALATED_HUMAN_REVIEW,
                incident_id=resolved_incident_id,
                agent_state=None,
                review_payload=review_payload,
                reason=f"Escalated to human review due to {len(normalized_decision['blocking_reasons'])} blocking gate check(s)",
                trace_id=trace_id,
            )

        # Case C: AUTO_TRIGGER -> Construct validated AgentState for CyberHeal intake
        elif dec_type == TriggerDecisionType.AUTO_TRIGGER.value:
            logger.info(
                f"[{trace_id}] Predictive event AUTO_TRIGGER certified for CyberHeal intake "
                f"(Risk: {normalized_decision['risk_score']:.2f}, Stage: {normalized_decision['predicted_stage']})."
            )

            # Extract top predictive features from ForecastExplanation if provided
            top_features: list[dict[str, Any]] = []
            if explanation is not None:
                top_features = [
                    {
                        "feature": f.feature_name,
                        "group": f.group,
                        "current_value": float(f.current_value),
                        "predicted_value": float(f.predicted_value),
                        "delta": float(f.delta),
                        "magnitude": float(f.magnitude),
                        "direction": f.direction,
                        "relative_share": float(f.relative_share),
                    }
                    for f in explanation.top_features
                ]

            # Construct structured alert payload compatible with IntakeAgent
            alert_payload: dict[str, Any] = {
                "source": "TRINETRA_PREDICTIVE",
                "trigger_type": "PREDICTIVE",
                "ip": resolved_ip,
                "severity": "CRITICAL" if normalized_decision["risk_score"] >= 0.85 else "HIGH",
                "metric": "forecasted_network_drift",
                "predicted_stage": normalized_decision["predicted_stage"],
                "risk_score": float(normalized_decision["risk_score"]),
                "confidence": float(normalized_decision["confidence"]) if normalized_decision["confidence"] is not None else None,
                "evidence_strength": float(normalized_decision["evidence_strength"]),
                "asset_criticality": normalized_decision["asset_criticality"],
                "action_safety": normalized_decision["action_safety"],
                "policy_version": normalized_decision["policy_version"],
                "trace_id": trace_id,
                "reasons": list(normalized_decision["reasons"]),
                "blocking_reasons": list(normalized_decision["blocking_reasons"]),
            }
            if metadata:
                alert_payload["context_metadata"] = dict(metadata)

            # Format forecasted state vector safely
            formatted_forecast: list[float] | None = None
            if forecasted_state is not None:
                if isinstance(forecasted_state, np.ndarray):
                    formatted_forecast = [float(x) for x in forecasted_state.flatten()]
                else:
                    formatted_forecast = [float(x) for x in forecasted_state]

            # Construct backward-compatible AgentState for CyberHeal graph
            confidence_val = float(normalized_decision["confidence"]) if normalized_decision["confidence"] is not None else 0.85

            agent_state = AgentState(
                incident_id=resolved_incident_id,
                alert_payload=alert_payload,
                incident_type="PREDICTIVE",
                trigger_decision=normalized_decision,
                forecasted_state=formatted_forecast,
                trajectory_risk_score=float(normalized_decision["risk_score"]),
                attack_stage=str(normalized_decision["predicted_stage"]),
                top_predictive_features=top_features,
                asset_criticality=str(normalized_decision["asset_criticality"]),
                action_safety=str(normalized_decision["action_safety"]),
                confidence_score=confidence_val,
                evidence_strength=float(normalized_decision["evidence_strength"]),
                predictive_reasons=list(normalized_decision["reasons"]),
                predictive_blocking_reasons=list(normalized_decision["blocking_reasons"]),
                risk_trajectory=normalized_decision.get("risk_trajectory_summary"),
                status="OPEN",
                current_step="INTAKE",
            )

            return PredictiveIntegrationResult(
                status=PredictiveHandoffStatus.AUTO_TRIGGERED,
                incident_id=resolved_incident_id,
                agent_state=agent_state,
                review_payload=None,
                reason="Successfully mapped AUTO_TRIGGER event into CyberHeal AgentState",
                trace_id=trace_id,
            )

        else:
            return PredictiveIntegrationResult(
                status=PredictiveHandoffStatus.REJECTED_INVALID,
                incident_id=None,
                agent_state=None,
                review_payload=None,
                reason=f"Unknown decision type: {dec_type}",
                trace_id=trace_id,
            )

    @staticmethod
    def _validate_and_normalize_decision(
        decision: TriggerDecision | dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate TriggerDecision or dictionary and normalize into a dictionary."""
        if isinstance(decision, TriggerDecision):
            raw = decision.to_dict()
        elif isinstance(decision, dict):
            raw = dict(decision)
        else:
            return None, f"Expected TriggerDecision or dict, got {type(decision)}"

        # Validate required fields
        required = ["decision", "risk_score", "evidence_strength", "asset_criticality", "action_safety", "predicted_stage"]
        for key in required:
            if key not in raw:
                return None, f"Missing required field in TriggerDecision: '{key}'"

        # Validate decision value
        valid_decisions = {t.value for t in TriggerDecisionType}
        if raw["decision"] not in valid_decisions:
            return None, f"Invalid decision value: '{raw['decision']}'. Must be one of {valid_decisions}"

        # Validate numerical ranges
        for num_field in ["risk_score", "evidence_strength"]:
            val = raw.get(num_field)
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return None, f"Field '{num_field}' must be a finite float, got {val}"
            if not 0.0 <= val <= 1.0:
                return None, f"Field '{num_field}' must be in [0.0, 1.0], got {val}"

        conf = raw.get("confidence")
        if conf is not None:
            if not isinstance(conf, (int, float)) or math.isnan(conf) or math.isinf(conf):
                return None, f"Field 'confidence' must be a finite float or None, got {conf}"
            if not 0.0 <= conf <= 1.0:
                return None, f"Field 'confidence' must be in [0.0, 1.0], got {conf}"

        raw.setdefault("reasons", [])
        raw.setdefault("blocking_reasons", [])
        raw.setdefault("policy_version", "1.0.0")

        return raw, None
