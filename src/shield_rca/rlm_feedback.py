"""TRINETRA Reflective Learning Mechanism (RLM) Feedback Engine.

Provides deterministic, structured reflection on cyber incident lifecycles:
    Capture -> Compare -> Reflect -> Recommend -> Store

Enforces strict data integrity, audit logging, and offline file-based persistence.
NOTE: RLM is NOT an LLM. It operates deterministically using metric-driven rules.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class PredictionQuality(str, Enum):
    """Quality classification of a forecast given post-incident observations."""
    USEFUL = "USEFUL"
    PARTIALLY_USEFUL = "PARTIALLY_USEFUL"
    POOR = "POOR"
    INCONCLUSIVE = "INCONCLUSIVE"


class ActionQuality(str, Enum):
    """Operational evaluation of remediation action taken."""
    EFFECTIVE = "EFFECTIVE"
    PARTIALLY_EFFECTIVE = "PARTIALLY_EFFECTIVE"
    INEFFECTIVE = "INEFFECTIVE"
    REJECTED = "REJECTED"
    NOT_EXECUTED = "NOT_EXECUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class VerificationQuality(str, Enum):
    """Integrity and certainty of the verification step."""
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


class LearningRecommendation(str, Enum):
    """Structured offline recommendations produced by RLM reflection."""
    NO_CHANGE = "NO_CHANGE"
    RETRAIN_CANDIDATE = "RETRAIN_CANDIDATE"
    REVIEW_POLICY = "REVIEW_POLICY"
    REVIEW_FEATURES = "REVIEW_FEATURES"
    REVIEW_THRESHOLD = "REVIEW_THRESHOLD"
    REVIEW_DATA = "REVIEW_DATA"


@dataclass
class PredictionMetadata:
    """Captured predictive intelligence metadata."""
    risk_score: float
    confidence: float
    predicted_stage: str
    risk_trajectory: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if math.isnan(self.risk_score) or math.isinf(self.risk_score):
            raise ValueError(f"Invalid risk_score: {self.risk_score}")
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(f"risk_score out of bounds [0, 1]: {self.risk_score}")
        if math.isnan(self.confidence) or math.isinf(self.confidence):
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of bounds [0, 1]: {self.confidence}")
        if not self.predicted_stage:
            raise ValueError("predicted_stage cannot be empty")


@dataclass
class TriggerMetadata:
    """Captured Trigger Gate decision context."""
    decision: str
    reasons: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.decision:
            raise ValueError("Trigger decision cannot be empty")


@dataclass
class ActionMetadata:
    """Captured remediation action context and execution outcome."""
    action_type: str
    target: str
    execution_mode: str
    status: str
    execution_result: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if not self.action_type:
            raise ValueError("action_type cannot be empty")
        if not self.execution_mode:
            raise ValueError("execution_mode cannot be empty")
        if not self.status:
            raise ValueError("status cannot be empty")


@dataclass
class VerificationMetadata:
    """Captured post-remediation verification metrics."""
    status: str
    threat_reduction_ratio: float
    residual_risk: float

    def validate(self) -> None:
        if math.isnan(self.threat_reduction_ratio) or math.isinf(self.threat_reduction_ratio):
            raise ValueError(f"Invalid threat_reduction_ratio: {self.threat_reduction_ratio}")
        if math.isnan(self.residual_risk) or math.isinf(self.residual_risk):
            raise ValueError(f"Invalid residual_risk: {self.residual_risk}")
        if not (0.0 <= self.residual_risk <= 1.0):
            raise ValueError(f"residual_risk out of bounds [0, 1]: {self.residual_risk}")
        valid_statuses = {s.value for s in VerificationQuality}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid verification status: '{self.status}'. Must be one of {valid_statuses}")


@dataclass
class OutcomeMetadata:
    """Overall operational outcome of the incident response."""
    success: bool
    failure_reason: Optional[str] = None


@dataclass
class ReflectionMetadata:
    """Structured results of deterministic RLM reflection."""
    prediction_quality: str
    action_quality: str
    verification_quality: str
    recommendations: List[str] = field(default_factory=list)
    analysis: str = ""

    def validate(self) -> None:
        valid_pq = {q.value for q in PredictionQuality}
        if self.prediction_quality not in valid_pq:
            raise ValueError(f"Invalid prediction_quality: '{self.prediction_quality}'")
        valid_aq = {q.value for q in ActionQuality}
        if self.action_quality not in valid_aq:
            raise ValueError(f"Invalid action_quality: '{self.action_quality}'")
        valid_vq = {q.value for q in VerificationQuality}
        if self.verification_quality not in valid_vq:
            raise ValueError(f"Invalid verification_quality: '{self.verification_quality}'")
        valid_rec = {r.value for r in LearningRecommendation}
        for rec in self.recommendations:
            if rec not in valid_rec:
                raise ValueError(f"Invalid recommendation: '{rec}'")


@dataclass
class StateReferences:
    """Reference copies of 280-dimensional Network State vectors (if available)."""
    current_state: Optional[List[float]] = None
    predicted_state: Optional[List[float]] = None
    observed_state: Optional[List[float]] = None

    def validate(self) -> None:
        for name, vec in [("current_state", self.current_state),
                          ("predicted_state", self.predicted_state),
                          ("observed_state", self.observed_state)]:
            if vec is not None:
                if not isinstance(vec, list):
                    raise ValueError(f"{name} must be a list of floats")
                for i, v in enumerate(vec):
                    if not isinstance(v, (int, float)):
                        raise ValueError(f"{name}[{i}] is not a float: {v}")
                    if math.isnan(v) or math.isinf(v):
                        raise ValueError(f"{name}[{i}] contains NaN or Inf")


@dataclass
class LearningRecord:
    """Typed, validated unit of learning produced by RLM."""
    record_id: str
    incident_id: str
    trace_id: str
    timestamp: float
    incident_type: str  # "PREDICTIVE" or "REACTIVE"
    prediction: PredictionMetadata
    trigger: TriggerMetadata
    action: ActionMetadata
    verification: VerificationMetadata
    outcome: OutcomeMetadata
    reflection: ReflectionMetadata
    state_references: StateReferences = field(default_factory=StateReferences)

    def validate(self) -> None:
        """Validate all sub-components strictly."""
        if not self.record_id:
            raise ValueError("record_id cannot be empty")
        if not self.incident_id:
            raise ValueError("incident_id cannot be empty")
        if not self.trace_id:
            raise ValueError("trace_id cannot be empty")
        if math.isnan(self.timestamp) or math.isinf(self.timestamp) or self.timestamp <= 0:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")
        if self.incident_type not in ("PREDICTIVE", "REACTIVE"):
            raise ValueError(f"Invalid incident_type: '{self.incident_type}'")

        self.prediction.validate()
        self.trigger.validate()
        self.action.validate()
        self.verification.validate()
        self.reflection.validate()
        self.state_references.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to fully serializable dictionary."""
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LearningRecord:
        """Parse and validate from dictionary."""
        rec = cls(
            record_id=data["record_id"],
            incident_id=data["incident_id"],
            trace_id=data["trace_id"],
            timestamp=float(data["timestamp"]),
            incident_type=data.get("incident_type", "PREDICTIVE"),
            prediction=PredictionMetadata(**data["prediction"]),
            trigger=TriggerMetadata(**data["trigger"]),
            action=ActionMetadata(**data["action"]),
            verification=VerificationMetadata(**data["verification"]),
            outcome=OutcomeMetadata(**data["outcome"]),
            reflection=ReflectionMetadata(**data["reflection"]),
            state_references=StateReferences(**data.get("state_references", {})),
        )
        rec.validate()
        return rec

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> LearningRecord:
        """Deserialize and validate from JSON string."""
        return cls.from_dict(json.loads(json_str))


class RLMReflectionEngine:
    """Deterministic reasoning engine for reflecting on incident outcomes.
    
    Translates observed vs predicted differences into structured feedback
    and non-binding offline recommendations.
    """

    @staticmethod
    def compare_predicted_vs_observed(
        predicted_state: Optional[List[float]],
        observed_state: Optional[List[float]],
        predicted_risk: float,
        observed_risk: float,
    ) -> Dict[str, Any]:
        """Quantify differences between forecasted and post-action observed metrics."""
        risk_error = abs(predicted_risk - observed_risk)
        risk_delta = predicted_risk - observed_risk  # Positive if threat was lowered

        state_l2_drift: Optional[float] = None
        if predicted_state is not None and observed_state is not None:
            if len(predicted_state) == len(observed_state) and len(predicted_state) > 0:
                p_arr = np.array(predicted_state, dtype=np.float32)
                o_arr = np.array(observed_state, dtype=np.float32)
                state_l2_drift = float(np.linalg.norm(p_arr - o_arr))

        return {
            "risk_error": risk_error,
            "risk_delta": risk_delta,
            "state_l2_drift": state_l2_drift,
        }

    @classmethod
    def reflect(
        cls,
        incident_type: str,
        predicted_risk: float,
        confidence: float,
        trigger_decision: str,
        action_status: str,
        guardian_approved: bool,
        verification_status: str,
        threat_reduction_ratio: float,
        residual_risk: float,
        predicted_state: Optional[List[float]] = None,
        observed_state: Optional[List[float]] = None,
    ) -> ReflectionMetadata:
        """Execute deterministic reflection rules to produce ReflectionMetadata."""
        comp = cls.compare_predicted_vs_observed(
            predicted_state=predicted_state,
            observed_state=observed_state,
            predicted_risk=predicted_risk,
            observed_risk=residual_risk,
        )

        recommendations: List[str] = []
        analysis_parts: List[str] = []

        # 1. Verification Quality
        if verification_status == "VERIFIED":
            v_qual = VerificationQuality.VERIFIED.value
        elif verification_status == "NOT_VERIFIED":
            v_qual = VerificationQuality.NOT_VERIFIED.value
        elif verification_status == "PARTIALLY_VERIFIED":
            v_qual = VerificationQuality.PARTIALLY_VERIFIED.value
        else:
            v_qual = VerificationQuality.INCONCLUSIVE.value

        # 2. Action Quality & Policy Reflection
        if not guardian_approved or action_status == "REJECTED":
            a_qual = ActionQuality.REJECTED.value
            analysis_parts.append("Remediation action was blocked by Guardian policy.")
            recommendations.append(LearningRecommendation.REVIEW_POLICY.value)
        elif action_status == "NOT_EXECUTED":
            a_qual = ActionQuality.NOT_EXECUTED.value
            analysis_parts.append("Remediation was not executed.")
            recommendations.append(LearningRecommendation.REVIEW_POLICY.value)
        elif v_qual == VerificationQuality.VERIFIED.value:
            a_qual = ActionQuality.EFFECTIVE.value
            analysis_parts.append(
                f"Remediation effective: threat reduced by {threat_reduction_ratio * 100:.1f}% "
                f"(Residual Risk: {residual_risk:.2f})."
            )
        elif v_qual == VerificationQuality.PARTIALLY_VERIFIED.value:
            a_qual = ActionQuality.PARTIALLY_EFFECTIVE.value
            analysis_parts.append("Remediation was partially effective but did not meet full target.")
            recommendations.append(LearningRecommendation.REVIEW_POLICY.value)
        elif v_qual == VerificationQuality.NOT_VERIFIED.value:
            a_qual = ActionQuality.INEFFECTIVE.value
            analysis_parts.append(
                f"Remediation ineffective: threat reduction {threat_reduction_ratio * 100:.1f}% "
                f"insufficient (Residual Risk: {residual_risk:.2f})."
            )
            recommendations.append(LearningRecommendation.REVIEW_POLICY.value)
        else:
            a_qual = ActionQuality.INCONCLUSIVE.value
            analysis_parts.append("Action impact inconclusive due to insufficient post-action telemetry.")
            recommendations.append(LearningRecommendation.REVIEW_DATA.value)

        # 3. Prediction Quality
        if incident_type == "REACTIVE":
            p_qual = PredictionQuality.USEFUL.value if v_qual == VerificationQuality.VERIFIED.value else PredictionQuality.INCONCLUSIVE.value
        elif v_qual == VerificationQuality.INCONCLUSIVE.value:
            p_qual = PredictionQuality.INCONCLUSIVE.value
            if LearningRecommendation.REVIEW_DATA.value not in recommendations:
                recommendations.append(LearningRecommendation.REVIEW_DATA.value)
        elif v_qual == VerificationQuality.VERIFIED.value:
            p_qual = PredictionQuality.USEFUL.value
            analysis_parts.append("Forecast accurately preempted verified threat.")
        else:
            # Threat was not verified after action
            # Check if prediction had high confidence but large error
            drift = comp.get("state_l2_drift")
            if confidence >= 0.80 and (drift is not None and drift > 2.5):
                p_qual = PredictionQuality.POOR.value
                analysis_parts.append(
                    f"High-confidence forecast exhibited significant trajectory drift (L2: {drift:.2f})."
                )
                recommendations.append(LearningRecommendation.RETRAIN_CANDIDATE.value)
            else:
                p_qual = PredictionQuality.PARTIALLY_USEFUL.value
                analysis_parts.append("Forecast identified anomalous pattern but containment did not resolve.")

        # 4. Human Review & Edge Cases
        if trigger_decision == "HUMAN_REVIEW" and not guardian_approved:
            if LearningRecommendation.REVIEW_POLICY.value not in recommendations:
                recommendations.append(LearningRecommendation.REVIEW_POLICY.value)

        # 5. Default Recommendation if successful
        if v_qual == VerificationQuality.VERIFIED.value and a_qual == ActionQuality.EFFECTIVE.value and not recommendations:
            recommendations.append(LearningRecommendation.NO_CHANGE.value)

        # Deduplicate recommendations while preserving order
        seen = set()
        dedup_recs = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                dedup_recs.append(r)

        return ReflectionMetadata(
            prediction_quality=p_qual,
            action_quality=a_qual,
            verification_quality=v_qual,
            recommendations=dedup_recs,
            analysis=" ".join(analysis_parts),
        )

    @classmethod
    def reflect_from_agent_state(
        cls,
        state: Any,
        store: Optional[RLMStore] = None,
    ) -> LearningRecord:
        """Create a complete LearningRecord from an AgentState and persist it."""
        import time

        incident_id = getattr(state, "incident_id", str(uuid.uuid4()))
        incident_type = getattr(state, "incident_type", "PREDICTIVE")
        trace_id = getattr(state, "trace_id", str(uuid.uuid4()))

        # Prediction metadata
        pred_risk = getattr(state, "trajectory_risk_score", getattr(state, "risk_score", 0.0))
        pred_meta = PredictionMetadata(
            risk_score=float(min(max(pred_risk, 0.0), 1.0)),
            confidence=float(min(max(getattr(state, "confidence_score", 0.8), 0.0), 1.0)),
            predicted_stage=str(getattr(state, "attack_stage", "NORMAL")),
            risk_trajectory=getattr(state, "risk_trajectory", None),
        )

        # Trigger metadata
        trig_decision_dict = getattr(state, "trigger_decision", {}) or {}
        trig_meta = TriggerMetadata(
            decision=str(trig_decision_dict.get("decision", "AUTO_TRIGGER" if getattr(state, "guardian_approved", True) else "HUMAN_REVIEW")),
            reasons=getattr(state, "predictive_reasons", []),
            blocking_reasons=getattr(state, "predictive_blocking_reasons", []),
        )

        # Action metadata
        plan = getattr(state, "remediation_plan", [])
        action_name = plan[0]["action"] if plan and isinstance(plan, list) else "none"
        target_name = plan[0].get("target", "none") if plan and isinstance(plan, list) and isinstance(plan[0], dict) else "none"
        exec_results = getattr(state, "execution_results", [])
        status = "EXECUTED" if exec_results else ("REJECTED" if not getattr(state, "guardian_approved", True) else "NOT_EXECUTED")
        
        act_meta = ActionMetadata(
            action_type=str(action_name),
            target=str(target_name),
            execution_mode="SIMULATION",
            status=status,
            execution_result={"results": exec_results} if exec_results else None,
        )

        # Verification metadata
        v_rep = getattr(state, "verification_report", None)
        if v_rep and hasattr(v_rep, "threat_reduction_ratio"):
            trr = float(v_rep.threat_reduction_ratio)
            resid = float(min(max(v_rep.residual_risk, 0.0), 1.0))
            v_status = str(v_rep.status.value if hasattr(v_rep.status, "value") else v_rep.status)
        else:
            passed = getattr(state, "verification_passed", False)
            trr = 0.85 if passed else 0.10
            resid = 0.15 if passed else 0.75
            v_status = "VERIFIED" if passed else "NOT_VERIFIED"

        ver_meta = VerificationMetadata(
            status=v_status,
            threat_reduction_ratio=trr,
            residual_risk=resid,
        )

        # Outcome metadata
        outcome_meta = OutcomeMetadata(
            success=getattr(state, "verification_passed", False),
            failure_reason="Verification failed" if not getattr(state, "verification_passed", False) else None,
        )

        # State references
        f_state = getattr(state, "forecasted_state", None)
        ref_states = StateReferences(
            current_state=None,
            predicted_state=f_state if isinstance(f_state, list) else None,
            observed_state=None,
        )

        # Reflection
        refl = cls.reflect(
            incident_type=incident_type,
            predicted_risk=pred_meta.risk_score,
            confidence=pred_meta.confidence,
            trigger_decision=trig_meta.decision,
            action_status=act_meta.status,
            guardian_approved=getattr(state, "guardian_approved", True),
            verification_status=ver_meta.status,
            threat_reduction_ratio=ver_meta.threat_reduction_ratio,
            residual_risk=ver_meta.residual_risk,
            predicted_state=ref_states.predicted_state,
            observed_state=ref_states.observed_state,
        )

        rec = LearningRecord(
            record_id=f"rec-{uuid.uuid4().hex[:12]}",
            incident_id=str(incident_id),
            trace_id=str(trace_id),
            timestamp=float(time.time()),
            incident_type=incident_type,
            prediction=pred_meta,
            trigger=trig_meta,
            action=act_meta,
            verification=ver_meta,
            outcome=outcome_meta,
            reflection=refl,
            state_references=ref_states,
        )

        if store is not None:
            store.append_record(rec, on_duplicate="skip")

        return rec


class RLMStore:
    """Offline, file-based store for RLM LearningRecords.
    
    Supports strictly append-only JSON Lines format with deduplication
    and format validation.
    """

    def __init__(self, storage_path: Path | str = "artifacts/rlm/learning_records.jsonl") -> None:
        self.storage_path = Path(storage_path)

    def _ensure_dir(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def append_record(
        self,
        record: LearningRecord,
        on_duplicate: str = "reject",
    ) -> bool:
        """Append a learning record to the JSONL dataset.
        
        Args:
            record: Validated LearningRecord.
            on_duplicate: "reject" (raise ValueError) or "skip" (return False).
            
        Returns:
            True if written, False if skipped.
        """
        record.validate()
        self._ensure_dir()

        # Check for duplicates by incident_id or record_id
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if row.get("incident_id") == record.incident_id or row.get("record_id") == record.record_id:
                            if on_duplicate == "reject":
                                raise ValueError(f"Duplicate record detected for incident {record.incident_id}")
                            return False
                    except json.JSONDecodeError:
                        continue

        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

        return True

    def read_records(self) -> List[LearningRecord]:
        """Read and validate all learning records in the store."""
        if not self.storage_path.exists():
            return []

        records: List[LearningRecord] = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(LearningRecord.from_dict(data))
                except Exception as e:
                    raise ValueError(f"Invalid record at line {line_no} in {self.storage_path}: {e}")

        return records

    def get_record(self, record_id: str) -> Optional[LearningRecord]:
        """Retrieve a specific record by record_id."""
        for rec in self.read_records():
            if rec.record_id == record_id:
                return rec
        return None

    def filter_records(self, predicate: Callable[[LearningRecord], bool]) -> List[LearningRecord]:
        """Filter records by a predicate function."""
        return [r for r in self.read_records() if predicate(r)]

    def count(self) -> int:
        """Count total valid records."""
        return len(self.read_records())
