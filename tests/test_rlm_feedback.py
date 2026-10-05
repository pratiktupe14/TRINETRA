"""Unit and Integration Tests for TRINETRA RLM Feedback Mechanism."""

import json
import math
import tempfile
from pathlib import Path
import pytest

from shield_rca.rlm_feedback import (
    ActionMetadata,
    ActionQuality,
    LearningRecommendation,
    LearningRecord,
    OutcomeMetadata,
    PredictionMetadata,
    PredictionQuality,
    ReflectionMetadata,
    RLMReflectionEngine,
    RLMStore,
    StateReferences,
    TriggerMetadata,
    VerificationMetadata,
    VerificationQuality,
)


def _create_sample_record(
    incident_id: str = "inc-1001",
    risk_score: float = 0.85,
    confidence: float = 0.90,
    v_status: str = "VERIFIED",
    trr: float = 0.80,
    residual: float = 0.20,
    rec_id: str = "rec-1001",
) -> LearningRecord:
    """Helper to build a valid sample LearningRecord."""
    pred = PredictionMetadata(
        risk_score=risk_score,
        confidence=confidence,
        predicted_stage="RECONNAISSANCE",
        risk_trajectory={"h1": 0.85},
    )
    trig = TriggerMetadata(
        decision="AUTO_TRIGGER",
        reasons=["High risk score"],
        blocking_reasons=[],
    )
    act = ActionMetadata(
        action_type="rate_limit_ip",
        target="192.168.1.50",
        execution_mode="SIMULATION",
        status="EXECUTED",
        execution_result={"success": True},
    )
    ver = VerificationMetadata(
        status=v_status,
        threat_reduction_ratio=trr,
        residual_risk=residual,
    )
    outcome = OutcomeMetadata(success=(v_status == "VERIFIED"))
    refl = ReflectionMetadata(
        prediction_quality=PredictionQuality.USEFUL.value,
        action_quality=ActionQuality.EFFECTIVE.value if v_status == "VERIFIED" else ActionQuality.INEFFECTIVE.value,
        verification_quality=v_status,
        recommendations=[LearningRecommendation.NO_CHANGE.value] if v_status == "VERIFIED" else [LearningRecommendation.REVIEW_POLICY.value],
        analysis="Evaluation completed.",
    )
    return LearningRecord(
        record_id=rec_id,
        incident_id=incident_id,
        trace_id="trc-1001",
        timestamp=1700000000.0,
        incident_type="PREDICTIVE",
        prediction=pred,
        trigger=trig,
        action=act,
        verification=ver,
        outcome=outcome,
        reflection=refl,
        state_references=StateReferences(),
    )


def test_valid_incident_produces_learning_record():
    rec = _create_sample_record()
    rec.validate()
    assert rec.record_id == "rec-1001"
    assert rec.incident_id == "inc-1001"
    assert rec.prediction.risk_score == 0.85
    assert rec.trigger.decision == "AUTO_TRIGGER"
    assert rec.action.action_type == "rate_limit_ip"
    assert rec.verification.status == "VERIFIED"


def test_json_serialization_and_deserialization():
    rec = _create_sample_record()
    json_str = rec.to_json()
    assert isinstance(json_str, str)
    
    rec_recovered = LearningRecord.from_json(json_str)
    assert rec_recovered.record_id == rec.record_id
    assert rec_recovered.incident_id == rec.incident_id
    assert rec_recovered.prediction.risk_score == pytest.approx(0.85)
    assert rec_recovered.verification.threat_reduction_ratio == pytest.approx(0.80)


def test_invalid_scores_rejected():
    with pytest.raises(ValueError, match="risk_score out of bounds"):
        _create_sample_record(risk_score=1.5).validate()

    with pytest.raises(ValueError, match="risk_score out of bounds"):
        _create_sample_record(risk_score=-0.1).validate()

    with pytest.raises(ValueError, match="confidence out of bounds"):
        _create_sample_record(confidence=1.2).validate()


def test_nan_and_inf_rejected():
    with pytest.raises(ValueError, match="Invalid risk_score"):
        _create_sample_record(risk_score=float("nan")).validate()

    with pytest.raises(ValueError, match="Invalid risk_score"):
        _create_sample_record(risk_score=float("inf")).validate()

    with pytest.raises(ValueError, match="Invalid threat_reduction_ratio"):
        _create_sample_record(trr=float("nan")).validate()


def test_invalid_enum_rejected():
    rec = _create_sample_record()
    rec.reflection.action_quality = "MAGIC_FIX"
    with pytest.raises(ValueError, match="Invalid action_quality"):
        rec.validate()


def test_duplicate_record_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_records.jsonl"
        store = RLMStore(storage_path)

        rec1 = _create_sample_record(incident_id="inc-same", rec_id="rec-1")
        rec2 = _create_sample_record(incident_id="inc-same", rec_id="rec-2")

        # 1. First record succeeds
        assert store.append_record(rec1) is True

        # 2. Duplicate record with on_duplicate='reject' raises ValueError
        with pytest.raises(ValueError, match="Duplicate record detected"):
            store.append_record(rec2, on_duplicate="reject")

        # 3. Duplicate record with on_duplicate='skip' returns False
        assert store.append_record(rec2, on_duplicate="skip") is False

        # Store still contains only 1 record
        assert store.count() == 1


def test_deterministic_reflection():
    # Case 1: Verified remediation -> positive outcome
    r1 = RLMReflectionEngine.reflect(
        incident_type="PREDICTIVE",
        predicted_risk=0.85,
        confidence=0.90,
        trigger_decision="AUTO_TRIGGER",
        action_status="EXECUTED",
        guardian_approved=True,
        verification_status="VERIFIED",
        threat_reduction_ratio=0.80,
        residual_risk=0.15,
    )
    assert r1.prediction_quality == PredictionQuality.USEFUL.value
    assert r1.action_quality == ActionQuality.EFFECTIVE.value
    assert r1.verification_quality == VerificationQuality.VERIFIED.value
    assert LearningRecommendation.NO_CHANGE.value in r1.recommendations

    # Case 2: Remediation executed but NOT_VERIFIED
    r2 = RLMReflectionEngine.reflect(
        incident_type="PREDICTIVE",
        predicted_risk=0.85,
        confidence=0.90,
        trigger_decision="AUTO_TRIGGER",
        action_status="EXECUTED",
        guardian_approved=True,
        verification_status="NOT_VERIFIED",
        threat_reduction_ratio=0.10,
        residual_risk=0.75,
    )
    assert r2.action_quality == ActionQuality.INEFFECTIVE.value
    assert r2.verification_quality == VerificationQuality.NOT_VERIFIED.value
    assert LearningRecommendation.REVIEW_POLICY.value in r2.recommendations

    # Case 3: Guardian rejection
    r3 = RLMReflectionEngine.reflect(
        incident_type="PREDICTIVE",
        predicted_risk=0.85,
        confidence=0.50,
        trigger_decision="AUTO_TRIGGER",
        action_status="REJECTED",
        guardian_approved=False,
        verification_status="NOT_VERIFIED",
        threat_reduction_ratio=0.0,
        residual_risk=0.85,
    )
    assert r3.action_quality == ActionQuality.REJECTED.value
    assert LearningRecommendation.REVIEW_POLICY.value in r3.recommendations

    # Case 4: High confidence + large prediction error
    pred_vec = [1.0] * 280
    obs_vec = [0.0] * 280  # L2 distance = sqrt(280) ≈ 16.73 >> 2.5
    r4 = RLMReflectionEngine.reflect(
        incident_type="PREDICTIVE",
        predicted_risk=0.88,
        confidence=0.95,
        trigger_decision="AUTO_TRIGGER",
        action_status="EXECUTED",
        guardian_approved=True,
        verification_status="NOT_VERIFIED",
        threat_reduction_ratio=0.05,
        residual_risk=0.80,
        predicted_state=pred_vec,
        observed_state=obs_vec,
    )
    assert r4.prediction_quality == PredictionQuality.POOR.value
    assert LearningRecommendation.RETRAIN_CANDIDATE.value in r4.recommendations

    # Case 5: Inconclusive telemetry -> never infer success from missing data
    r5 = RLMReflectionEngine.reflect(
        incident_type="PREDICTIVE",
        predicted_risk=0.80,
        confidence=0.85,
        trigger_decision="AUTO_TRIGGER",
        action_status="EXECUTED",
        guardian_approved=True,
        verification_status="INCONCLUSIVE",
        threat_reduction_ratio=0.0,
        residual_risk=0.50,
    )
    assert r5.verification_quality == VerificationQuality.INCONCLUSIVE.value
    assert r5.action_quality == ActionQuality.INCONCLUSIVE.value
    assert LearningRecommendation.REVIEW_DATA.value in r5.recommendations


def test_insufficient_evidence_no_automatic_retrain():
    # Moderate confidence, ordinary verification failure without state vector drift evidence
    r = RLMReflectionEngine.reflect(
        incident_type="PREDICTIVE",
        predicted_risk=0.72,
        confidence=0.65,  # Moderate/marginal confidence
        trigger_decision="AUTO_TRIGGER",
        action_status="EXECUTED",
        guardian_approved=True,
        verification_status="NOT_VERIFIED",
        threat_reduction_ratio=0.20,
        residual_risk=0.55,
    )
    # Must NOT recommend RETRAIN_CANDIDATE without strong evidence of model drift
    assert LearningRecommendation.RETRAIN_CANDIDATE.value not in r.recommendations
    assert LearningRecommendation.REVIEW_POLICY.value in r.recommendations
