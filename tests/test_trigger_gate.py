"""Unit tests for the Predictive Trigger Gate (Phase 2).

Verifies multi-dimensional safety gating:
1. Low risk -> NO_TRIGGER
2. High risk + low confidence -> HUMAN_REVIEW
3. High risk + high confidence + strong evidence + safe action + acceptable asset -> AUTO_TRIGGER
4. High risk + unknown action safety -> HUMAN_REVIEW
5. Critical asset + strong prediction -> HUMAN_REVIEW
6. Weak evidence -> HUMAN_REVIEW
7. Unknown predicted stage -> HUMAN_REVIEW
8. Normal / low-risk prediction -> NO_TRIGGER
9. Boundary values at thresholds
10. Values just below thresholds
11. Values just above thresholds
12. Invalid risk score
13. Invalid confidence
14. Invalid evidence strength
15. Invalid asset criticality
16. Invalid action safety
17. NaN rejection
18. Inf rejection
19. Determinism
20. Reason generation
21. Blocking reason generation
22. Policy customization
23. Different policies produce different decisions
24. Risk trajectory consistency handling
25. JSON serialization
26. Phase 1 output compatibility
27. No mutation of input objects
28. Offline execution
"""

from __future__ import annotations

import json
import math
import pytest
import numpy as np

from shield_rca.network_state.state_schema import STATE_FEATURE_DIM
from shield_rca.world_model.explainability import (
    FeatureChange,
    ForecastExplanation,
    GroupAttribution,
)
from shield_rca.world_model.risk_estimator import (
    HorizonRiskEstimate,
    PredictiveRiskEstimator,
    RiskTrajectory,
    StateChangeSummary,
)
from shield_rca.world_model.stage_mapper import AttackStage, StageMappingResult
from shield_rca.world_model.trigger_gate import (
    ActionSafety,
    AssetCriticality,
    PredictiveTriggerGate,
    TriggerDecision,
    TriggerDecisionType,
    TriggerPolicy,
)


@pytest.fixture
def default_gate() -> PredictiveTriggerGate:
    """Fixture providing a trigger gate with default policy."""
    return PredictiveTriggerGate()


# --- Core Outcomes Tests (1 - 8) ---

def test_low_risk_yields_no_trigger(default_gate: PredictiveTriggerGate):
    """1. Low risk below review threshold must yield NO_TRIGGER."""
    decision = default_gate.evaluate(
        risk_score=0.20,
        confidence=0.85,
        evidence_strength=0.70,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.RECONNAISSANCE,
    )
    assert decision.decision == TriggerDecisionType.NO_TRIGGER
    assert any("below review threshold" in r for r in decision.reasons)
    assert len(decision.blocking_reasons) > 0


def test_high_risk_low_confidence_yields_human_review(default_gate: PredictiveTriggerGate):
    """2. High risk with confidence below auto threshold must escalate to HUMAN_REVIEW."""
    decision = default_gate.evaluate(
        risk_score=0.85,
        confidence=0.30,  # Below confidence_auto_threshold (0.75)
        evidence_strength=0.75,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert decision.decision == TriggerDecisionType.HUMAN_REVIEW
    assert any("Confidence" in br and "below auto threshold" in br for br in decision.blocking_reasons)
    assert any("meets or exceeds review threshold" in r for r in decision.reasons)


def test_high_risk_high_confidence_safe_yields_auto_trigger(default_gate: PredictiveTriggerGate):
    """3. High risk, high confidence, strong evidence, safe action, acceptable asset yields AUTO_TRIGGER."""
    decision = default_gate.evaluate(
        risk_score=0.88,
        confidence=0.85,
        evidence_strength=0.80,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert decision.decision == TriggerDecisionType.AUTO_TRIGGER
    assert len(decision.blocking_reasons) == 0
    assert len(decision.reasons) >= 5
    assert any("approved for automated remediation" in r for r in decision.reasons)


def test_high_risk_unknown_action_safety_yields_human_review(default_gate: PredictiveTriggerGate):
    """4. High risk with UNKNOWN action safety must conservatively escalate to HUMAN_REVIEW."""
    decision = default_gate.evaluate(
        risk_score=0.88,
        confidence=0.85,
        evidence_strength=0.80,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.UNKNOWN,
        predicted_stage=AttackStage.IMPACT,
    )
    assert decision.decision == TriggerDecisionType.HUMAN_REVIEW
    assert any("Action safety is UNKNOWN" in br for br in decision.blocking_reasons)


def test_critical_asset_yields_human_review(default_gate: PredictiveTriggerGate):
    """5. Critical asset with otherwise strong prediction requires human sign-off."""
    decision = default_gate.evaluate(
        risk_score=0.90,
        confidence=0.90,
        evidence_strength=0.85,
        asset_criticality=AssetCriticality.CRITICAL,  # Exceeds maximum MEDIUM
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert decision.decision == TriggerDecisionType.HUMAN_REVIEW
    assert any("Asset criticality (CRITICAL) exceeds auto threshold" in br for br in decision.blocking_reasons)


def test_weak_evidence_yields_human_review(default_gate: PredictiveTriggerGate):
    """6. High risk but weak supporting evidence must escalate to HUMAN_REVIEW."""
    decision = default_gate.evaluate(
        risk_score=0.85,
        confidence=0.85,
        evidence_strength=0.25,  # Below evidence_auto_threshold (0.60)
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert decision.decision == TriggerDecisionType.HUMAN_REVIEW
    assert any("Evidence strength" in br and "below auto threshold" in br for br in decision.blocking_reasons)


def test_unknown_predicted_stage_yields_conservative_review(default_gate: PredictiveTriggerGate):
    """7. Unknown attack stage with high risk must be gated to HUMAN_REVIEW."""
    decision = default_gate.evaluate(
        risk_score=0.85,
        confidence=0.85,
        evidence_strength=0.80,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.UNKNOWN,
    )
    assert decision.decision == TriggerDecisionType.HUMAN_REVIEW
    assert any("Predicted attack stage is UNKNOWN" in br for br in decision.blocking_reasons)


def test_normal_stage_low_risk_yields_no_trigger(default_gate: PredictiveTriggerGate):
    """8. Normal stage with low or moderate risk must yield NO_TRIGGER."""
    decision = default_gate.evaluate(
        risk_score=0.45,  # Above 0.35 review threshold, but stage is NORMAL and < 0.75
        confidence=0.90,
        evidence_strength=0.20,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.NORMAL,
    )
    assert decision.decision == TriggerDecisionType.NO_TRIGGER
    assert any("NORMAL" in r for r in decision.reasons)


# --- Threshold Boundary Tests (9 - 11) ---

def test_exact_threshold_boundaries():
    """9. Exact auto and review threshold values must qualify."""
    policy = TriggerPolicy(
        risk_auto_threshold=0.75,
        risk_review_threshold=0.35,
        confidence_auto_threshold=0.75,
        evidence_auto_threshold=0.60,
    )
    gate = PredictiveTriggerGate(policy)

    # Exact auto thresholds
    d_auto = gate.evaluate(
        risk_score=0.75,
        confidence=0.75,
        evidence_strength=0.60,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert d_auto.decision == TriggerDecisionType.AUTO_TRIGGER

    # Exact review threshold
    d_review = gate.evaluate(
        risk_score=0.35,
        confidence=0.75,
        evidence_strength=0.60,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert d_review.decision == TriggerDecisionType.HUMAN_REVIEW


def test_values_just_below_thresholds(default_gate: PredictiveTriggerGate):
    """10. Values epsilon below auto thresholds must degrade to HUMAN_REVIEW, below review to NO_TRIGGER."""
    # Just below auto risk threshold (0.7499 < 0.75)
    d_below_auto = default_gate.evaluate(
        risk_score=0.7499,
        confidence=0.85,
        evidence_strength=0.80,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert d_below_auto.decision == TriggerDecisionType.HUMAN_REVIEW
    assert any("below auto threshold" in br for br in d_below_auto.blocking_reasons)

    # Just below review risk threshold (0.3499 < 0.35)
    d_below_review = default_gate.evaluate(
        risk_score=0.3499,
        confidence=0.85,
        evidence_strength=0.80,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert d_below_review.decision == TriggerDecisionType.NO_TRIGGER


def test_values_just_above_thresholds(default_gate: PredictiveTriggerGate):
    """11. Values epsilon above auto thresholds must qualify for AUTO_TRIGGER."""
    d_above_auto = default_gate.evaluate(
        risk_score=0.7501,
        confidence=0.7501,
        evidence_strength=0.6001,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert d_above_auto.decision == TriggerDecisionType.AUTO_TRIGGER


# --- Validation and Error Handling Tests (12 - 18) ---

def test_invalid_risk_score(default_gate: PredictiveTriggerGate):
    """12. Negative or out-of-range risk scores must raise ValueError."""
    with pytest.raises(ValueError, match="risk_score"):
        default_gate.evaluate(risk_score=-0.1)

    with pytest.raises(ValueError, match="risk_score"):
        default_gate.evaluate(risk_score=1.5)


def test_invalid_confidence(default_gate: PredictiveTriggerGate):
    """13. Negative or out-of-range confidence must raise ValueError."""
    with pytest.raises(ValueError, match="confidence"):
        default_gate.evaluate(risk_score=0.8, confidence=-0.05)

    with pytest.raises(ValueError, match="confidence"):
        default_gate.evaluate(risk_score=0.8, confidence=1.2)


def test_invalid_evidence_strength(default_gate: PredictiveTriggerGate):
    """14. Negative or out-of-range evidence strength must raise ValueError."""
    with pytest.raises(ValueError, match="evidence_strength"):
        default_gate.evaluate(risk_score=0.8, evidence_strength=-0.1)

    with pytest.raises(ValueError, match="evidence_strength"):
        default_gate.evaluate(risk_score=0.8, evidence_strength=2.0)


def test_invalid_asset_criticality(default_gate: PredictiveTriggerGate):
    """15. Invalid string for asset criticality must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid asset_criticality"):
        default_gate.evaluate(risk_score=0.8, asset_criticality="ULTRA_HIGH")


def test_invalid_action_safety(default_gate: PredictiveTriggerGate):
    """16. Invalid string for action safety must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid action_safety"):
        default_gate.evaluate(risk_score=0.8, action_safety="ABSOLUTELY_SAFE")


def test_nan_rejection(default_gate: PredictiveTriggerGate):
    """17. NaN inputs must be rejected with ValueError."""
    with pytest.raises(ValueError, match="finite float"):
        default_gate.evaluate(risk_score=float("nan"))

    with pytest.raises(ValueError, match="finite float"):
        default_gate.evaluate(risk_score=0.8, confidence=float("nan"))

    with pytest.raises(ValueError, match="finite float"):
        default_gate.evaluate(risk_score=0.8, evidence_strength=float("nan"))


def test_inf_rejection(default_gate: PredictiveTriggerGate):
    """18. Inf inputs must be rejected with ValueError."""
    with pytest.raises(ValueError, match="finite float"):
        default_gate.evaluate(risk_score=float("inf"))

    with pytest.raises(ValueError, match="finite float"):
        default_gate.evaluate(risk_score=0.8, confidence=float("-inf"))

    with pytest.raises(ValueError, match="finite float"):
        default_gate.evaluate(risk_score=0.8, evidence_strength=float("inf"))


# --- Determinism & Explainability Tests (19 - 21) ---

def test_strict_determinism(default_gate: PredictiveTriggerGate):
    """19. Repeated evaluations of identical inputs must yield identical decisions."""
    kwargs = {
        "risk_score": 0.82,
        "confidence": 0.78,
        "evidence_strength": 0.65,
        "asset_criticality": AssetCriticality.LOW,
        "action_safety": ActionSafety.SAFE,
        "predicted_stage": AttackStage.RECONNAISSANCE,
    }
    d1 = default_gate.evaluate(**kwargs)
    d2 = default_gate.evaluate(**kwargs)

    assert d1.decision == d2.decision
    assert d1.reasons == d2.reasons
    assert d1.blocking_reasons == d2.blocking_reasons
    assert d1.to_dict() == d2.to_dict()


def test_reason_generation(default_gate: PredictiveTriggerGate):
    """20. Every decision must emit descriptive, non-empty reasons."""
    decision = default_gate.evaluate(
        risk_score=0.85,
        confidence=0.80,
        evidence_strength=0.75,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert len(decision.reasons) > 0
    for reason in decision.reasons:
        assert isinstance(reason, str)
        assert len(reason.strip()) > 0


def test_blocking_reason_generation(default_gate: PredictiveTriggerGate):
    """21. Gated decisions (HUMAN_REVIEW / NO_TRIGGER) must explain why AUTO_TRIGGER was blocked."""
    decision = default_gate.evaluate(
        risk_score=0.85,
        confidence=0.50,  # Below auto
        evidence_strength=0.40,  # Below auto
        asset_criticality=AssetCriticality.HIGH,  # Exceeds MEDIUM
        action_safety=ActionSafety.RESTRICTED,  # Not SAFE
        predicted_stage=AttackStage.UNKNOWN,  # Blocked stage
    )
    assert decision.decision == TriggerDecisionType.HUMAN_REVIEW
    assert len(decision.blocking_reasons) >= 4
    # Check that each failing dimension is represented
    br_text = " ".join(decision.blocking_reasons)
    assert "Confidence" in br_text
    assert "Evidence strength" in br_text
    assert "Asset criticality" in br_text
    assert "RESTRICTED" in br_text
    assert "UNKNOWN" in br_text


# --- Policy Customization Tests (22 - 23) ---

def test_policy_customization_and_validation():
    """22. Custom policy parameters must be enforced and invalid policies rejected."""
    custom_policy = TriggerPolicy(
        risk_auto_threshold=0.90,
        risk_review_threshold=0.50,
        confidence_auto_threshold=0.85,
        maximum_auto_asset_criticality=AssetCriticality.LOW,
    )
    gate = PredictiveTriggerGate(custom_policy)

    # 0.85 risk satisfies default (0.75) but NOT custom (0.90)
    d = gate.evaluate(
        risk_score=0.85,
        confidence=0.90,
        evidence_strength=0.80,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert d.decision == TriggerDecisionType.HUMAN_REVIEW

    # Invalid policy: review threshold higher than auto threshold
    with pytest.raises(ValueError, match="cannot exceed"):
        TriggerPolicy(risk_review_threshold=0.80, risk_auto_threshold=0.60)


def test_different_policies_produce_different_decisions():
    """23. Strict policy gates to HUMAN_REVIEW while lenient policy allows AUTO_TRIGGER."""
    lenient_policy = TriggerPolicy(
        risk_auto_threshold=0.60,
        confidence_auto_threshold=0.60,
        evidence_auto_threshold=0.50,
        maximum_auto_asset_criticality=AssetCriticality.HIGH,
    )
    strict_policy = TriggerPolicy(
        risk_auto_threshold=0.95,
        confidence_auto_threshold=0.95,
        evidence_auto_threshold=0.90,
        maximum_auto_asset_criticality=AssetCriticality.LOW,
    )

    args = {
        "risk_score": 0.80,
        "confidence": 0.80,
        "evidence_strength": 0.70,
        "asset_criticality": AssetCriticality.HIGH,
        "action_safety": ActionSafety.SAFE,
        "predicted_stage": AttackStage.IMPACT,
    }

    lenient_decision = PredictiveTriggerGate(lenient_policy).evaluate(**args)
    strict_decision = PredictiveTriggerGate(strict_policy).evaluate(**args)

    assert lenient_decision.decision == TriggerDecisionType.AUTO_TRIGGER
    assert strict_decision.decision == TriggerDecisionType.HUMAN_REVIEW


# --- Trajectory Consistency & Phase 1 Compatibility (24 - 27) ---

def test_risk_trajectory_consistency_handling():
    """24. Gate extracts multi-horizon risk trajectory and can enforce escalation constraint."""
    # Synthetic horizon estimates for k=1, 3, 5
    summary = StateChangeSummary(
        total_l1_drift=1.5,
        total_l2_drift=0.8,
        group_drifts={"continuous": 0.3},
        max_drift_feature="macro_mean_byte_rate",
        max_drift_value=0.5,
    )
    h1 = HorizonRiskEstimate(1, 10.0, 0.40, 0.20, 0.85, summary)
    h3 = HorizonRiskEstimate(3, 30.0, 0.65, 0.40, 0.85, summary)
    h5 = HorizonRiskEstimate(5, 50.0, 0.85, 0.60, 0.85, summary)

    escalating_traj = RiskTrajectory(
        aggregate_risk=0.85,
        aggregate_infiltration=0.60,
        horizon_estimates={1: h1, 3: h3, 5: h5},
        peak_horizon=5,
        is_escalating=True,
    )

    policy = TriggerPolicy(require_trajectory_escalation_for_borderline=True, risk_auto_threshold=0.80)
    gate = PredictiveTriggerGate(policy)

    decision = gate.evaluate(
        risk_score=escalating_traj,
        confidence=0.85,
        evidence_strength=0.75,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
    )
    assert decision.decision == TriggerDecisionType.AUTO_TRIGGER
    assert decision.risk_trajectory_summary is not None
    assert decision.risk_trajectory_summary["is_escalating"] is True
    assert any("Multi-horizon trajectory confirms escalating" in r for r in decision.reasons)


def test_json_serialization(default_gate: PredictiveTriggerGate):
    """25. TriggerDecision must be 100% JSON-serializable via to_dict()."""
    decision = default_gate.evaluate(
        risk_score=0.85,
        confidence=0.80,
        evidence_strength=0.70,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
        metadata={"incident_id": "INC-001", "ip": "10.0.0.1"},
    )
    d_dict = decision.to_dict()
    serialized = json.dumps(d_dict)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["decision"] == "AUTO_TRIGGER"
    assert deserialized["metadata"]["incident_id"] == "INC-001"


def test_phase_1_output_compatibility(default_gate: PredictiveTriggerGate):
    """26. Seamlessly ingest Phase 1 HorizonRiskEstimate, StageMappingResult, and ForecastExplanation."""
    state_change = StateChangeSummary(
        total_l1_drift=2.4,
        total_l2_drift=1.1,
        group_drifts={"macro": 0.8},
        max_drift_feature="macro_mean_byte_rate",
        max_drift_value=0.7,
    )
    horizon_est = HorizonRiskEstimate(
        horizon=3,
        delta_t=15.0,
        risk_score=0.86,
        infiltration_score=0.55,
        confidence=0.88,
        state_change=state_change,
    )
    stage_res = StageMappingResult(
        stage=AttackStage.IMPACT,
        confidence=0.88,
        supporting_features=[{"feature": "macro_mean_byte_rate", "delta": 0.7}],
        supporting_groups=["macro"],
        evidence={"volumetric_surge": 0.7},
        horizon=3,
        mitre_technique=None,
    )
    explanation = ForecastExplanation(
        explanation_method="State Delta Attribution",
        horizon=3,
        total_drift_magnitude=2.4,
        top_features=[
            FeatureChange(
                feature_name="macro_mean_byte_rate",
                group="macro",
                current_value=1.0,
                predicted_value=1.7,
                delta=0.7,
                magnitude=0.7,
                direction="INCREASE",
                relative_share=0.50,
            )
        ],
        group_attributions=[
            GroupAttribution(
                group_name="macro",
                absolute_delta_sum=1.2,
                relative_share=0.50,
                dominant_features=["macro_mean_byte_rate"],
            )
        ],
        summary_text="Macro surge detected",
    )

    # Evaluate passing Phase 1 objects directly
    decision = default_gate.evaluate(
        risk_score=horizon_est,
        predicted_stage=stage_res,
        explanation=explanation,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
    )
    assert decision.decision == TriggerDecisionType.AUTO_TRIGGER
    assert decision.risk_score == 0.86
    assert decision.confidence == 0.88
    assert decision.evidence_strength > 0.60
    assert decision.predicted_stage == AttackStage.IMPACT


def test_no_mutation_of_input_objects(default_gate: PredictiveTriggerGate):
    """27. Gate evaluation must NOT mutate passed input metadata dictionaries."""
    input_meta = {"incident_id": "TEST-123", "tags": ["prod"]}
    input_meta_copy = dict(input_meta)

    decision = default_gate.evaluate(
        risk_score=0.80,
        confidence=0.80,
        evidence_strength=0.70,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
        metadata=input_meta,
    )
    # Modify returned metadata to verify isolation
    decision.metadata["new_key"] = 999

    assert input_meta == input_meta_copy
    assert "new_key" not in input_meta


def test_offline_execution(default_gate: PredictiveTriggerGate):
    """28. Execution must be pure Python/NumPy logic without network or remote I/O."""
    import time
    start = time.perf_counter()
    for _ in range(100):
        default_gate.evaluate(
            risk_score=0.85,
            confidence=0.80,
            evidence_strength=0.75,
            asset_criticality=AssetCriticality.LOW,
            action_safety=ActionSafety.SAFE,
            predicted_stage=AttackStage.IMPACT,
        )
    elapsed = time.perf_counter() - start
    # 100 evaluations should easily take < 0.05 seconds
    assert elapsed < 0.10, f"Evaluation too slow for pure offline logic: {elapsed:.4f}s"
