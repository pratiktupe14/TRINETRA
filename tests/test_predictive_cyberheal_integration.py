"""Integration tests for TRINETRA Predictive Intelligence → CyberHeal Response Engine (Phase 3).

Verifies the integration boundary:
A. AUTO_TRIGGER creates valid predictive AgentState with source TRINETRA_PREDICTIVE.
B. HUMAN_REVIEW creates HumanReviewPayload without initiating remediation.
C. NO_TRIGGER suppresses response (no incident, no execution).
D. GUARDIAN PRESERVATION: AUTO_TRIGGER does not bypass Guardian safety checks.
E. PREDICTIVE VS REACTIVE: clear distinction between reactive alerts and predictive events.
F. BACKWARD COMPATIBILITY: existing reactive AgentState and workflows work unchanged.
G. EVIDENCE PRESERVATION: predictive risk, confidence, stage, features, trajectory preserved.
H. INVALID INPUT: malformed TriggerDecision fails safely to REJECTED_INVALID.
I. DETERMINISM: identical inputs produce identical integration outputs.
J. NO DIRECT EXECUTION: adapter does not execute OS firewall or remediation commands.
K. SERIALIZATION: complete JSON serializability of handoff results.
L. TRACEABILITY: incident IDs and trace IDs remain strictly linked.
M. REAL GRAPH EXECUTION: full ShieldLangGraph workflow execution in safe simulated mode.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from shield_rca.langgraph_core import AgentState, ShieldLangGraph
from shield_rca.predictive_integration import (
    HumanReviewPayload,
    IncidentType,
    PredictiveHandoffStatus,
    PredictiveIncidentAdapter,
    PredictiveIntegrationResult,
)
from shield_rca.world_model.explainability import FeatureChange, ForecastExplanation, GroupAttribution
from shield_rca.world_model.stage_mapper import AttackStage
from shield_rca.world_model.trigger_gate import (
    ActionSafety,
    AssetCriticality,
    TriggerDecision,
    TriggerDecisionType,
)


@pytest.fixture
def adapter() -> PredictiveIncidentAdapter:
    """Fixture providing a configured PredictiveIncidentAdapter."""
    return PredictiveIncidentAdapter(default_target_ip="192.168.1.100")


@pytest.fixture
def sample_explanation() -> ForecastExplanation:
    """Fixture providing a sample ForecastExplanation."""
    return ForecastExplanation(
        explanation_method="State Delta Attribution",
        horizon=3,
        total_drift_magnitude=2.8,
        top_features=[
            FeatureChange(
                feature_name="macro_mean_byte_rate",
                group="macro",
                current_value=100.0,
                predicted_value=500.0,
                delta=400.0,
                magnitude=400.0,
                direction="INCREASE",
                relative_share=0.45,
            ),
            FeatureChange(
                feature_name="flags_syn_count",
                group="flags",
                current_value=10.0,
                predicted_value=150.0,
                delta=140.0,
                magnitude=140.0,
                direction="INCREASE",
                relative_share=0.35,
            ),
        ],
        group_attributions=[
            GroupAttribution("macro", 400.0, 0.45, ["macro_mean_byte_rate"]),
            GroupAttribution("flags", 140.0, 0.35, ["flags_syn_count"]),
        ],
        summary_text="Severe macro volumetric surge and SYN flood detected.",
    )


@pytest.fixture
def auto_trigger_decision() -> TriggerDecision:
    """Fixture providing an authorized AUTO_TRIGGER TriggerDecision."""
    return TriggerDecision(
        decision=TriggerDecisionType.AUTO_TRIGGER,
        risk_score=0.88,
        confidence=0.85,
        evidence_strength=0.80,
        asset_criticality=AssetCriticality.MEDIUM,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.IMPACT,
        reasons=[
            "Risk score (0.880) meets or exceeds auto threshold (0.750)",
            "Confidence (0.850) meets or exceeds auto threshold (0.750)",
            "Action safety (SAFE) is approved for automated remediation",
        ],
        blocking_reasons=[],
        policy_version="1.0.0",
        risk_trajectory_summary={
            "aggregate_risk": 0.88,
            "aggregate_infiltration": 0.65,
            "peak_horizon": 5,
            "is_escalating": True,
            "horizons": [1, 3, 5],
        },
    )


@pytest.fixture
def human_review_decision() -> TriggerDecision:
    """Fixture providing a HUMAN_REVIEW TriggerDecision."""
    return TriggerDecision(
        decision=TriggerDecisionType.HUMAN_REVIEW,
        risk_score=0.85,
        confidence=0.50,  # Below auto threshold
        evidence_strength=0.45,
        asset_criticality=AssetCriticality.HIGH,
        action_safety=ActionSafety.RESTRICTED,
        predicted_stage=AttackStage.UNKNOWN,
        reasons=["Predictive risk score (0.850) meets or exceeds review threshold (0.350)"],
        blocking_reasons=[
            "Confidence (0.500) below auto threshold (0.750)",
            "Asset criticality (HIGH) exceeds auto threshold (MEDIUM)",
            "Predicted attack stage is UNKNOWN",
        ],
        policy_version="1.0.0",
    )


@pytest.fixture
def no_trigger_decision() -> TriggerDecision:
    """Fixture providing a NO_TRIGGER TriggerDecision."""
    return TriggerDecision(
        decision=TriggerDecisionType.NO_TRIGGER,
        risk_score=0.15,
        confidence=0.90,
        evidence_strength=0.10,
        asset_criticality=AssetCriticality.LOW,
        action_safety=ActionSafety.SAFE,
        predicted_stage=AttackStage.NORMAL,
        reasons=["Risk score (0.150) is below review threshold (0.350); response suppressed"],
        blocking_reasons=["Risk score below review threshold"],
        policy_version="1.0.0",
    )


# --- Test Cases ---

def test_auto_trigger_creates_predictive_agent_state(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
    sample_explanation: ForecastExplanation,
):
    """A. AUTO_TRIGGER creates a fully-populated predictive AgentState."""
    result = adapter.process_trigger_decision(
        decision=auto_trigger_decision,
        target_ip="10.0.0.42",
        incident_id="INC-PRED-TEST-001",
        explanation=sample_explanation,
    )

    assert result.status == PredictiveHandoffStatus.AUTO_TRIGGERED
    assert result.incident_id == "INC-PRED-TEST-001"
    assert result.trace_id == "TRACE-INC-PRED-TEST-001"
    assert result.agent_state is not None
    assert result.review_payload is None

    state = result.agent_state
    assert state.incident_type == "PREDICTIVE"
    assert state.alert_payload["source"] == "TRINETRA_PREDICTIVE"
    assert state.alert_payload["trigger_type"] == "PREDICTIVE"
    assert state.alert_payload["ip"] == "10.0.0.42"
    assert state.trajectory_risk_score == 0.88
    assert state.attack_stage == "IMPACT"
    assert state.confidence_score == 0.85
    assert len(state.top_predictive_features) == 2
    assert state.top_predictive_features[0]["feature"] == "macro_mean_byte_rate"
    assert state.risk_trajectory is not None
    assert state.risk_trajectory["is_escalating"] is True


def test_human_review_creates_payload_without_remediation(
    adapter: PredictiveIncidentAdapter,
    human_review_decision: TriggerDecision,
    sample_explanation: ForecastExplanation,
):
    """B. HUMAN_REVIEW creates an escalation review bundle and suppresses remediation execution."""
    result = adapter.process_trigger_decision(
        decision=human_review_decision,
        target_ip="10.0.0.99",
        incident_id="INC-REVIEW-001",
        explanation=sample_explanation,
    )

    assert result.status == PredictiveHandoffStatus.ESCALATED_HUMAN_REVIEW
    assert result.incident_id == "INC-REVIEW-001"
    assert result.agent_state is None  # No automated execution initiated
    assert result.review_payload is not None

    payload = result.review_payload
    assert payload.source == "TRINETRA_PREDICTIVE"
    assert payload.trigger_decision == "HUMAN_REVIEW"
    assert payload.risk_score == 0.85
    assert payload.confidence == 0.50
    assert payload.target_ip == "10.0.0.99"
    assert payload.predicted_stage == "UNKNOWN"
    assert len(payload.blocking_reasons) == 3
    assert payload.explanation_summary == sample_explanation.summary_text


def test_no_trigger_suppresses_response(
    adapter: PredictiveIncidentAdapter,
    no_trigger_decision: TriggerDecision,
):
    """C. NO_TRIGGER suppresses response; no incident or state is created."""
    result = adapter.process_trigger_decision(
        decision=no_trigger_decision,
    )

    assert result.status == PredictiveHandoffStatus.SUPPRESSED
    assert result.incident_id is None
    assert result.agent_state is None
    assert result.review_payload is None
    assert "suppressed" in result.reason.lower()


def test_guardian_safety_gate_preservation(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
):
    """D. AUTO_TRIGGER does not bypass Guardian; Guardian evaluates the plan normally."""
    result = adapter.process_trigger_decision(
        decision=auto_trigger_decision,
        incident_id="INC-PRED-GUARDIAN",
    )
    state = result.agent_state
    assert state is not None

    # Graph agents
    graph = ShieldLangGraph(db_client=None)

    # 1. Intake normalizes
    state = graph.intake.run(state)
    assert state.normalized_alert is not None
    assert state.normalized_alert["source"] == "TRINETRA_PREDICTIVE"

    # 2. Sentinel extracts anomaly
    state = graph.sentinel.run(state)
    assert len(state.anomalies) > 0
    assert state.anomalies[0]["metric"] == "forecasted_network_drift"

    # 3. Causor RCA
    state = graph.causor.run(state)
    assert "IMPACT" in state.root_cause["description"]

    # 4. Planner generates plan
    state = graph.planner.run(state)
    assert len(state.remediation_plan) > 0

    # 5. Guardian inspects confidence
    # High confidence (0.85) -> Guardian approves
    state = graph.guardian.run(state)
    assert state.guardian_approved is True

    # Now verify Guardian REJECTS if confidence is below threshold (< 0.70)
    low_conf_state = state.model_copy(deep=True) if hasattr(state, "model_copy") else state.copy(deep=True)
    low_conf_state.confidence_score = 0.50
    low_conf_state = graph.guardian.run(low_conf_state)
    assert low_conf_state.guardian_approved is False
    assert low_conf_state.risk_score == "HIGH"


def test_predictive_vs_reactive_distinction(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
):
    """E. Predictive events are clearly distinguished from legacy reactive alerts."""
    # 1. Predictive state
    pred_res = adapter.process_trigger_decision(auto_trigger_decision)
    pred_state = pred_res.agent_state
    assert pred_state.incident_type == IncidentType.PREDICTIVE.value
    assert pred_state.alert_payload["trigger_type"] == "PREDICTIVE"

    # 2. Reactive state
    react_state = AgentState(
        incident_id="INC-REACT-001",
        alert_payload={"source": "Suricata_IDS", "ip": "192.168.1.50"},
    )
    assert react_state.incident_type == IncidentType.REACTIVE.value
    assert react_state.alert_payload.get("trigger_type") is None


def test_backward_compatibility_with_reactive_workflow():
    """F. Legacy reactive AgentState without predictive fields runs unchanged."""
    legacy_state = AgentState(
        incident_id="INC-LEGACY-001",
        alert_payload={"source": "snort", "metric": "ddos_syn", "severity": "HIGH", "ip": "1.2.3.4"},
    )
    graph = ShieldLangGraph(db_client=None)

    # Run step-by-step
    s1 = graph.intake.run(legacy_state)
    assert s1.normalized_alert["source"] == "snort"

    s2 = graph.sentinel.run(s1)
    assert s2.anomalies[0]["id"] == "ANOM-1"

    s3 = graph.causor.run(s2)
    assert s3.confidence_score >= 0.85

    s4 = graph.planner.run(s3)
    assert len(s4.remediation_plan) > 0

    s5 = graph.guardian.run(s4)
    assert s5.guardian_approved is True


def test_predictive_evidence_preservation(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
    sample_explanation: ForecastExplanation,
):
    """G. All TRINETRA evidence fields survive conversion into AgentState and dictionary representations."""
    forecast_vec = [0.1 * i for i in range(280)]
    result = adapter.process_trigger_decision(
        decision=auto_trigger_decision,
        target_ip="172.16.0.5",
        incident_id="INC-EVIDENCE-CHECK",
        explanation=sample_explanation,
        forecasted_state=forecast_vec,
    )
    state = result.agent_state
    assert state.incident_id == "INC-EVIDENCE-CHECK"
    assert state.forecasted_state == forecast_vec
    assert state.trajectory_risk_score == 0.88
    assert state.attack_stage == "IMPACT"
    assert state.asset_criticality == "MEDIUM"
    assert state.action_safety == "SAFE"
    assert len(state.predictive_reasons) > 0
    assert state.trigger_decision["risk_score"] == 0.88


def test_invalid_input_fails_safely(adapter: PredictiveIncidentAdapter):
    """H. Malformed TriggerDecision fails safely to REJECTED_INVALID."""
    # 1. Non-dict, non-decision object
    r1 = adapter.process_trigger_decision("INVALID_STRING")
    assert r1.status == PredictiveHandoffStatus.REJECTED_INVALID
    assert "Expected TriggerDecision or dict" in r1.reason

    # 2. Missing required fields in dict
    r2 = adapter.process_trigger_decision({"decision": "AUTO_TRIGGER"})
    assert r2.status == PredictiveHandoffStatus.REJECTED_INVALID
    assert "Missing required field" in r2.reason

    # 3. Out of bounds numerical field
    r3 = adapter.process_trigger_decision({
        "decision": "AUTO_TRIGGER",
        "risk_score": 1.5,  # Invalid
        "evidence_strength": 0.5,
        "asset_criticality": "LOW",
        "action_safety": "SAFE",
        "predicted_stage": "IMPACT",
    })
    assert r3.status == PredictiveHandoffStatus.REJECTED_INVALID
    assert "risk_score" in r3.reason


def test_determinism_and_immutability(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
):
    """I. Identical inputs yield identical outputs and caller objects are not mutated."""
    meta = {"cluster": "us-east-1", "nodes": 4}
    meta_copy = dict(meta)

    r1 = adapter.process_trigger_decision(auto_trigger_decision, incident_id="INC-FIXED-01", metadata=meta)
    r2 = adapter.process_trigger_decision(auto_trigger_decision, incident_id="INC-FIXED-01", metadata=meta)

    assert r1.to_dict()["agent_state"] == r2.to_dict()["agent_state"]
    assert meta == meta_copy  # Immutability


def test_no_direct_remediation_execution(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
    monkeypatch: pytest.MonkeyPatch,
):
    """J. PredictiveIncidentAdapter must NOT execute shell or firewall commands."""
    import subprocess

    def mock_subprocess_run(*args, **kwargs):
        raise AssertionError("Subprocess run called by integration adapter!")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    # Must process without invoking subprocess
    result = adapter.process_trigger_decision(auto_trigger_decision)
    assert result.status == PredictiveHandoffStatus.AUTO_TRIGGERED


def test_complete_json_serialization(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
    human_review_decision: TriggerDecision,
    sample_explanation: ForecastExplanation,
):
    """K. Both AUTO_TRIGGER and HUMAN_REVIEW results are 100% JSON serializable."""
    res_auto = adapter.process_trigger_decision(auto_trigger_decision, explanation=sample_explanation)
    res_rev = adapter.process_trigger_decision(human_review_decision, explanation=sample_explanation)

    json_auto = json.dumps(res_auto.to_dict())
    json_rev = json.dumps(res_rev.to_dict())

    assert isinstance(json_auto, str)
    assert isinstance(json_rev, str)
    assert json.loads(json_auto)["status"] == "AUTO_TRIGGERED"
    assert json.loads(json_rev)["status"] == "ESCALATED_HUMAN_REVIEW"


def test_traceability_linkage(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
):
    """L. Event, incident, and trace IDs maintain strict traceability."""
    result = adapter.process_trigger_decision(auto_trigger_decision, incident_id="INC-PRED-999")
    assert result.incident_id == "INC-PRED-999"
    assert result.trace_id == "TRACE-INC-PRED-999"
    assert result.agent_state.alert_payload["trace_id"] == "TRACE-INC-PRED-999"


@pytest.mark.anyio
async def test_real_langgraph_workflow_routing(
    adapter: PredictiveIncidentAdapter,
    auto_trigger_decision: TriggerDecision,
    sample_explanation: ForecastExplanation,
    monkeypatch: pytest.MonkeyPatch,
):
    """M. Real ShieldLangGraph workflow processes predictive AgentState to final resolution."""
    async def mock_sleep(_duration: float):
        return  # No-op to speed up test execution

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    result = adapter.process_trigger_decision(
        decision=auto_trigger_decision,
        target_ip="192.168.1.200",
        incident_id="INC-GRAPH-ROUTING",
        explanation=sample_explanation,
    )
    initial_state = result.agent_state
    assert initial_state is not None

    graph = ShieldLangGraph(db_client=None)
    final_state = await graph.execute_workflow(initial_state)

    # Verification passes on attempt #1 in CyberHeal logic
    assert final_state.current_step == "COMPLETED"
    assert final_state.verification_passed is True
    assert final_state.status == "RESOLVED"
    assert len(final_state.execution_results) > 0
