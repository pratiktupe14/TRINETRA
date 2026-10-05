"""Unit tests for Post-Remediation Verification Engine (Phase 4).

Verifies:
1. Threat reduction ratio (TRR) and residual risk calculation.
2. VERIFIED outcome when threat is significantly extinguished.
3. NOT_VERIFIED outcome when threat remains high (triggers RLM).
4. INCONCLUSIVE outcome on empty, missing, or non-finite metric inputs.
5. Verification from raw 280-dimensional Network State vectors.
6. Realistic post-remediation telemetry simulation helper.
7. VerificationReport JSON serialization.
8. Custom verification policy enforcement.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from shield_rca.network_state.state_schema import STATE_FEATURE_DIM
from shield_rca.post_verification import (
    PostRemediationVerifier,
    VerificationPolicy,
    VerificationReport,
    VerificationStatus,
)


@pytest.fixture
def default_verifier() -> PostRemediationVerifier:
    return PostRemediationVerifier()


def test_successful_verification(default_verifier: PostRemediationVerifier):
    """1. Significant threat reduction with low residual risk yields VERIFIED."""
    before_metrics = {
        "risk_score": 0.85,
        "anomalous_pps": 10000.0,
        "syn_flood_ratio": 0.75,
    }
    after_metrics = {
        "risk_score": 0.15,
        "anomalous_pps": 200.0,
        "syn_flood_ratio": 0.05,
    }

    report = default_verifier.verify(
        incident_id="INC-VERIFY-001",
        before_metrics=before_metrics,
        after_metrics=after_metrics,
    )

    assert report.status == VerificationStatus.VERIFIED
    assert report.verification_passed is True
    assert report.threat_reduction_ratio >= 0.80
    assert report.residual_risk == 0.15
    assert report.recommended_action == "CLOSE_INCIDENT"
    assert len(report.verification_reasons) >= 2


def test_failed_verification_triggers_rlm(default_verifier: PostRemediationVerifier):
    """2. Incomplete threat mitigation yields NOT_VERIFIED and recommends TRIGGER_RLM."""
    before_metrics = {
        "risk_score": 0.90,
        "anomalous_pps": 15000.0,
    }
    after_metrics = {
        "risk_score": 0.70,  # Still high, reduction is only ~22%
        "anomalous_pps": 12000.0,
    }

    report = default_verifier.verify(
        incident_id="INC-VERIFY-002",
        before_metrics=before_metrics,
        after_metrics=after_metrics,
    )

    assert report.status == VerificationStatus.NOT_VERIFIED
    assert report.verification_passed is False
    assert report.residual_risk == 0.70
    assert report.recommended_action == "TRIGGER_RLM"
    assert any("below required threshold" in r for r in report.verification_reasons)


def test_inconclusive_verification_on_invalid_inputs(default_verifier: PostRemediationVerifier):
    """3. Empty or non-finite inputs yield safe INCONCLUSIVE report without raising uncaught exceptions."""
    # Empty metrics
    r_empty = default_verifier.verify("INC-ERR-1", {}, {})
    assert r_empty.status == VerificationStatus.INCONCLUSIVE
    assert r_empty.verification_passed is False
    assert r_empty.recommended_action == "ESCALATE_HUMAN"

    # NaN input
    r_nan = default_verifier.verify(
        "INC-ERR-2",
        {"risk_score": float("nan")},
        {"risk_score": 0.2},
    )
    assert r_nan.status == VerificationStatus.INCONCLUSIVE
    assert r_nan.verification_passed is False


def test_verification_from_state_vectors(default_verifier: PostRemediationVerifier):
    """4. Verifies remediation directly from 280-dimensional Network State vectors."""
    state_before = np.ones(STATE_FEATURE_DIM, dtype=np.float32) * 2.0   # Large drift
    state_after = np.ones(STATE_FEATURE_DIM, dtype=np.float32) * 0.02   # Quiescent nominal (<0.35 risk)

    report = default_verifier.verify_from_state_vectors(
        incident_id="INC-VEC-001",
        state_before=state_before,
        state_after=state_after,
    )
    assert report.status == VerificationStatus.VERIFIED
    assert report.verification_passed is True
    assert report.residual_risk < 0.35


def test_telemetry_simulation_helper():
    """5. Helper simulates realistic before-after reductions for tests and offline demos."""
    before = {"risk_score": 0.80, "pps": 5000.0}

    # Successful simulation
    after_success = PostRemediationVerifier.simulate_post_remediation_telemetry(
        before, remediation_successful=True, reduction_factor=0.85
    )
    assert after_success["risk_score"] < 0.20
    assert after_success["pps"] < 1000.0

    # Failed simulation
    after_fail = PostRemediationVerifier.simulate_post_remediation_telemetry(
        before, remediation_successful=False
    )
    assert after_fail["risk_score"] >= 0.70


def test_verification_report_serialization(default_verifier: PostRemediationVerifier):
    """6. VerificationReport is 100% JSON-serializable."""
    report = default_verifier.verify(
        incident_id="INC-JSON-VERIFY",
        before_metrics={"risk_score": 0.85},
        after_metrics={"risk_score": 0.10},
    )
    serialized = json.dumps(report.to_dict())
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["status"] == "VERIFIED"
    assert deserialized["incident_id"] == "INC-JSON-VERIFY"


def test_custom_verification_policy():
    """7. Custom strict policy thresholds are enforced."""
    strict_policy = VerificationPolicy(
        minimum_threat_reduction_ratio=0.90,  # Requires 90% drop
        maximum_residual_risk=0.10,           # Residual risk must drop <= 0.10
    )
    verifier = PostRemediationVerifier(strict_policy)

    before = {"risk_score": 0.80}
    after = {"risk_score": 0.25}  # 68% drop: passes default (50%), fails strict (90%)

    report = verifier.verify("INC-STRICT", before, after)
    assert report.status == VerificationStatus.NOT_VERIFIED
    assert report.verification_passed is False
