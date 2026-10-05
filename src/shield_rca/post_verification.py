"""Post-Remediation Verification Engine for CyberHeal AI / TRINETRA.

Replaces heuristic verification with observable Before-vs-After Telemetry Comparison:
1. Mathematical Threat Reduction Ratio (TRR):
       TRR = (Threat_before - Threat_after) / (Threat_before + epsilon)
2. Residual Risk Assessment:
       ResidualRisk in [0.0, 1.0]
3. Clear Verification States:
       - VERIFIED: TRR >= threshold AND ResidualRisk <= threshold (Mitigation successful).
       - NOT_VERIFIED: Threat remains elevated (Triggers RLM retry or escalation).
       - INCONCLUSIVE: Missing, contradictory, or ambiguous telemetry.
4. Structured VerificationReport with diagnostic reasons and recommended action.
5. Safe Simulation Helper for test and demo environments.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .network_state.state_schema import STATE_FEATURE_DIM

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Categorical outcome of post-remediation verification."""

    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Configurable thresholds governing post-remediation verification."""

    minimum_threat_reduction_ratio: float = 0.50  # Must reduce threat by at least 50%
    maximum_residual_risk: float = 0.35           # Residual risk must drop below 0.35
    epsilon: float = 1e-6                         # Numerical stability constant

    def __post_init__(self) -> None:
        """Validate verification policy thresholds."""
        for name in ("minimum_threat_reduction_ratio", "maximum_residual_risk"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Verification policy threshold {name} must be a finite float")
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Verification policy threshold {name} must be in [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Detailed audit report of post-remediation system state and verification status."""

    incident_id: str
    status: VerificationStatus
    verification_passed: bool
    threat_reduction_ratio: float
    residual_risk: float
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]
    verification_reasons: list[str]
    recommended_action: str  # "CLOSE_INCIDENT", "TRIGGER_RLM", "ESCALATE_HUMAN"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert verification report to JSON-serializable dictionary."""
        return {
            "incident_id": self.incident_id,
            "status": self.status.value,
            "verification_passed": self.verification_passed,
            "threat_reduction_ratio": float(self.threat_reduction_ratio),
            "residual_risk": float(self.residual_risk),
            "before_metrics": {k: float(v) for k, v in self.before_metrics.items()},
            "after_metrics": {k: float(v) for k, v in self.after_metrics.items()},
            "verification_reasons": list(self.verification_reasons),
            "recommended_action": self.recommended_action,
            "timestamp": float(self.timestamp),
        }


class PostRemediationVerifier:
    """Compares pre-remediation and post-remediation telemetry to evaluate fix efficacy."""

    def __init__(self, policy: VerificationPolicy | None = None) -> None:
        """Initialize the verifier with specified or default verification policy."""
        self.policy = policy if policy is not None else VerificationPolicy()

    def verify(
        self,
        incident_id: str,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> VerificationReport:
        """Verify remediation success by comparing before and after metric states.

        Args:
            incident_id: Unique incident tracking identifier.
            before_metrics: Dictionary of metric values prior to remediation.
                            Must include 'risk_score' or primary threat metric.
            after_metrics: Dictionary of metric values observed post-remediation.

        Returns:
            VerificationReport containing decision, reduction ratio, and recommendations.
        """
        # Validate inputs
        if not before_metrics or not after_metrics:
            return self._build_inconclusive_report(
                incident_id, before_metrics, after_metrics, "Empty or missing metric dictionaries"
            )

        # Validate finite floats
        for label, m_dict in (("before", before_metrics), ("after", after_metrics)):
            for k, v in m_dict.items():
                if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    return self._build_inconclusive_report(
                        incident_id, before_metrics, after_metrics, f"Non-finite value in {label} metrics: {k}={v}"
                    )

        # Extract primary risk scores
        risk_before = float(before_metrics.get("risk_score", 0.0))
        risk_after = float(after_metrics.get("risk_score", 0.0))

        # Clamp risks to [0.0, 1.0]
        risk_before = max(0.0, min(1.0, risk_before))
        risk_after = max(0.0, min(1.0, risk_after))

        # Compute Threat Reduction Ratio (TRR)
        denom = risk_before + self.policy.epsilon
        threat_reduction = (risk_before - risk_after) / denom

        reasons: list[str] = []
        status: VerificationStatus
        passed: bool
        rec_action: str

        # Evaluate against policy thresholds
        meets_reduction = threat_reduction >= self.policy.minimum_threat_reduction_ratio
        meets_residual = risk_after <= self.policy.maximum_residual_risk

        if meets_reduction and meets_residual:
            status = VerificationStatus.VERIFIED
            passed = True
            rec_action = "CLOSE_INCIDENT"
            reasons.append(
                f"Threat reduction ratio ({threat_reduction:.1%}) meets threshold ({self.policy.minimum_threat_reduction_ratio:.1%})"
            )
            reasons.append(
                f"Residual risk ({risk_after:.2f}) is safely below maximum ({self.policy.maximum_residual_risk:.2f})"
            )
        else:
            status = VerificationStatus.NOT_VERIFIED
            passed = False
            rec_action = "TRIGGER_RLM"
            if not meets_reduction:
                reasons.append(
                    f"Threat reduction ratio ({threat_reduction:.1%}) below required threshold ({self.policy.minimum_threat_reduction_ratio:.1%})"
                )
            if not meets_residual:
                reasons.append(
                    f"Residual risk ({risk_after:.2f}) remains elevated above acceptable limit ({self.policy.maximum_residual_risk:.2f})"
                )

        # Log metric comparison
        for k in before_metrics:
            if k in after_metrics and k != "risk_score":
                val_b = before_metrics[k]
                val_a = after_metrics[k]
                pct_change = ((val_b - val_a) / (val_b + 1e-6)) * 100.0
                reasons.append(f"Metric '{k}': {val_b:.1f} -> {val_a:.1f} ({pct_change:+.1f}%)")

        return VerificationReport(
            incident_id=incident_id,
            status=status,
            verification_passed=passed,
            threat_reduction_ratio=threat_reduction,
            residual_risk=risk_after,
            before_metrics=dict(before_metrics),
            after_metrics=dict(after_metrics),
            verification_reasons=reasons,
            recommended_action=rec_action,
        )

    def verify_from_state_vectors(
        self,
        incident_id: str,
        state_before: np.ndarray,
        state_after: np.ndarray,
        nominal_state: np.ndarray | None = None,
    ) -> VerificationReport:
        """Verify remediation from raw 280-dimensional Network State vectors."""
        if state_before.shape != (STATE_FEATURE_DIM,) or state_after.shape != (STATE_FEATURE_DIM,):
            return self._build_inconclusive_report(
                incident_id, {}, {}, "State vectors do not match STATE_FEATURE_DIM (280)"
            )

        ref = nominal_state if nominal_state is not None else np.zeros(STATE_FEATURE_DIM)
        drift_before = float(np.linalg.norm(state_before - ref))
        drift_after = float(np.linalg.norm(state_after - ref))

        # Risk saturation proxy from L2 drift
        risk_before = 1.0 - math.exp(-0.5 * drift_before)
        risk_after = 1.0 - math.exp(-0.5 * drift_after)

        before_m = {"risk_score": risk_before, "l2_drift": drift_before}
        after_m = {"risk_score": risk_after, "l2_drift": drift_after}

        return self.verify(incident_id, before_m, after_m)

    @staticmethod
    def simulate_post_remediation_telemetry(
        before_metrics: dict[str, float],
        remediation_successful: bool = True,
        reduction_factor: float = 0.80,
    ) -> dict[str, float]:
        """Simulate realistic post-remediation telemetry for testing and offline demonstrations.

        Args:
            before_metrics: Starting metrics before fix application.
            remediation_successful: If True, reduces threat metrics significantly.
                                    If False, metrics stay elevated or worsen.
            reduction_factor: Fraction of threat to extinguish when successful (default 80%).
        """
        after: dict[str, float] = {}
        factor = max(0.0, min(1.0, float(reduction_factor))) if remediation_successful else 0.10

        for k, v in before_metrics.items():
            if k == "risk_score":
                after[k] = max(0.05, float(v) * (1.0 - factor))
            else:
                after[k] = max(0.0, float(v) * (1.0 - factor))

        return after

    @staticmethod
    def _build_inconclusive_report(
        incident_id: str,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        reason: str,
    ) -> VerificationReport:
        """Build a safe INCONCLUSIVE report upon encountering invalid inputs."""
        return VerificationReport(
            incident_id=incident_id,
            status=VerificationStatus.INCONCLUSIVE,
            verification_passed=False,
            threat_reduction_ratio=0.0,
            residual_risk=1.0,
            before_metrics=dict(before_metrics),
            after_metrics=dict(after_metrics),
            verification_reasons=[f"Verification inconclusive: {reason}"],
            recommended_action="ESCALATE_HUMAN",
        )
