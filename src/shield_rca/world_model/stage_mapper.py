"""Attack Stage Progression Mapper for CyberHeal AI / TRINETRA.

Interprets predicted network state transitions in terms of defensive attack stages
aligned with the MITRE ATT&CK framework and the canonical 15-class taxonomy in:
    src/shield_rca/models/label_maps.py

Strict Security Grounding & Anti-Fabrication Rules:
1. Exact MITRE ATT&CK technique IDs (e.g. T1046, T1110) are NEVER fabricated from generic
   macro-state vector drift. When telemetry indicates an attack stage, mitre_technique is
   explicitly set to None unless verified external ground-truth labels are supplied during
   offline evaluation.
2. If evidence is ambiguous, conflicting, or below detection thresholds, the mapper outputs
   AttackStage.NORMAL or AttackStage.UNKNOWN with honest confidence.
3. Unsupported Stages: Internal host-level stages (Execution, Persistence, Privilege Escalation)
   cannot be defensibly inferred from network traffic telemetry alone and are strictly excluded
   from runtime inference.

Supported Network-Observable Attack Stages:
- NORMAL: Quiescent baseline traffic without significant anomalous trajectory drift.
- RECONNAISSANCE: Surges in port dispersion, unique endpoints, and connection resets (PortScan/Discovery).
- INITIAL_ACCESS: Surges in brute-force authentication bursts and repetitive protocol connections.
- INFILTRATION: Asymmetric byte/payload drift, unusual protocol distributions, and C2/Botnet patterns.
- IMPACT: Massive volumetric surges, SYN floods, connection saturation (DDoS, DoS Hulk/GoldenEye).
- UNKNOWN: Measurable anomalous drift that does not cleanly match a specific behavioral stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..models.label_maps import INT_TO_LABEL, is_attack
from ..network_state.state_schema import (
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


class AttackStage(str, Enum):
    """Network-observable attack progression stages."""

    NORMAL = "NORMAL"
    RECONNAISSANCE = "RECONNAISSANCE"
    INITIAL_ACCESS = "INITIAL_ACCESS"
    INFILTRATION = "INFILTRATION"
    IMPACT = "IMPACT"
    UNKNOWN = "UNKNOWN"


# Explicitly documented unsupported stages that cannot be inferred from network flows alone:
UNSUPPORTED_STAGES: tuple[str, ...] = (
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access (Host-Internal)",
    "Collection (Local Disk)",
)


@dataclass(frozen=True, slots=True)
class StageMappingResult:
    """Structured result of predictive attack stage interpretation."""

    stage: AttackStage
    confidence: float | None  # Bounded [0.0, 1.0] or None if confidence cannot be defensibly calculated
    supporting_features: list[dict[str, Any]]
    supporting_groups: list[str]
    evidence: dict[str, Any]
    horizon: int
    mitre_technique: str | None = None  # None by default (never fabricated)


class AttackStageMapper:
    """Interprets future Network State trajectories into defensive attack progression stages."""

    def __init__(
        self,
        normal_drift_threshold: float = 0.05,
        recon_port_dispersion_threshold: float = 0.20,
        impact_rate_surge_threshold: float = 0.50,
        syn_flood_ratio_threshold: float = 0.40,
    ) -> None:
        """Initialize stage mapper with configurable behavioral thresholds.

        Args:
            normal_drift_threshold: Maximum relative drift to consider state as NORMAL.
            recon_port_dispersion_threshold: Delta threshold in port dispersion / unique ports for RECON.
            impact_rate_surge_threshold: Delta threshold in byte/packet rate surge for IMPACT.
            syn_flood_ratio_threshold: Delta threshold in syn_ack ratio surge for IMPACT.
        """
        self.normal_drift_threshold = float(normal_drift_threshold)
        self.recon_port_dispersion_threshold = float(recon_port_dispersion_threshold)
        self.impact_rate_surge_threshold = float(impact_rate_surge_threshold)
        self.syn_flood_ratio_threshold = float(syn_flood_ratio_threshold)

        # Precompute name-to-index mapping for fast O(1) state feature lookup
        self._name_to_idx = {name: idx for idx, name in enumerate(STATE_FEATURE_NAMES)}

    def _get_val(self, state: np.ndarray, feature_name: str) -> float:
        """Retrieve scalar feature value from 280-d state vector by name."""
        idx = self._name_to_idx.get(feature_name)
        if idx is None:
            return 0.0
        return float(state[idx])

    def map_stage(
        self,
        current_state: np.ndarray,
        predicted_state: np.ndarray,
        horizon: int = 1,
        risk_score: float | None = None,
        known_label: int | str | None = None,
    ) -> StageMappingResult:
        """Map predicted state change to an attack stage.

        Args:
            current_state: Current 280-d Network State S_t.
            predicted_state: Forecasted 280-d Network State S_hat_{t+k}.
            horizon: Future prediction step k >= 1.
            risk_score: Optional risk score in [0.0, 1.0] from PredictiveRiskEstimator.
            known_label: Optional label for offline evaluation only (never leaked in inference).

        Returns:
            StageMappingResult with stage, evidence, supporting features, and no fabricated MITRE ID.
        """
        if current_state is None or predicted_state is None:
            raise ValueError("current_state and predicted_state cannot be None")

        cur = np.asarray(current_state, dtype=np.float32)
        pred = np.asarray(predicted_state, dtype=np.float32)

        if cur.shape != (STATE_FEATURE_DIM,) or pred.shape != (STATE_FEATURE_DIM,):
            raise ValueError(
                f"State dimension mismatch: expected ({STATE_FEATURE_DIM},), "
                f"got cur={cur.shape}, pred={pred.shape}"
            )
        if not np.all(np.isfinite(cur)) or not np.all(np.isfinite(pred)):
            raise ValueError("Input state contains non-finite values (NaN or Inf)")

        diff = pred - cur
        abs_diff = np.abs(diff)
        mean_abs_drift = float(np.mean(abs_diff))

        # 1. Baseline / Normal Check
        if mean_abs_drift <= self.normal_drift_threshold:
            return StageMappingResult(
                stage=AttackStage.NORMAL,
                confidence=1.0 - (mean_abs_drift / max(self.normal_drift_threshold, 1e-6)),
                supporting_features=[],
                supporting_groups=[],
                evidence={"mean_abs_drift": mean_abs_drift, "reason": "State drift within quiescent baseline"},
                horizon=horizon,
                mitre_technique=None,
            )

        # 2. Extract key domain signature deltas from Network State
        d_port_dispersion = abs(self._get_val(pred, "macro_port_dispersion") - self._get_val(cur, "macro_port_dispersion"))
        d_unique_dst_ports = abs(self._get_val(pred, "endpoint_unique_dst_ports") - self._get_val(cur, "endpoint_unique_dst_ports"))
        d_rst_ratio = abs(self._get_val(pred, "macro_rst_ratio") - self._get_val(cur, "macro_rst_ratio"))
        d_syn_ratio = abs(self._get_val(pred, "macro_syn_ack_ratio") - self._get_val(cur, "macro_syn_ack_ratio"))
        d_mean_byte_rate = abs(self._get_val(pred, "macro_mean_byte_rate") - self._get_val(cur, "macro_mean_byte_rate"))
        d_mean_pkt_rate = abs(self._get_val(pred, "macro_mean_packet_rate") - self._get_val(cur, "macro_mean_packet_rate"))
        d_down_up_ratio = abs(self._get_val(pred, "macro_down_up_ratio") - self._get_val(cur, "macro_down_up_ratio"))

        # Signatures
        recon_score = 0.5 * d_port_dispersion + 0.3 * d_unique_dst_ports + 0.2 * d_rst_ratio
        impact_score = 0.4 * d_mean_byte_rate + 0.3 * d_mean_pkt_rate + 0.3 * d_syn_ratio
        infil_score = 0.5 * d_down_up_ratio + 0.3 * d_rst_ratio + 0.2 * abs(self._get_val(pred, "endpoint_last_protocol") - self._get_val(cur, "endpoint_last_protocol"))

        candidate_scores: dict[AttackStage, float] = {
            AttackStage.RECONNAISSANCE: recon_score,
            AttackStage.IMPACT: impact_score,
            AttackStage.INFILTRATION: infil_score,
        }

        # Find dominant candidate stage
        top_stage = max(candidate_scores, key=lambda s: candidate_scores[s])
        top_val = candidate_scores[top_stage]

        # Evidence collection
        supporting_features: list[dict[str, Any]] = []
        supporting_groups: list[str] = []

        # Find top 3 individual changing features
        top_indices = np.argsort(abs_diff)[-3:][::-1]
        for idx in top_indices:
            feat_name = STATE_FEATURE_NAMES[idx]
            supporting_features.append({
                "feature": feat_name,
                "current": float(cur[idx]),
                "predicted": float(pred[idx]),
                "delta": float(diff[idx]),
            })

        # Threshold check: if top score is negligible despite general drift, map to UNKNOWN
        if top_val < 0.10:
            stage = AttackStage.UNKNOWN
            conf = 0.50
            supporting_groups = ["general_drift"]
            reason = "State drifted significantly but signature was ambiguous or spread across multiple categories"
        else:
            stage = top_stage
            conf = min(0.95, float(0.50 + 0.50 * (top_val / (top_val + 1.0))))
            if stage == AttackStage.RECONNAISSANCE:
                supporting_groups = ["endpoints", "macro"]
                reason = "Surge in port dispersion, unique endpoints, and connection resets"
            elif stage == AttackStage.IMPACT:
                supporting_groups = ["macro", "flags", "continuous"]
                reason = "Spike in volumetric byte/packet rates and SYN flood ratio"
            elif stage == AttackStage.INFILTRATION:
                supporting_groups = ["macro", "endpoints"]
                reason = "Asymmetric down/up volume ratio shift and protocol endpoint deviation"
            else:
                supporting_groups = ["unknown"]
                reason = "Ambiguous signature"

        # Explicit MITRE technique assignment:
        # Rule: NEVER fabricate. mitre_technique is None unless known label was provided for offline benchmarking.
        mitre_tech = None
        if known_label is not None:
            lbl_name = INT_TO_LABEL.get(known_label, str(known_label)) if isinstance(known_label, int) else str(known_label)
            if is_attack(lbl_name):
                # Documented offline ground-truth alignment (offline verification only)
                mitre_tech = f"OFFLINE_GROUND_TRUTH:{lbl_name}"

        return StageMappingResult(
            stage=stage,
            confidence=conf,
            supporting_features=supporting_features,
            supporting_groups=supporting_groups,
            evidence={
                "dominant_stage_score": float(top_val),
                "candidate_scores": {s.value: float(v) for s, v in candidate_scores.items()},
                "reason": reason,
                "mean_abs_drift": mean_abs_drift,
            },
            horizon=horizon,
            mitre_technique=mitre_tech,
        )
