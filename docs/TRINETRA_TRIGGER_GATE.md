# TRINETRA Predictive Trigger Gate Specification

**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Internal Response Engine:** CyberHeal — Autonomous Self-Healing Response Engine  
**Module:** `src/shield_rca/world_model/trigger_gate.py`  
**Phase:** Phase 2 — Predictive Trigger Gate  
**Implementation Status:** IMPLEMENTED  

---

## 1. Purpose

The **Predictive Trigger Gate** serves as a deterministic, policy-driven safety boundary between TRINETRA's predictive intelligence layer and CyberHeal's autonomous response orchestration.

> [!IMPORTANT]
> **The Trigger Gate does NOT execute remediation.**  
> It does not alter firewall configurations, execute OS scripts, or modify running services. Its sole responsibility is to evaluate predictive threat signals against multi-dimensional safety policies and output one of three categorical decisions: `NO_TRIGGER`, `HUMAN_REVIEW`, or `AUTO_TRIGGER`.

---

## 2. Position in Architecture

```
    ┌───────────────────────────┐
    │ Network Traffic / PCAP   │
    └─────────────┬─────────────┘
                  ↓
    ┌───────────────────────────┐
    │ Feature Extraction (78-D) │
    └─────────────┬─────────────┘
                  ↓
    ┌───────────────────────────┐
    │ Network State S_t (280-D) │
    └─────────────┬─────────────┘
                  ↓
    ┌───────────────────────────┐
    │ Temporal World Model      │
    └─────────────┬─────────────┘
                  ↓
    ┌───────────────────────────┐
    │ K-Step Forecast S_hat     │
    └─────────────┬─────────────┘
                  ↓
    ┌───────────────────────────┐
    │ Risk Estimator            │
    └─────────────┬─────────────┘
                  ↓
    ┌───────────────────────────┐
    │ Attack Stage Mapper       │
    └─────────────┬─────────────┘
                  ↓
    ┌───────────────────────────┐
    │ Explainability (Attrib.)  │
    └─────────────┬─────────────┘
                  ↓
    ╔═══════════════════════════╗
    ║   PREDICTIVE TRIGGER      ║
    ║          GATE             ║
    ╚═════════════╤═════════════╝
                  │
          ┌───────┼────────┐
          ↓       ↓        ↓
       NO       HUMAN    AUTO
      TRIGGER   REVIEW  TRIGGER
                          │
                          ↓
                    PHASE 3
                    CYBERHEAL
```

---

## 3. Inputs

The gate accepts either raw scalar inputs or direct Phase 1 output dataclasses:

| Input Field | Type | Valid Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `risk_score` | `float` \| `HorizonRiskEstimate` \| `RiskTrajectory` | $[0.0, 1.0]$ | Predictive group-weighted trajectory risk index. |
| `confidence` | `float` \| `None` | $[0.0, 1.0]$ or `None` | Model/evaluation confidence proxy. |
| `evidence_strength` | `float` \| `None` | $[0.0, 1.0]$ or `None` | Telemetry drift and attribution concentration. Derived if omitted. |
| `asset_criticality` | `AssetCriticality` \| `str` | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Operational sensitivity of the target asset. |
| `action_safety` | `ActionSafety` \| `str` | `SAFE`, `RESTRICTED`, `UNSAFE`, `UNKNOWN` | Blast-radius classification of downstream action. |
| `predicted_stage` | `AttackStage` \| `StageMappingResult` \| `str` | `NORMAL`, `RECONNAISSANCE`, `INITIAL_ACCESS`, `INFILTRATION`, `IMPACT`, `UNKNOWN` | MITRE-grounded attack stage. |
| `explanation` | `ForecastExplanation` \| `None` | Phase 1 object | Used to derive `evidence_strength` via State Delta Attribution. |
| `risk_trajectory` | `RiskTrajectory` \| `None` | Phase 1 object | Multi-horizon risk summary ($k \in \{1, 3, 5\}$). |
| `metadata` | `dict[str, Any]` \| `None` | Key-value pairs | Contextual identifiers (incident ID, IP) preserved immutably. |

---

## 4. Outputs

The gate returns an immutable, JSON-serializable `TriggerDecision` dataclass:

```python
@dataclass(frozen=True, slots=True)
class TriggerDecision:
    decision: TriggerDecisionType      # NO_TRIGGER, HUMAN_REVIEW, AUTO_TRIGGER
    risk_score: float                  # Evaluated risk score in [0.0, 1.0]
    confidence: float | None           # Evaluated confidence proxy
    evidence_strength: float           # Evaluated evidence score in [0.0, 1.0]
    asset_criticality: AssetCriticality# Asset criticality enum
    action_safety: ActionSafety        # Action safety enum
    predicted_stage: AttackStage       # Attack stage enum
    reasons: list[str]                 # Why this decision was reached
    blocking_reasons: list[str]        # Why AUTO_TRIGGER was withheld (if applicable)
    policy_version: str                # e.g. "1.0.0"
    risk_trajectory_summary: dict | None
    metadata: dict[str, Any]
```

---

## 5. Decision States

1. **`NO_TRIGGER`:**
   - **Meaning:** Do not initiate any response; network is either quiescent or risk is below the operational review threshold.
   - **Trigger Conditions:**
     - `risk_score < policy.risk_review_threshold` (default $< 0.35$).
     - OR `predicted_stage == AttackStage.NORMAL` and `risk_score < policy.risk_auto_threshold`.

2. **`HUMAN_REVIEW`:**
   - **Meaning:** Concerning predictive signal detected, but automatic intervention is unsafe or insufficiently justified. Alert human SOC analyst.
   - **Trigger Conditions:**
     - `risk_score >= policy.risk_review_threshold`.
     - AND one or more `AUTO_TRIGGER` safety criteria fail (e.g. low confidence, ambiguous evidence, critical asset, unknown action safety, or unknown attack stage).

3. **`AUTO_TRIGGER`:**
   - **Meaning:** Threat is high-confidence, well-supported, targeted at an allowable asset, and verified safe for automated intervention. Forward to CyberHeal orchestration.
   - **Trigger Conditions (All Must Pass Simultaneously):**
     1. `risk_score >= policy.risk_auto_threshold` (default $\ge 0.75$).
     2. `confidence >= policy.confidence_auto_threshold` (default $\ge 0.75$).
     3. `evidence_strength >= policy.evidence_auto_threshold` (default $\ge 0.60$).
     4. `action_safety == ActionSafety.SAFE` (if `require_safe_action=True`).
     5. `asset_criticality <= policy.maximum_auto_asset_criticality` (default $\le \text{MEDIUM}$).
     6. `predicted_stage in policy.allowed_auto_stages` and $\notin \text{blocked\_auto\_stages}$.
     7. Multi-horizon trajectory escalation constraint satisfied (if enabled).

---

## 6. Policy Dimensions & Thresholds

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `risk_auto_threshold` | `0.75` | Minimum risk score required for automated response. |
| `risk_review_threshold` | `0.35` | Minimum risk score required to alert human operators. |
| `confidence_auto_threshold` | `0.75` | Minimum model confidence required for automated response. |
| `confidence_review_threshold`| `0.40` | Baseline confidence for escalating ambiguous signals. |
| `evidence_auto_threshold` | `0.60` | Minimum telemetry evidence concentration required for auto action. |
| `maximum_auto_asset_criticality` | `AssetCriticality.MEDIUM` | Highest asset criticality permitted for automated mitigation without human sign-off. |
| `require_safe_action` | `True` | Enforces that only `ActionSafety.SAFE` actions can be automatically executed. |
| `allowed_auto_stages` | `RECON, INITIAL_ACCESS, INFILTRATION, IMPACT` | Permitted stages for automated mitigation. |
| `blocked_auto_stages` | `NORMAL, UNKNOWN` | Stages strictly prohibited from automated mitigation. |

---

## 7. Mathematical Evidence Strength Derivation

When `evidence_strength` is not explicitly supplied by an operator, the gate computes it deterministically from Phase 1 artifacts:

1. **From `ForecastExplanation`:**
   $$\text{drift\_saturation} = 1.0 - \exp(-1.5 \cdot \text{total\_drift\_magnitude})$$
   $$\text{top\_concentration} = \min\left(1.0, \sum_{i=1}^{\min(3, |F|)} \text{relative\_share}_i\right)$$
   $$\text{group\_factor} = \min\left(1.0, \frac{\text{active\_groups}}{3.0}\right)$$
   $$\text{evidence\_strength} = 0.50 \cdot \text{drift\_saturation} + 0.30 \cdot \text{top\_concentration} + 0.20 \cdot \text{group\_factor} \in [0.0, 1.0]$$

2. **From `RiskTrajectory`:**
   $$\text{evidence\_strength} = 0.60 \cdot \text{aggregate\_risk} + 0.40 \cdot (0.80 \text{ if is\_escalating else } 0.40)$$

3. **From `HorizonRiskEstimate`:**
   $$\text{evidence\_strength} = 0.70 \cdot (1.0 - \exp(-0.5 \cdot \text{total\_l1\_drift})) + 0.30 \cdot \text{risk\_score}$$

---

## 8. Multi-Horizon Trajectory Handling

When a `RiskTrajectory` ($k \in \{1, 3, 5\}$) is passed:
- The peak horizon risk and trajectory-wide escalation flag (`is_escalating`) are evaluated.
- If `require_trajectory_escalation_for_borderline=True`, borderline risks ($[0.75, 0.85]$) without sustained multi-horizon escalation are gated to `HUMAN_REVIEW`.
- The trajectory metadata is preserved in `TriggerDecision.risk_trajectory_summary`.

---

## 9. Explainability: Reasons & Blocking Reasons

Every decision emits human-readable rationale:
- **`reasons`:** Affirmative statements describing which criteria were satisfied or why escalation occurred.
- **`blocking_reasons`:** Explicit gating reasons detailing exactly why `AUTO_TRIGGER` was withheld (e.g. `Action safety is UNKNOWN`, `Asset criticality (CRITICAL) exceeds auto threshold (MEDIUM)`).

---

## 10. Conservative Safety Behavior & Anti-Fabrication

1. **Unknown Action Safety:** An action whose safety profile has not been verified (`UNKNOWN`) is NEVER permitted to auto-trigger.
2. **Unknown Attack Stage:** Network anomalies lacking a clear attack signature are gated to `HUMAN_REVIEW`.
3. **Critical Infrastructure Protection:** Assets marked `HIGH` or `CRITICAL` always require human approval by default to prevent unintended operational outages.
4. **Finite Bound Enforcement:** NaN, Inf, negative, and out-of-bounds values immediately raise `ValueError`.

---

## 11. Offline Operation & Determinism

- **Zero Cloud/API Dependencies:** Pure Python/NumPy logic.
- **Execution Speed:** 100 evaluations execute in $< 0.01$ seconds.
- **Deterministic:** Identical inputs and policy always produce bitwise identical decisions.

---

## 12. Future Phase 3 Integration Boundary

In Phase 3, CyberHeal will consume the `TriggerDecision`:
- If `decision == AUTO_TRIGGER`: Hand off `incident_id`, `predicted_stage`, and `risk_score` to `SentinelAgent` / `PlannerAgent`.
- If `decision == HUMAN_REVIEW`: Post review notification to the SOC queue and pause automated remediation pending analyst confirmation.
- If `decision == NO_TRIGGER`: Record telemetry summary to rolling buffer and suppress response.
