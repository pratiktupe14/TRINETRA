# TRINETRA Reflective Learning Mechanism (RLM) Architecture

## 1. Executive Summary & Core Identity

The **TRINETRA Reflective Learning Mechanism (RLM)** is the autonomous post-incident evaluation and feedback layer of the TRINETRA and CyberHeal platform.

### What RLM Is:
- A **deterministic, structured evaluation engine** that analyzes the complete lifecycle of cyber defense actions:
  $$\text{Observe} \to \text{Understand} \to \text{Predict} \to \text{Explain} \to \text{Trigger} \to \text{Respond} \to \text{Verify} \to \text{Learn}$$
- A bridge between runtime operational telemetry and offline model improvement.
- An append-only audit trail generating typed `LearningRecord` artifacts for offline training.

### What RLM Is NOT:
- **RLM is NOT an LLM:** It does not use OpenAI, external cloud models, generative AI, or natural language prompts.
- **RLM does NOT perform online weight adaptation:** Individual live incidents NEVER directly invoke PyTorch optimizers, update model weights, or overwrite production artifacts (`best_model.pt`).
- **RLM does NOT use an external database:** All persistence operates strictly through validated, append-only JSON Lines (`JSONL`) files in `artifacts/rlm/`.

---

## 2. Runtime Feedback Flow

During live operation, the runtime flow executes safely without modifying model state:

```
    TRINETRA Forecast & Explanation
                 │
                 ▼
       Predictive Trigger Gate
                 │ (AUTO_TRIGGER / HUMAN_REVIEW)
                 ▼
          CyberHeal Intake
                 │
                 ▼
        Diagnosis & Root Cause
                 │
                 ▼
       Guardian Safety Check
          ├── REJECTED ──┐
          │ (Approved)   │
          ▼              │
      Safe Remediation   │
          │              │
          ▼              │
    Post-Verification    │
          │              │
          ▼              │
    ┌────────────────────┴───┐
    │ ReflectiveLearningAgent│
    │  - Capture context     │
    │  - Compare metrics     │
    │  - Reflect rules       │
    │  - Generate recs       │
    └───────────┬────────────┘
                │
                ▼
         LearningRecord
                │
                ▼
   artifacts/rlm/learning_records.jsonl
         (Append-Only)
```

---

## 3. Structured LearningRecord Schema

Every completed or escalated incident generates a validated `LearningRecord`:

```json
{
  "record_id": "rec-1a2b3c4d5e6f",
  "incident_id": "INC-2026-0042",
  "trace_id": "TRC-2026-0042",
  "timestamp": 1700000000.0,
  "incident_type": "PREDICTIVE",
  "prediction": {
    "risk_score": 0.85,
    "confidence": 0.90,
    "predicted_stage": "RECONNAISSANCE",
    "risk_trajectory": { "h1": 0.72, "h3": 0.85, "h5": 0.91 }
  },
  "trigger": {
    "decision": "AUTO_TRIGGER",
    "reasons": ["Risk score 0.85 >= threshold 0.70", "Confidence 0.90 >= threshold 0.75"],
    "blocking_reasons": []
  },
  "action": {
    "action_type": "rate_limit_ip",
    "target": "192.168.1.50",
    "execution_mode": "SIMULATION",
    "status": "EXECUTED",
    "execution_result": { "simulated_throttled": true }
  },
  "verification": {
    "status": "VERIFIED",
    "threat_reduction_ratio": 0.82,
    "residual_risk": 0.15
  },
  "outcome": {
    "success": true,
    "failure_reason": null
  },
  "reflection": {
    "prediction_quality": "USEFUL",
    "action_quality": "EFFECTIVE",
    "verification_quality": "VERIFIED",
    "recommendations": ["NO_CHANGE"],
    "analysis": "Remediation effective: threat reduced by 82.0% (Residual Risk: 0.15). Forecast accurately preempted verified threat."
  },
  "state_references": {
    "current_state": [...],
    "predicted_state": [...],
    "observed_state": [...]
  }
}
```

---

## 4. Deterministic Reflection Rules

The `RLMReflectionEngine` applies deterministic logic without heuristic drift:

| Case | Scenario & Signals | Prediction Quality | Action Quality | Verification Quality | Recommendation |
|:---|:---|:---|:---|:---|:---|
| **1** | Prediction strong + AUTO_TRIGGER + Remediation executed + VERIFIED | `USEFUL` | `EFFECTIVE` | `VERIFIED` | `NO_CHANGE` |
| **2** | AUTO_TRIGGER + Remediation executed + NOT_VERIFIED | `PARTIALLY_USEFUL` | `INEFFECTIVE` | `NOT_VERIFIED` | `REVIEW_POLICY` |
| **3** | Guardian safety gate blocks remediation | `USEFUL` / `INCONCLUSIVE` | `REJECTED` | `NOT_VERIFIED` | `REVIEW_POLICY` |
| **4** | High confidence ($\ge 0.80$) + Large state drift ($L_2 > 2.5$) | `POOR` | `INEFFECTIVE` | `NOT_VERIFIED` | `RETRAIN_CANDIDATE` |
| **5** | Telemetry missing, corrupt, or truncated | `INCONCLUSIVE` | `INCONCLUSIVE` | `INCONCLUSIVE` | `REVIEW_DATA` |

### Critical Rule for Missing Data:
Missing or incomplete post-remediation observations **never** infer success. They default strictly to `INCONCLUSIVE` and recommend `REVIEW_DATA`.

---

## 5. Offline File Store (`RLMStore`)

1. **Storage Path:** `artifacts/rlm/learning_records.jsonl`.
2. **Deduplication:** Ingestion enforces unique `incident_id` and `record_id` keys. Repeated records for an identical incident are rejected (`on_duplicate="reject"`) or skipped (`on_duplicate="skip"`).
3. **Data Integrity:** Records containing `NaN`, `Inf`, out-of-bound probabilities ($<0$ or $>1$), or malformed enums are rejected with explicit `ValueError` exceptions. No silent truncation or silent correction is permitted.
4. **Privacy & Minimization:** Raw command lines, credentials, and packet payloads are excluded; only structured state vectors and normalized metrics are stored.
