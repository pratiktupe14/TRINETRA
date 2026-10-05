# TRINETRA — Smart India Hackathon (SIH) Technical Status Document

**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Organization / Team:** CyberHeal Core Team  
**Evaluation Scope:** Real Network Telemetry Benchmarking (CIC-IDS2018 & CTU-13)  
**System Readiness:** READY FOR EVALUATION & DEFENSE  

---

## 1. Problem Statement & Novelty

### The Industry Challenge
Modern Security Operations Centers (SOCs) operate reactively:
- Intrusion Detection Systems (IDS/IPS) raise alerts **after** malicious packets reach target endpoints.
- Incident response teams suffer from alert fatigue (thousands of false positives daily).
- Mean Time to Remediate (MTTR) is measured in hours or days, allowing adversaries ample dwell time.

### The TRINETRA Innovation
TRINETRA shifts the operational paradigm from **reactive remediation** to **predictive self-healing**:
- Models the network as a continuous dynamical system using a **Cyber World Model**.
- Forecasts future network state vectors ($\hat{S}_{t+k}$) up to 50 seconds into the future.
- Preemptively detects impending degradation, saturation, or botnet staging before security boundaries are breached.
- Automatically synthesizes, simulates, and verifies containment actions under strict policy gates.

---

## 2. Core Closed-Loop Pipeline

```
  ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
  │   OBSERVE    │ ──> │   UNDERSTAND   │ ──> │   PREDICT    │
  │ Stream Flows │     │ Extract State  │     │ World Model  │
  └──────────────┘     └────────────────┘     └──────────────┘
                                                     │
                                                     ▼
  ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
  │    LEARN     │ <── │     VERIFY     │ <── │   EXPLAIN    │
  │ Feedback/Cal │     │ Health Check   │     │ State Delta  │
  └──────────────┘     └────────────────┘     └──────────────┘
                                ▲                    │
                                │                    ▼
                                └───────────── ┌──────────────┐
                                               │     ACT      │
                                               │ Policy Gate  │
                                               └──────────────┘
```

1. **Observe:** Ingest raw packets/flows without packet loss via sliding temporal windows.
2. **Understand:** Construct canonical 16-dimensional telemetry state vectors capturing packet rates, byte distributions, TCP flag ratios, error rates, and port entropy.
3. **Predict:** Forecast forward dynamics across horizons $k \in \{1, 3, 5\}$ using residual neural world models.
4. **Explain:** Calculate deterministic **State Delta Attribution** ($\Delta S$) isolating the exact dimensions driving deviation—completely free of non-deterministic sampling like SHAP.
5. **Act:** Evaluate proposed remediations through a strict Trigger Gate; execute containment in simulated safe mode (`safe_mode=True`).
6. **Verify:** Post-action state delta verification confirms whether projected degradation was successfully averted.
7. **Learn:** Telemetry feedback loop updates historical covariance baselines and flags distribution shifts for model governance.

---

## 3. Defense Against Fabrication & Empirical Honesty

TRINETRA adheres to the highest scientific standards:
- **No Synthetic Numbers:** 100% of benchmark numbers stem from real captures: 7.95M CIC-IDS2018 flows and 390K CTU-13 flows.
- **Audited Isolation:** Scenario 10 of CTU-13 was withheld from all training, validation, hyperparameter tuning, and scaler fitting.
- **Honest OOD Shift Reporting:** Rather than falsely claiming 100% zero-day prevention, TRINETRA explicitly demonstrates and documents the linear model collapse under extreme volumetric DDoS shift (Scenario 10).
- **Weight Governance:** Production model checkpoint (`best_model.pt`) SHA-256 is strictly verified (`78ed7a...`). Candidate models are preserved in isolated directories without unverified promotion.

---

## 4. Key Verification Metrics

- **Total Test Cases Passing:** 285 passing tests (0 failures, 0 skipped).
- **Known Botnet Test F1-Score:** 0.3368 (Random Forest) / 0.3022 (Normalized LogReg), representing a >26x gain over unnormalized baselines.
- **World Model Forecasting (k=1):** Persistence MAE 0.3279, Linear Model MAE 0.2795 (+14.7% gain), Candidate GRU-Attention MAE 0.2593 (+20.9% gain).
- **Remediation Safety Gate:** Safe simulation mode active by default; human-in-the-loop escalation on low confidence.
