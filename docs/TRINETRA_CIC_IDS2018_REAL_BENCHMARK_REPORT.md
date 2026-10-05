# TRINETRA CSE-CIC-IDS2018 Real Benchmark Report (Phase 9A)

**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Internal Engine:** CyberHeal AI  
**SIH Problem Statement:** SIH26153  
**Target Dataset:** CSE-CIC-IDS2018 (Primary Real Benchmark Target)  
**Execution Phase:** Phase 9A — Real CSE-CIC-IDS2018 Benchmark Execution  

---

## 1. Dataset Availability
**REAL DATASET STATUS: NOT AVAILABLE LOCALLY**

### ⚠️ Strict Scientific Integrity & Anti-Fabrication Disclosure
In strict adherence to the project charter and SIH26153 integrity rules:
- A recursive search of all repository folders (`d:\sih 2026\CyberHeal`), user download directories, and connected drive volumes confirmed that the official multi-gigabyte **CSE-CIC-IDS2018** dataset captures are not present locally.
- Per strict operating constraints, automated internet downloads were **not performed**.
- **No real benchmark metrics have been fabricated, guessed, or extrapolated.**
- Real benchmark status is documented as **PENDING LOCAL DATA PLACEMENT**.
- The complete pipeline infrastructure was verified against deterministic fixtures explicitly labeled **`SYNTHETIC SOFTWARE VALIDATION RESULTS`**.

---

## 2. Dataset Provenance
- **Official Title:** Communications Security Establishment & Canadian Institute for Cybersecurity IDS 2018 (CSE-CIC-IDS2018).
- **Authors / Institutions:** Canadian Institute for Cybersecurity (University of New Brunswick) & Communications Security Establishment (Canada).
- **Capture Environment:** 50 target servers, 420 workstations, 30 victim servers, spanning 5 distinct department subnets on AWS infrastructure.
- **Capture Duration:** 10 days of network activity (February 14, 2018 – March 2, 2018).
- **Official Volume:** ~16.2 million bidirectional flow records across 10 daily CSV captures; ~400 GB raw PCAPs.
- **Expected Directory:** `d:\sih 2026\CyberHeal\data\cic_ids2018/`.
- **Local Status:** `NOT_FOUND` (0 files, 0 bytes).

---

## 3. Dataset Statistics (Real vs. Synthetic Validation)

| Attribute | Real CSE-CIC-IDS2018 (Expected) | Synthetic Software Validation (Tested) |
|---|---|---|
| **Total Ingested Records** | ~16,233,002 | 80 |
| **Usable Records** | Full corpus | 80 ($100.0\%$) |
| **Discarded Header Repeats** | Varies by file | 0 ($0.0\%$) |
| **Benign Samples** | ~13,484,708 ($~83\%$) | 40 ($50.0\%$) |
| **Attack Samples** | ~2,748,294 ($~17\%$) | 40 ($50.0\%$) |
| **Attack Classes Present** | 14 attack families + Benign | `BENIGN`: 40, `DDoS`: 20, `Bot`: 20 |
| **Temporal Span** | 10 days (Feb 14 – Mar 02, 2018) | 80 minutes ($1518602400$ to $1518607140$) |

---

## 4. Feature Coverage
- **Flow Feature Schema (78-D):** Aligned via `align_dataframe_to_schema` and `CIC_COLUMN_ALIASES`. Ingests all standard CIC-IDS2018 headers (packet lengths, inter-arrival times, subflows, active/idle times, TCP flag counts).
- **Packet Feature Schema (18-D):** Enforces TTL statistics, TCP retransmissions, packet IAT, and packet jitter.
- **Real Input Assessment:**
  - When CSV flow records are provided:
    > *"Packet-level metrics could not be independently evaluated because the available input contains flow-level records rather than raw packets."*
  - When raw PCAP captures are supplied:
    > Full 18-D packet-level enrichment is computed and reported.

---

## 5. Data Quality Audit
*Audit results from automated validation on the deterministic pipeline fixture:*
- **Total Rows Processed:** 80
- **Usable Records:** 80 ($100.0\%$)
- **Discarded Rows:** 0 ($0.0\%$)
- **Duplicate Records Sanitized:** 0
- **Missing Values Sanitized:** 0
- **Infinite Values Sanitized:** 0
- **Feature Coverage Ratio:** $100\%$ canonical features populated.
- **Class Balance:** $50\%$ Benign, $50\%$ Attack.

*(Upon placement of real CSE-CIC-IDS2018 files, `DataQualityReport` will automatically audit the full ~16.2M records and emit `data_quality_report.json`).*

---

## 6. Leakage-Safe Chronological Split Methodology
- **Split Ratio:** $70\%$ Train ($56$ rows), $15\%$ Validation ($12$ rows), $15\%$ Test ($12$ rows).
- **Temporal Monotonicity:** Preserves strict time ordering:
  $$\max(t_{\text{train}}) = 1518605700.0 \le \min(t_{\text{val}}) = 1518605760.0$$
  $$\max(t_{\text{val}}) = 1518606420.0 \le \min(t_{\text{test}}) = 1518606480.0$$
- **Leakage Prevention:**
  - Zero cross-split window leakage.
  - Preprocessing scalers fitted **strictly on the training split**.
  - Decision threshold tuned on the validation split without test leakage.

---

## 7. Supervised Logistic Regression Baseline Results

| Metric | Real CSE-CIC-IDS2018 | Synthetic Software Validation |
|---|---|---|
| **Precision** | PENDING LOCAL DATA | **1.0000** |
| **Recall** | PENDING LOCAL DATA | **0.5000** |
| **F1 Score** | PENDING LOCAL DATA | **0.6667** |
| **False Positive Rate (FPR)** | PENDING LOCAL DATA | **0.0000** |
| **ROC-AUC** | PENDING LOCAL DATA | **0.9583** |
| **Target Positive Class** | `ATTACK (1)` | `ATTACK (1)` |
| **Target Negative Class** | `BENIGN (0)` | `BENIGN (0)` |

---

## 8. Multi-Horizon World Model Forecasting Results
*Evaluated on the existing production World Model (`StateResidualDynamicsModel`) without retraining:*

| Horizon | Real CSE-CIC-IDS2018 MAE | Real CSE-CIC-IDS2018 MSE | Synthetic Validation MAE | Synthetic Validation MSE |
|---|---|---|---|---|
| **k=1** | PENDING LOCAL DATA | PENDING LOCAL DATA | **0.3812** | **0.2844** |
| **k=3** | PENDING LOCAL DATA | PENDING LOCAL DATA | **0.3812** | **0.2844** |
| **k=5** | PENDING LOCAL DATA | PENDING LOCAL DATA | **0.3812** | **0.2844** |

---

## 9. Baseline Comparisons (World Model vs. Persistence vs. Linear)

| Horizon | Persistence Baseline MAE | Linear Dynamics Baseline MAE | World Model MAE | Relative Forecasting Gain |
|---|---|---|---|---|
| **k=1** | 0.4079 | 0.4079 | **0.3812** | **+6.55%** over Persistence |
| **k=3** | 0.4079 | 0.4079 | **0.3812** | **+6.55%** over Persistence |
| **k=5** | 0.4079 | 0.4079 | **0.3812** | **+6.55%** over Persistence |

---

## 10. Real Unseen Attack Generalization Evaluation
*Evaluated using holdout isolation on available attack categories:*
- **Held-Out Category:** `Bot` (completely excluded from training and validation splits).
- **Known Categories:** `DDoS`.

| Metric | Real CSE-CIC-IDS2018 | Synthetic Software Validation |
|---|---|---|
| **Known Attack F1** | PENDING LOCAL DATA | **1.0000** |
| **Unseen Attack F1** | PENDING LOCAL DATA | **0.6667** |
| **Known Attack Recall** | PENDING LOCAL DATA | **1.0000** |
| **Unseen Attack Recall** | PENDING LOCAL DATA | **1.0000** |
| **Known Attack FPR** | PENDING LOCAL DATA | **0.0000** |
| **Unseen Attack FPR** | PENDING LOCAL DATA | **0.0000** |
| **F1 Degradation ($\Delta F1$)** | PENDING LOCAL DATA | **+0.3333** |

---

## 11. Predictive Risk & Infiltration Estimation
- **Risk Score:** Normalised bounded indicator $R \in [0.0, 1.0]$.
- **Infiltration Score:** Documented as a **heuristic infiltration score** (uncalibrated proxy for progressive adversary foothold).
- **Calibration Status:** Uncalibrated (not claimed as true posterior probability).

---

## 12. Attack Stage Progression Mapping
- **Stages Evaluated:** `NORMAL`, `RECONNAISSANCE`, `INITIAL_ACCESS`, `INFILTRATION`, `IMPACT`.
- **MITRE ATT&CK Compliance:** `mitre_technique = None` (no technique IDs fabricated without verified external ground truth).

---

## 13. Explainability (State Delta Attribution)
- **Method:** Deterministic State Delta Attribution across 280 dimensions and 5 structural groups (`flags`, `macro`, `endpoints`, `continuous`, `shape`).
- **Disclosure:** Explicitly documented as **not SHAP**.

---

## 14. Packet-Level Evaluation Status
- **Current Input:** No raw PCAPs present in workspace.
- **Reporting Status:** `"Packet-level metrics could not be independently evaluated because the available input contains flow-level records rather than raw packets."`
- **Readiness:** The 18-D packet feature schema stands fully implemented and verified.

---

## 15. CyberHeal Dry-Run Integration
- **Gating Mechanism:** `PredictiveTriggerGate` correctly evaluates risk, stage, and explanation to generate `AUTO_TRIGGER` / `HUMAN_REVIEW`.
- **Adapter Bridge:** `PredictiveIncidentAdapter` formats `AgentState` for CyberHeal intake.
- **Remediation Execution Mode:** **`SIMULATION / DRY_RUN`**.
- **Real Host Mutation Check:** `real_host_remediation_performed == False` (zero real firewall or service disruption).

---

## 16. Limitations
1. **Local Data Prerequisite:** Official benchmark metrics require the human operator to place the ~10 GB CSV or ~400 GB PCAP dataset in `data/cic_ids2018/`.
2. **Packet Granularity:** Flow-only CSV records do not provide raw TCP/IP headers; raw PCAPs are necessary for TTL and retransmission metrics.
3. **Infiltration Calibration:** Heuristic infiltration score is an ordinal indicator rather than an empirically calibrated probability.

---

## 17. Reproducibility Command (For Operator)
To run the benchmark once CSE-CIC-IDS2018 is copied locally:
```powershell
python scripts/run_real_data_benchmark.py `
    --dataset cic-ids2018 `
    --data-dir data/cic_ids2018 `
    --output-dir artifacts/real_data_benchmark/cic_ids2018 `
    --input-format auto
```

---

## 18. Final Conclusion & SIH Compliance
- **CSE-CIC-IDS2018 Benchmark Infrastructure:** **100% IMPLEMENTED & VERIFIED**
- **Real CSE-CIC-IDS2018 Execution on Local Files:** **NOT VERIFIED (PENDING LOCAL DATA PLACEMENT)**
- **Regression Suite:** **233 / 233 TESTS PASSING (100%)**
- **Production Model Protection:** Verified untouched ([`artifacts/world_model/best_model.pt`](file:///d:/sih%202026/CyberHeal/artifacts/world_model/best_model.pt)).
