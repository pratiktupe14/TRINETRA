# TRINETRA Phase 9 Implementation & Evaluation Report

### Phase
**Phase 9 — Real CIC-IDS2018 / CTU-13 Evaluation & Benchmarking**

---

### Dataset Status
**REAL DATA NOT AVAILABLE LOCALLY**

> **Mandatory Anti-Fabrication Statement:**  
> In strict accordance with the TRINETRA engineering principles and SIH26153 specifications, no real-world benchmark metrics (such as CIC-IDS2018 precision, recall, F1, FPR, zero-day generalization, or multi-horizon MAE) have been synthesized, fabricated, or inferred from synthetic fixtures. A recursive filesystem inspection across `d:\sih 2026\` and all system paths confirmed that neither CSE-CIC-IDS2018 nor CTU-13 dataset captures (CSV or PCAP) were present locally. Internet downloads were prohibited.  
> Consequently, real-data benchmark execution was halted in accordance with Phase 9 Section 3, and the complete end-to-end benchmark architecture was validated against deterministic software fixtures explicitly categorized as `SYNTHETIC SOFTWARE VALIDATION`.

---

### Dataset
- **Target Primary Dataset:** Communications Security Establishment & Canadian Institute for Cybersecurity IDS 2018 (CSE-CIC-IDS2018).
- **Target Secondary Dataset:** CTU-13 Dataset (Stratosphere Laboratory, Czech Technical University).
- **Format Capabilities Implemented:**
  - Automated format detector (`discover_dataset`) for CSV flow captures and PCAP/PCAPNG packet traces.
  - Dedicated dataset adapters (`CicIds2018Adapter`, `Ctu13Adapter`) with schema alignment to canonical 78 flow features and Phase 8 packet features.
  - Full cryptographic provenance tracking with SHA-256 file hashing (`compute_file_sha256`).

---

### Data Quality (Software Validation Fixture)
*Audited on isolated deterministic validation fixture:*
- **Total Ingested Rows:** 80
- **Usable Records:** 80 (100.0%)
- **Discarded Rows:** 0 (0.0%)
- **Duplicate Records:** 0
- **Missing Values Sanitized:** 0
- **Infinite Values Sanitized:** 0
- **Class Distribution:** 40 Benign (50.0%), 40 Attack (50.0%)
- **Attack Categories:** `BENIGN`: 40, `DDoS`: 20, `Bot`: 20
- **Timestamp Span:** 1518602400.0 to 1518607140.0 (strict chronological ordering verified)

*(For real datasets, actual numbers will be computed and logged automatically to `data_quality_report.json` upon dataset placement).*

---

### Feature Coverage
- **Flow Feature Dimensions:** 78 canonical flow features (`CANONICAL_FEATURES`).
- **Feature Schema Alignment:** Automatic mapping from `CIC_COLUMN_ALIASES` with float32 casting.
- **Packet-Level Feature Status:** Evaluated and reported transparently. On flow-only CSV inputs, the engine explicitly outputs:
  > *"Packet-level features could not be evaluated from the flow-only input."*  
  When PCAP inputs are supplied, Phase 8 packet extraction (TTL, TCP retransmissions, packet IAT, packet IAT jitter) is invoked.

---

### Logistic Regression Baseline
*Note: Real-data benchmark pending local dataset placement. The supervised baseline was verified against the software validation fixture:*
- **Task:** Supervised Binary Classification (`0 = BENIGN`, `1 = ATTACK`).
- **Class Weighting:** `class_weight="balanced"`.
- **Threshold Tuning:** Performed exclusively on the validation split (strictly zero test-data leakage).
- **Software Validation Results (80 samples):**
  - **Precision:** 1.0000
  - **Recall:** 0.5000
  - **F1 Score:** 0.6667
  - **False Positive Rate (FPR):** 0.0000
  - **ROC-AUC:** 0.9583
  - **Positive Class:** ATTACK (1)
  - **Negative Class:** BENIGN (0)

---

### World Model Multi-Horizon Forecasting
*Evaluated on the current production checkpoint (`artifacts/world_model/best_model.pt`) without retraining:*
- **State Representation:** 280-dimensional Network State $S_t \in \mathbb{R}^{280}$.
- **Evaluation Horizons:** $k \in \{1, 3, 5\}$.
- **Baselines Evaluated:** Persistence Baseline ($\hat{S}_{t+k} = S_t$) and Linear Dynamics Baseline ($\hat{S}_{t+k} = S_t \mathbf{W}_k + \mathbf{b}_k$).
- **Software Validation Results (Horizons k=1, 3, 5):**

| Horizon | Persistence MAE | Linear Dynamics MAE | World Model MAE | Relative Gain over Persistence |
|---|---|---|---|---|
| **k=1** | 0.4079 | 0.4079 | 0.3812 | **+6.55%** |
| **k=3** | 0.4079 | 0.4079 | 0.3812 | **+6.55%** |
| **k=5** | 0.4079 | 0.4079 | 0.3812 | **+6.55%** |

*(Deterministic output verified; production model weights remained strictly untouched).*

---

### Unseen Attack Generalization
*Evaluated via category-holdout protocol:*
- **Held-Out Category:** `Bot` (strictly barred from training and validation splits).
- **Known Categories:** `DDoS`.
- **Software Validation Generalization Results:**
  - **Known Attack F1:** 1.0000
  - **Unseen Attack F1:** 0.6667
  - **Known Attack Recall:** 1.0000
  - **Unseen Attack Recall:** 1.0000
  - **$\Delta F1$ (Degradation):** +0.3333
  - **$\Delta \text{Recall}$:** 0.0000
  - **$\Delta \text{FPR}$:** 0.0000

---

### Risk, Stage & Explainability Evaluation
- **Predictive Risk Estimator:** Evaluated on forecasted state trajectories. Output is bounded in $[0.0, 1.0]$ with honest disclosure of heuristic infiltration score.
- **Attack Stage Progression Mapper:** Mapped trajectory drift into network-observable stages (`RECONNAISSANCE`, `INITIAL_ACCESS`, `INFILTRATION`, `IMPACT`). Anti-fabrication check verified: granular MITRE technique IDs are not fabricated without ground truth (`mitre_technique = None`).
- **Explainability:** Evaluated via deterministic **State Delta Attribution** (disclosed as not SHAP), extracting the top 5 contributing feature drivers and relative group shares across the 5 architectural state groups.

---

### CyberHeal Dry-Run Integration Result
- **Trigger Gate Decision:** Verified handoff of high-risk predictive signals into `PredictiveIncidentAdapter`.
- **Execution Mode:** Strictly enforced `SIMULATION / DRY_RUN`.
- **Safety Invariant:** `real_host_remediation_performed == False`. Zero real network firewall, iptables, netsh, or service shutdown actions were executed against the operating system.

---

### Reproducibility & CLI Execution
A unified CLI tool was created for one-command execution once datasets are available locally:
```powershell
python scripts/run_real_data_benchmark.py `
    --dataset cic-ids2018 `
    --data-dir data/cic_ids2018 `
    --output-dir artifacts/real_data_benchmark/cic_ids2018
```
- If the target data directory is missing, the CLI fails fast with exit code 1 and outputs:
  `ERROR: Real dataset not found in '<path>'. No benchmark metrics were generated.`

---

### Artifacts Generated & Isolated
Benchmark artifacts are isolated in versioned directories under `artifacts/real_data_benchmark/<dataset>/<exp_id>/`:
- `provenance.json`: Cryptographic SHA-256 hashes, file sizes, and record counts.
- `data_quality_report.json`: Data scrubbing audit, missing/infinite counts, class distribution.
- `split_summary.json`: Leakage-safe chronological split boundaries and sample counts.
- `feature_report.json`: Canonical feature coverage and packet-level status.
- `logistic_baseline_metrics.json`: Precision, Recall, F1, FPR, ROC-AUC, Confusion Matrix.
- `world_model_forecasting_metrics.json`: Multi-horizon MAE/MSE vs Persistence & Linear baselines.
- `unseen_attack_metrics.json`: Holdout category generalization degradation.
- `risk_and_stage_metrics.json`: Predictive risk indices, infiltration score, and attack stages.
- `cyberheal_simulation_summary.json`: Dry-run handoff and simulation audit.
- `benchmark_manifest.json`: Execution metadata and safety certification.
- `BENCHMARK_SUMMARY.md`: Human-readable markdown executive summary.

**Model Weight Safety:** `artifacts/world_model/best_model.pt` was checked before and after all benchmark executions; file hash and modification time were confirmed completely unchanged.

---

### Tests & Regression Verification
- **Baseline Prior to Phase 9:** 213 tests passing.
- **New Phase 9 Tests Implemented:** 20 tests (`tests/test_real_data_benchmark.py`).
- **Current Regression Suite:** **233 tests passing (100%), 0 failed, 0 skipped.**
  - Test duration: 11.34s.

---

### Limitations
1. **Local Dataset Absence:** Real multi-gigabyte PCAP/CSV archives must be placed locally in `data/` by the operator to generate official real-data numbers.
2. **Packet Features on CSV:** Flow-only CSV records do not contain packet headers; packet-level features require PCAP/PCAPNG captures.
3. **Infiltration Calibration:** Infiltration scores are deterministic drift indicators, not calibrated probabilities.

---

### SIH Compliance Standing
- **Real Dataset Benchmark Infrastructure:** **VERIFIED (COMPLETE)**
- **Real Dataset Benchmark Execution on Local Data:** **NOT VERIFIED (PENDING LOCAL DATA PLACEMENT)**
- **Overall System Integrity:** **100% (233 / 233 TESTS PASSING)**
