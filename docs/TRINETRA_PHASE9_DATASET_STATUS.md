# TRINETRA Phase 9 — Real Dataset Availability Status

## DATASET STATUS:
**NOT AVAILABLE LOCALLY**

---

### 1. Executive Summary & Anti-Fabrication Notice
In strict compliance with the **TRINETRA anti-fabrication mandate**, no real-world benchmark metrics (Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, MAE, MSE, Zero-Day Generalization) have been fabricated, estimated, or conflated with synthetic test results.

A recursive scan across the local development machine and workspace (`d:\sih 2026\CyberHeal` and `d:\sih 2026\`) verified that neither **CSE-CIC-IDS2018** nor **CTU-13** labelled capture files (CSV or PCAP) are present locally. In accordance with system constraints, automated internet downloads were strictly prohibited.

The complete **Phase 9 Real-Data Ingestion, Normalization, Data Quality Audit, Chronological Splitting, Supervised Baseline, Multi-Horizon World Model Forecasting, Unseen Attack Holdout, and CyberHeal Dry-Run Orchestration Engine** has been fully implemented, verified, and unit-tested against isolated synthetic test fixtures labeled `SYNTHETIC SOFTWARE VALIDATION`.

---

### 2. Expected Dataset Specifications

#### Primary Target: CSE-CIC-IDS2018
- **Provider:** Communications Security Establishment (CSE) & Canadian Institute for Cybersecurity (CIC)
- **Official Title:** Communications Security Establishment & Canadian Institute for Cybersecurity IDS 2018 (CSE-CIC-IDS2018)
- **Expected Formats:**
  - **Option A (Flow CSV Records):** 10 daily CSV flow capture files (`02-14-2018.csv` through `03-02-2018.csv`), ~16.2M records, ~80 columns including `Timestamp` and `Label`.
  - **Option B (Raw PCAPs):** Raw network packet captures (PCAP/PCAPNG) for packet-level signal extraction.
- **Recommended Local Directory:** `d:\sih 2026\CyberHeal\data\cic_ids2018`

#### Secondary Target: CTU-13
- **Provider:** Czech Technical University (Stratosphere Laboratory)
- **Official Title:** CTU-13 Dataset (A Labeled Dataset with Botnet, Normal and Background traffic)
- **Expected Formats:** BiArgus flow CSVs or bidirectional NetFlow records with `Tag` / `Label` columns, or raw PCAP files across 13 capture scenarios.
- **Recommended Local Directory:** `d:\sih 2026\CyberHeal\data\ctu13`

---

### 3. Local Execution Commands (For Human Operator)

Once the human operator downloads and unpacks the official dataset files into the local directories, the real benchmark is initiated with a single deterministic command:

#### For CSE-CIC-IDS2018:
```powershell
python scripts/run_real_data_benchmark.py `
    --dataset cic-ids2018 `
    --data-dir data/cic_ids2018 `
    --output-dir artifacts/real_data_benchmark/cic_ids2018 `
    --input-format auto
```

#### For CTU-13:
```powershell
python scripts/run_real_data_benchmark.py `
    --dataset ctu13 `
    --data-dir data/ctu13 `
    --output-dir artifacts/real_data_benchmark/ctu13 `
    --input-format auto
```

---

### 4. Benchmark Execution Pipeline Ready for Execution

When the dataset is present, `run_real_data_benchmark.py` will autonomously execute:
1. **Cryptographic Ingestion & Provenance:** Computes SHA-256 hashes of all input files, measures total byte volume, counts raw records, and logs timestamped metadata into `provenance.json`.
2. **Data Quality Audit:** Detects and scrubs infinite values, missing features, duplicate rows, and header repeats. Audits benign/attack ratio, attack category distribution, and feature coverage. Emits `data_quality_report.json`.
3. **Leakage-Safe Chronological Splitting:** Enforces 70% Train, 15% Validation, and 15% Test partition strictly ordered by Unix epoch timestamps (`train_time_max <= val_time_min <= val_time_max <= test_time_min`). Emits `split_summary.json`.
4. **Feature Extraction & Alignment:** Normalizes column headers via `CIC_COLUMN_ALIASES` into the 78 canonical flow feature schema. If PCAP input is provided, executes Phase 8 packet-level signal extraction. Emits `feature_report.json`.
5. **Supervised Logistic Regression Baseline:** Fits scikit-learn Logistic Regression on training split, tunes decision threshold on validation split strictly without test leakage, and evaluates against test split. Emits `logistic_baseline_metrics.json`.
6. **Network State Construction & World Model Forecasting:** Encodes sliding temporal windows into $S_t \in \mathbb{R}^{280}$, forms future state pairs for horizons $k \in \{1, 3, 5\}$, and compares the current production World Model against Persistence and Linear Dynamics baselines. Emits `world_model_forecasting_metrics.json`.
7. **Real Unseen Attack Generalization:** Identifies available attack families, systematically isolates a holdout category from training/validation, and computes generalization degradation metrics ($\Delta F1$, $\Delta Recall$, $\Delta FPR$). Emits `unseen_attack_metrics.json`.
8. **Risk, Stage Progression & Explainability:** Evaluates `PredictiveRiskEstimator`, `AttackStageMapper`, and `ForecastExplainabilityEngine` (State Delta Attribution) on observed trajectories. Emits `risk_and_stage_metrics.json`.
9. **CyberHeal Dry-Run Gating:** Validates `PredictiveTriggerGate` evaluation and `PredictiveIncidentAdapter` intake in safe `SIMULATION / DRY_RUN` mode (zero real firewall/host modifications). Emits `cyberheal_simulation_summary.json`.
10. **Isolated Artifact Storage:** Writes all outputs into `artifacts/real_data_benchmark/<dataset>/<exp_id>/` without ever modifying `artifacts/world_model/best_model.pt`. Emits `benchmark_manifest.json` and `BENCHMARK_SUMMARY.md`.

---

### 5. Benchmark Stages Remaining Pending

| Pipeline Stage | Implementation Status | Real-Data Execution Status | Pending Prerequisite |
|---|---|---|---|
| Dataset Discovery & Format Detection | COMPLETE | PENDING LOCAL DATA | Place dataset files in `data/` |
| SHA-256 Provenance Logging | COMPLETE | PENDING LOCAL DATA | Place dataset files in `data/` |
| Data Quality Audit & Scrubbing | COMPLETE | PENDING LOCAL DATA | Place dataset files in `data/` |
| Leakage-Safe Chronological Split | COMPLETE | PENDING LOCAL DATA | Place dataset files in `data/` |
| 78-D Flow Feature Alignment | COMPLETE | PENDING LOCAL DATA | Place dataset files in `data/` |
| 18-D Packet Feature Extraction | COMPLETE | PENDING LOCAL PCAP | Provide `.pcap` / `.pcapng` files |
| Logistic Regression Real Benchmark | COMPLETE | PENDING LOCAL DATA | Execute CLI on placed files |
| Multi-Horizon World Model Forecasting | COMPLETE | PENDING LOCAL DATA | Execute CLI on placed files |
| Persistence & Linear Baselines | COMPLETE | PENDING LOCAL DATA | Execute CLI on placed files |
| Real Unseen Attack Generalization | COMPLETE | PENDING LOCAL DATA | Multi-class attacks in placed dataset |
| Risk / Infiltration Estimation | COMPLETE | PENDING LOCAL DATA | Execute CLI on placed files |
| Attack Stage Progression Mapping | COMPLETE | PENDING LOCAL DATA | Execute CLI on placed files |
| State Delta Attribution (Explainability) | COMPLETE | PENDING LOCAL DATA | Execute CLI on placed files |
| CyberHeal Dry-Run Simulation | COMPLETE | PENDING LOCAL DATA | Execute CLI on placed files |

---

### 6. SIH Compliance Standing
- **Real Dataset Benchmark Infrastructure:** **VERIFIED (COMPLETE)**
- **Real Dataset Execution on Local Files:** **NOT VERIFIED (PENDING LOCAL DATA PLACEMENT)**
- **Regression Suite:** **233 / 233 TESTS PASSING (100%)**
