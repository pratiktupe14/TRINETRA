# TRINETRA Real-Data Evaluation & Benchmarking Architecture (Phase 9)

**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Internal Engine:** CyberHeal AI  
**SIH Problem Statement:** SIH26153  
**Document Type:** Technical Architecture, Methodology, and Benchmark Specification  

---

## 1. Primary Objective
The objective of Phase 9 is to execute reproducible, leakage-safe real-world evaluations of TRINETRA on labelled network traffic datasets (primary: CSE-CIC-IDS2018, secondary: CTU-13), without fabricating metrics or modifying production model weights.

The evaluation measures:
1. Mathematical fidelity of continuous Network State forecasts ($S_t \in \mathbb{R}^{280}$) across multi-horizon steps ($k \in \{1, 3, 5\}$).
2. Comparative forecasting gain against Persistence and Linear Dynamics baselines.
3. Classification performance of an aligned Logistic Regression baseline on 78 canonical flow features.
4. Generalization degradation under real held-out unseen attack scenarios.
5. Autonomous triggering, risk estimation, attack-stage progression mapping, and state-delta attribution in a zero-risk simulated dry-run mode.

---

## 2. Supported Datasets
The engine supports two primary open-source network traffic corpora via dedicated adapter classes:

### A. CSE-CIC-IDS2018 (`CicIds2018Adapter`)
- **Origin:** Canadian Institute for Cybersecurity / Communications Security Establishment.
- **Topology:** Comprehensive enterprise network (50 servers, 420 PCs, 30 servers across multiple subnets).
- **Attack Types:** DDoS (LOIC-HTTP, HOIC, LOIC-UDP), DoS (Hulk, GoldenEye, Slowloris, SlowHTTPTest), BruteForce (FTP, SSH), Web Attacks (SQLi, XSS, Brute Force), Botnet, Infiltration.
- **Inputs Supported:**
  - 10 daily CSV flow record files (e.g. `02-14-2018.csv`, `02-15-2018.csv`, etc.).
  - Raw packet captures (`.pcap` / `.pcapng`).

### B. CTU-13 (`Ctu13Adapter`)
- **Origin:** Stratosphere Laboratory, Czech Technical University in Prague.
- **Topology:** Mixed real enterprise/campus background traffic merged with real botnet capture scenarios.
- **Attack Types:** Botnet C&C traffic (Mirai, Torii, Neris, Rbot, Virut, Menti, Sogou, Murlo), DDoS, and PortScan.
- **Inputs Supported:** BiArgus flow CSV captures and raw scenario PCAPs.

---

## 3. Dataset Availability Status
- **Current Local Status:** **NOT AVAILABLE LOCALLY**.
- **Anti-Fabrication Enforcement:** In accordance with Phase 9 Section 4, benchmark performance metrics are NEVER synthesized or estimated. If dataset files are not discovered in the target directory, the runner terminates immediately with a clear missing-dataset error.
- **Validation Fallback:** Unit tests and regression suites run against strictly partitioned synthetic fixtures labeled `SYNTHETIC SOFTWARE VALIDATION`.

---

## 4. Input Formats & Discovery Engine
The CLI and discovery engine (`discover_dataset`) support:
- `--input-format csv`: Searches for single or multi-file `.csv` captures.
- `--input-format pcap`: Searches for `.pcap` and `.pcapng` raw capture traces.
- `--input-format auto`: Automatically detects whether CSV flow records or PCAP packet captures are present. If both or neither are found, reports clear status without ambiguous fallback.

---

## 5. Label Normalization & Canonical Taxonomy
To prevent label fragmentation, raw dataset labels are normalized into a unified 3-tuple structure:
1. `raw_label`: Preserved verbatim string for complete auditability.
2. `label_int`: Binary classification target (`0 = BENIGN`, `1 = ATTACK`).
3. `attack_class`: Canonical multi-class attack category mapped directly to the 15-class taxonomy defined in `src/shield_rca/models/label_maps.py`:
   - `BENIGN`, `Bot`, `DDoS`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `FTP-Patator`, `Heartbleed`, `Infiltration`, `PortScan`, `SSH-Patator`, `Web Attack - Brute Force`, `Web Attack - Sql Injection`, `Web Attack - XSS`.

---

## 6. Data Quality Audit & Sanitization
Before splitting or model execution, `audit_quality` executes a rigorous data-health check:
- **Total vs. Usable Records:** Detects header repetition lines (common in merged CIC CSVs) and scrubs them.
- **Deduplication:** Counts exact duplicate flow entries.
- **Missing & Infinite Values:** Identifies all `NaN`, `Inf`, and `-Inf` entries across the 78 canonical numerical columns, logging their exact count before sanitizing them to `0.0`.
- **Class Balance:** Reports exact benign sample count, attack count, attack ratio, and breakdown across attack categories.
- **Protocol Distribution:** Audits transport protocol counts (TCP=6, UDP=17, ICMP=1).
- **Output:** Emitted to `data_quality_report.json`.

---

## 7. Leakage-Safe Chronological Splitting
Network traffic exhibits severe temporal correlation. Random cross-validation leaks temporal patterns and yields artificially inflated metrics.

TRINETRA enforces strict **leakage-safe chronological splitting**:
- **Train Split:** First 70% of chronological flows.
- **Validation Split:** Middle 15% of chronological flows (used exclusively for hyperparameter and threshold selection).
- **Test Split:** Final 15% of chronological flows (held out strictly for final reporting).
- **Boundary Verification:** The engine guarantees:
  $$\max(t_{\text{train}}) \le \min(t_{\text{val}}) \le \max(t_{\text{val}}) \le \min(t_{\text{test}})$$
  Temporal sliding windows are formed strictly within split boundaries and never cross splits.

---

## 8. Feature Extraction & Schema Alignment
- **Canonical Flow Schema:** 78 numerical features aligned via `align_dataframe_to_schema`.
- **Aliases:** Uses `CIC_COLUMN_ALIASES` to map divergent column headers (e.g. `Tot Fwd Pkts` vs `Total Fwd Packets`) to canonical identifiers.
- **Coverage Ratio:** Computed as:
  $$\text{Coverage Ratio} = \frac{\text{Mapped Canonical Features}}{78}$$
  Unmapped features default to `0.0` and are reported transparently in `feature_report.json`.

---

## 9. Packet-Level Feature Coverage
When PCAPs are available, the Phase 8 packet-level feature extraction engine is invoked:
- Measures TTL, TCP retransmission counts, packet IAT, and packet IAT jitter across flows.
- If only CSV flow records are provided, the report explicitly states:
  > *"Packet-level features could not be evaluated from the flow-only input."*

---

## 10. Supervised Logistic Regression Real-Data Baseline
Reuses `LogisticRegressionBaseline` from Phase 6:
- Standardized via `StandardScaler` fitted **only on the training split**.
- Class imbalance handled via `class_weight="balanced"`.
- Decision threshold tuned on the validation split strictly without test leakage.
- Evaluated on the test split for:
  - Precision, Recall, F1 Score, False Positive Rate (FPR), ROC-AUC, PR-AUC, Confusion Matrix ($TP, TN, FP, FN$), and class support.
- Primary positive class: `ATTACK (1)`.

---

## 11. Multi-Horizon World Model Forecasting
Evaluates the continuous dynamics forecasting of the production World Model (`StateResidualDynamicsModel`) without retraining:
- Encodes temporal sliding windows into 280-dimensional Network State vectors:
  $$S_t = [\mathbf{f}_{\text{cont}} \in \mathbb{R}^{150}, \mathbf{f}_{\text{flags}} \in \mathbb{R}^{48}, \mathbf{f}_{\text{shape}} \in \mathbb{R}^{66}, \mathbf{f}_{\text{endpoints}} \in \mathbb{R}^{6}, \mathbf{f}_{\text{macro}} \in \mathbb{R}^{10}]$$
- Evaluates multi-horizon future forecasts:
  $$\hat{S}_{t+k} = S_t + \Delta_{\theta}(S_t, k, \Delta t), \quad k \in \{1, 3, 5\}$$
- Computes Mean Absolute Error (MAE) and Mean Squared Error (MSE) globally and across the 5 architectural feature groups.

---

## 12. World Model Baselines (Persistence & Linear Dynamics)
To ensure rigorous scientific benchmarking, the World Model is compared at each horizon $k \in \{1, 3, 5\}$ against:
1. **Persistence Baseline:**
   $$\hat{S}_{t+k}^{\text{pers}} = S_t \implies \text{Error} = \|S_{t+k} - S_t\|$$
2. **Linear Dynamics Baseline:**
   $$\hat{S}_{t+k}^{\text{lin}} = S_t \mathbf{W}_k + \mathbf{b}_k$$
   Fitted via Ridge Regression on training/validation state pairs.
- Relative improvement percentage is calculated:
  $$\text{Improvement over Persistence (\%)} = \left(1 - \frac{\text{MAE}_{\text{WM}}}{\text{MAE}_{\text{Persistence}}}\right) \times 100\%$$

---

## 13. Real Unseen Attack Generalization Protocol
When the dataset contains two or more distinct attack classes:
1. A designated attack category (e.g. `Bot` or `DDoS`) is held out completely.
2. The model and scaler are trained only on `BENIGN` + Known Attacks.
3. Performance is evaluated separately on Known Attack test samples and Unseen Attack test samples.
4. Degradation metrics are computed:
   $$\Delta F1 = F1_{\text{known}} - F1_{\text{unseen}}, \quad \Delta \text{Recall} = \text{Recall}_{\text{known}} - \text{Recall}_{\text{unseen}}, \quad \Delta \text{FPR} = \text{FPR}_{\text{unseen}} - \text{FPR}_{\text{known}}$$

---

## 14. Risk & Infiltration Estimation
Evaluates `PredictiveRiskEstimator` on predicted state trajectories:
- Computes normalized bounded risk score $R \in [0.0, 1.0]$ via group-weighted saturating exponential drift:
  $$R = 1.0 - \exp(-\lambda \cdot \text{weighted\_drift})$$
- Computes heuristic infiltration score $I \in [0.0, 1.0]$ (uncalibrated proxy for progressive network foothold).
- Evaluates multi-horizon trajectory escalation detection.

---

## 15. Attack Stage Progression Mapping
Evaluates `AttackStageMapper` to categorize future trajectory drift into defensive stages:
- Supported network-observable stages: `NORMAL`, `RECONNAISSANCE`, `INITIAL_ACCESS`, `INFILTRATION`, `IMPACT`, `UNKNOWN`.
- Anti-fabrication rule: If stage-level ground truth is absent in dataset labels, technique IDs are NOT fabricated; the output honestly reports `mitre_technique = None`.

---

## 16. State Delta Attribution (Explainability)
Evaluates `ForecastExplainabilityEngine`:
- Decomposes state trajectory drift $\hat{S}_{t+k} - S_t$ across all 280 dimensions.
- Identifies the top feature drivers and group-level contribution shares:
  $$\text{Relative Share}_g = \frac{\sum_{i \in \text{Group}_g} |\hat{S}_{t+k}[i] - S_t[i]|}{\sum_{j=1}^{280} |\hat{S}_{t+k}[j] - S_t[j]|}$$
- Method name: *"State Delta Attribution"* (explicitly disclosed as not SHAP).

---

## 17. Safe CyberHeal Dry-Run Integration
When predictive signals exceed gating thresholds:
- The event routes through `PredictiveTriggerGate` to determine `AUTO_TRIGGER`, `HUMAN_REVIEW`, or `NO_TRIGGER`.
- Eligible events are passed to `PredictiveIncidentAdapter`.
- **Absolute Safety Invariant:** All remediation execution occurs in `SIMULATION / DRY_RUN` mode. Zero iptables, Windows netsh, or service stop commands are executed against the machine.

---

## 18. Cryptographic Provenance Tracking
To ensure total reproducibility, `DatasetProvenance` computes and stores:
- Dataset name and format.
- Source filenames and individual file sizes.
- SHA-256 cryptographic hashes for every ingested file.
- Ingestion timestamp (UTC).
- Total record count.

---

## 19. Isolated Artifact Structure
All real-data benchmark outputs are written into dedicated, versioned directories:
```
artifacts/
    real_data_benchmark/
        <dataset_name>/
            <timestamp_or_exp_id>/
                provenance.json
                data_quality_report.json
                split_summary.json
                feature_report.json
                logistic_baseline_metrics.json
                world_model_forecasting_metrics.json
                unseen_attack_metrics.json
                risk_and_stage_metrics.json
                cyberheal_simulation_summary.json
                benchmark_manifest.json
                BENCHMARK_SUMMARY.md
```
Production model checkpoint `artifacts/world_model/best_model.pt` is strictly protected and never modified.

---

## 20. Known Limitations
1. **Flow vs. Packet Granularity:** Ingesting CSV flow records alone does not permit deep packet inspection (TTL distribution, TCP retransmission). Raw PCAP captures are required for Phase 8 packet signals.
2. **Heuristic Infiltration Score:** Infiltration scores are deterministic drift indicators, not calibrated probabilities.
3. **MITRE Stage Mapping:** Granular MITRE ATT&CK technique IDs cannot be inferred from flow data alone and are omitted unless verified ground-truth labels exist.
