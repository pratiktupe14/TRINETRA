# TRINETRA: SIH26153 Full Repository Gap Analysis
**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Internal Response Engine:** CyberHeal — Autonomous Self-Healing Response Engine  
**Problem Statement:** SIH26153 — AI based Network Attack Forecasting from Network Traffic Data  
**Audit Stage:** Phase 0 — Full Repository Audit  
**Date:** October 2026  

---

## 1. Executive Summary

This audit assesses the codebase at `D:\sih 2026\CyberHeal` against the target architecture of **TRINETRA** and the technical requirements of **SIH26153**.

TRINETRA establishes a closed-loop cyber defence pipeline:
$$\text{Observe} \longrightarrow \text{Understand} \longrightarrow \text{Predict} \longrightarrow \text{Explain} \longrightarrow \text{Act} \longrightarrow \text{Verify} \longrightarrow \text{Learn}$$

Where:
- The **Temporal World Model** answers: *"What may happen next?"*
- The **CyberHeal Engine** answers: *"What should we safely do now?"*

The repository contains high-grade telemetry ingestion, flow aggregation, 78 canonical statistical features, 280-dimensional Network State representation ($S_t$), $K$-step future-state dataset building, and a Pre-LayerNorm Residual Dynamics World Model ($\mathcal{W}_\theta$) with 62 passing automated unit and integration tests. It also contains the CyberHeal multi-agent response workflow (`ShieldLangGraph`), blockchain audit logging (`blockchain.py`), and a FastAPI + React dashboard.

However, the predictive forecasting output from the World Model is currently decoupled from CyberHeal's reactive multi-agent triggers, and key bridging components (trajectory risk estimation, MITRE attack stage mapping, explainability attribution, and the predictive trigger gate) remain to be implemented.

---

## 2. SIH Requirement Compliance Matrix

The table below catalogs all functional and architectural requirements. Every requirement is assessed with a strict status of **IMPLEMENTED**, **PARTIALLY IMPLEMENTED**, **MISSING**, or **NOT VERIFIED**.

| SIH Requirement | Existing Code | Status | Missing Work | File(s) |
|---|---|---|---|---|
| **Raw Network Traffic Ingestion** | Scapy streaming packet iterator `iter_packets()` and byte-buffer parser `iter_packets_from_bytes()`. | IMPLEMENTED | Add live background network interface sniffing loop and upload endpoint in FastAPI. | `src/shield_rca/telemetry/pcap_parser.py` |
| **PCAP / PCAPNG Support** | File reader supporting standard PCAP and PCAPNG capture formats via Scapy `PcapReader`. | IMPLEMENTED | None. Verified by unit tests on synthetic PCAP captures. | `src/shield_rca/telemetry/pcap_parser.py`, `tests/fixtures/generate_test_pcap.py` |
| **PyShark Ingestion** | None. Repository standardizes on Scapy for zero-C-extension streaming parsing. | MISSING | PyShark is not present; Scapy fulfills streaming ingestion without tshark installation dependency. | N/A |
| **Stateful 5-Tuple Flow Aggregation** | Bidirectional flow aggregator grouping packets by `(src_ip, dst_ip, src_port, dst_port, protocol)` with 120s active and 30s idle timeouts, FIN/RST teardown. | IMPLEMENTED | Add real-time flow eviction hook to trigger streaming inference. | `src/shield_rca/telemetry/flow_aggregator.py` |
| **Statistical Flow Feature Extraction** | Pure NumPy CICFlowMeter-compliant feature extractor extracting 78 numerical features (packet lengths, inter-arrival times, rates, TCP flags, active/idle bursts). | IMPLEMENTED | None for flow-level features. | `src/shield_rca/telemetry/flow_features.py`, `src/shield_rca/telemetry/feature_schema.py` |
| **Packet-Level Features (TTL, Retransmissions, Jitter)** | Dedicated 18-d packet feature schema (`packet_feature_schema.py`) and extractor (`packet_features.py`) computing IPv4 TTL/IPv6 hop limit, sequence-overlap TCP retransmissions, and IAT percentiles/jitter. | IMPLEMENTED | Verified by 32 unit tests in `test_packet_features.py`. | `src/shield_rca/telemetry/packet_features.py`, `src/shield_rca/telemetry/packet_feature_schema.py` |
| **Canonical Feature Schema & Validation** | Locked 78-feature schema (`feature_schema.py`), CIC column aliasing (`CIC_COLUMN_ALIASES`), NaN/Inf rejection, and physical bound checks. | IMPLEMENTED | None. Strictly validated with 11 unit tests. | `src/shield_rca/telemetry/feature_schema.py`, `src/shield_rca/telemetry/validator.py` |
| **Leakage-Safe Partitioning** | Chronological splitting (70% train, 15% val, 15% test) and scenario-aware splitting with zero temporal overlap verification. | IMPLEMENTED | None. Validated in `test_split.py`. | `src/shield_rca/telemetry/split.py` |
| **Temporal Sliding Window Batching** | 3D tensor batching $[N, T, F]$ ($T=20, F=78$) with sequence labeling rules and FIFO rolling `WindowRingBuffer`. | IMPLEMENTED | None. Validated in `test_windows.py`. | `src/shield_rca/telemetry/windows.py` |
| **Network State Representation ($S_t \in \mathbb{R}^{280}$)** | Deterministic 280-d state encoder partitioning windows into Continuous Dynamics (150), TCP Flags (48), Distribution Shape (66), Endpoints (6), and Macro Indicators (10). | IMPLEMENTED | None. Verified by 9 unit tests proving zero future leakage and strict determinism. | `src/shield_rca/network_state/state_schema.py`, `src/shield_rca/network_state/state_encoder.py` |
| **Future-State Dataset Generation** | Multi-horizon prediction dataset builder pairing $(S_t, S_{t+k})$ for $k \in \{1, 3, 5\}$ with exact wall-clock elapsed time $\Delta \tau$ and scenario boundary drop policy. | IMPLEMENTED | None. Verified by 8 unit tests in `test_future_dataset.py`. | `src/shield_rca/network_state/future_dataset.py`, `src/shield_rca/network_state/quality_report.py` |
| **Temporal World Model ($\mathcal{W}_\theta$)** | Pre-LayerNorm Residual Dynamics Model learning residual trajectory delta $\Delta \hat{S}(S_t, k, \Delta \tau)$ with GELU activations and skip connections (143,816 parameters). | IMPLEMENTED | Note: The model is a Residual Dynamics MLP, NOT an LSTM or Transformer. Keep terminology mathematically accurate. | `src/shield_rca/world_model/model.py` |
| **Time & Horizon Conditioning** | Discrete horizon embedding $k \in \{1, \dots, 16\} \to \mathbb{R}^{16}$ combined with continuous $\log(1 + \Delta \tau)$ 2-layer MLP projection $\to \mathbb{R}^{16}$. | IMPLEMENTED | None. Validated in `test_world_model.py`. | `src/shield_rca/world_model/time_conditioning.py` |
| **World Model Loss Function** | Huber loss ($\delta=1.0$) with per-group MAE slicing (Continuous, Flags, Shape, Endpoints, Macro). | IMPLEMENTED | None. Robust to burst outliers. | `src/shield_rca/world_model/loss.py` |
| **Baselines Comparison** | Persistence Baseline ($\hat{S}_{t+k} = S_t$) and Closed-Form Linear Ridge Dynamics Baseline. | IMPLEMENTED | Logistic regression is not implemented for state regression (linear ridge is used). | `src/shield_rca/world_model/baselines.py` |
| **K-Step Network State Forecasting** | Multi-horizon rollout forecasting for $k=1, 3, 5$. Model beats Linear Baseline by +12.67% ($k=3$) and +27.26% ($k=5$) on synthetic dynamics sanity check. | IMPLEMENTED | Real dataset retraining once full CIC-IDS2018 captures are downloaded. | `src/shield_rca/world_model/inference.py`, `artifacts/world_model/` |
| **Attack Risk & Infiltration Estimation** | Trajectory drift formulation exists in evaluator ($||\hat{S}_{t+k} - S_t||$), but no standalone risk scoring module. | PARTIALLY IMPLEMENTED | Create `RiskEstimator` module mapping forecasted state deltas to normalized risk scores ($0.0 \dots 1.0$) and infiltration alarms. | `src/shield_rca/world_model/risk_estimator.py` (to create) |
| **MITRE ATT&CK Stage Mapping** | Canonical 15-class attack label mapping (`label_maps.py`) exists; narrative strings in Causor. Formal stage progression mapping missing. | PARTIALLY IMPLEMENTED | Create `AttackStageMapper` module mapping predicted surges to stages: Reconnaissance $\to$ Initial Access $\to$ Infiltration $\to$ Impact. | `src/shield_rca/models/label_maps.py`, `src/shield_rca/world_model/stage_mapper.py` (to create) |
| **Explainable Forecasting & Attribution** | Evaluator tracks MAE across the 5 state feature groups, but runtime state-level attribution is missing. | PARTIALLY IMPLEMENTED | Create `ExplainabilityEngine` to compute top-$K$ feature drivers and group attribution for each forecasted anomaly without heavy SHAP dependencies. | `src/shield_rca/world_model/explainability.py` (to create) |
| **Predictive Trigger Gate** | Gating logic between World Model predictions and CyberHeal agent trigger is absent; Guardian evaluates reactive plan only. | MISSING | Create `PredictiveTriggerGate` enforcing: $\text{Prediction} \to \text{Risk} + \text{Confidence} + \text{Evidence} + \text{Asset Criticality} + \text{Action Safety} \to \text{Gate}$. | `src/shield_rca/world_model/trigger_gate.py` (to create) |
| **CyberHeal Multi-Agent Engine** | 10-step asynchronous agent orchestration (`ShieldLangGraph`): Intake $\to$ Sentinel $\to$ ThreatIntel $\to$ Causor $\to$ Planner $\to$ Guardian $\to$ Executor $\to$ Verifier $\to$ RLM $\to$ Scribe. | IMPLEMENTED | Wire Sentinel & Causor to consume real telemetry and World Model predictions instead of mock strings. | `src/shield_rca/langgraph_core.py` |
| **Guardian Safety & Blast-Radius Gate** | Guardian agent evaluates plan confidence ($<0.70 \to \text{HIGH RISK}$) and rejects unsafe actions before execution. | IMPLEMENTED | Extend Guardian to evaluate asset criticality and action blast-radius. | `src/shield_rca/langgraph_core.py` |
| **Safe Demo Remediation Default** | Executor agent attempts `netsh`/`iptables` and falls back to `blocked_ips.txt`. | PARTIALLY IMPLEMENTED | Enforce safe simulation mode by default so offline demo never executes destructive OS firewall changes unexpectedly. | `src/shield_rca/langgraph_core.py` |
| **Closed-Loop Verification** | Verifier agent validates post-remediation telemetry (currently emulates failure on retry 0, passes on retry 1). | PARTIALLY IMPLEMENTED | Connect Verifier to live `FlowAggregator` statistics to verify flow volume from blocked IP drops to zero. | `src/shield_rca/langgraph_core.py` |
| **Reflective Learning Module (RLM)** | RLM agent catches Guardian rejection or Verifier failure and pivots strategy (`EXPAND_BLOCK_RANGE`, `ADD_RATE_LIMIT`, `INCREASE_THRESHOLD`). | IMPLEMENTED | Connect RLM strategy pivots back to Sentinel detection thresholds. | `src/shield_rca/langgraph_core.py` |
| **Tamper-Evident Blockchain Audit** | SHA-256 block ledger (`SimpleBlockchain`) storing immutable event hashes, proof, and verification of fixes and resolutions. | IMPLEMENTED | None. Persists locally in `blockchain.json`. | `src/shield_rca/blockchain.py` |
| **Multiclass Evaluation Metrics** | Pure stateless calculation of Macro-F1, Weighted-F1, per-class P/R/F1, AUROC (One-vs-Rest), AUPR, and confusion matrix. | IMPLEMENTED | None. Tested in `test_metrics.py`. | `src/shield_rca/evaluation/metrics.py` |
| **Real CIC-IDS2018 / CTU-13 Benchmark** | Schema and label maps support CIC-IDS2017/2018 datasets. Synthetic sanity check benchmark documented. Real full-scale benchmark not executed. | NOT VERIFIED | Execute training and evaluation when real multi-gigabyte PCAP/CSV files are placed in workspace. Never fabricate benchmark metrics. | `configs/default.yaml`, `scripts/train_world_model.py` |
| **Held-Out / Unseen Attack Evaluation** | Zero-leakage unseen attack evaluation protocol (`unseen_eval.py`) supporting held-out category evaluation and generalization scoring. | IMPLEMENTED | Completed in Phase 7; verified by 12 unit tests in `test_unseen_attack.py`. | `src/shield_rca/evaluation/unseen_eval.py` |
| **TRINETRA Offline Demo Interface** | React + Vite + Tailwind dashboard with incident cards, log console, and simulation controls. Backend FastAPI with routes and worker fallback. | PARTIALLY IMPLEMENTED | Add forecasting views: multi-horizon trajectory curves ($k=1, 3, 5$), risk gauge, attack stage progression, and trigger gate review. | `src/App.jsx`, `configs/api/routes.py`, `configs/api/main.py` |

---

## 3. Search Terms Verification Summary

| Search Term | Found in Codebase | Notes / Exact Location |
|---|:---:|---|
| `telemetry` | YES | `src/shield_rca/telemetry/`, `docs/REAL_DATA_PIPELINE.md`, `configs/default.yaml` |
| `PCAP` / `PCAPNG` | YES | `src/shield_rca/telemetry/pcap_parser.py`, `tests/test_pcap_parser.py` |
| `Scapy` | YES | Used in `pcap_parser.py` (`PcapReader`, `IP`, `TCP`, `UDP`, `ICMP`) and `generate_test_pcap.py` |
| `PyShark` | NO | Not present. Scapy used for streaming parsing. |
| `flow` | YES | `flow_aggregator.py`, `flow_features.py`, `test_flow_features.py` |
| `feature` / `feature_schema` | YES | `feature_schema.py` (78 features), `test_feature_schema.py` |
| `window` | YES | `windows.py` (`WindowDataset`, `WindowRingBuffer`), `test_windows.py` |
| `dataset` | YES | `dataset_builder.py`, `future_dataset.py`, `test_dataset_builder.py` |
| `CIC` / `CIC-IDS` / `CSE-CIC` | YES | `feature_schema.py` (`CIC_COLUMN_ALIASES`), `label_maps.py`, `REAL_DATA_PIPELINE.md` |
| `CTU` | NO | Not in active code or docs. |
| `packet` / `packet_length` | YES | `pcap_parser.py` (`RawPacketRecord`), `flow_features.py` (`pkt_len_mean`, etc.) |
| `TTL` | NO | Not extracted by flow features. Identified as missing packet-level metric. |
| `TCP` / `flags` | YES | `pcap_parser.py`, `flow_features.py` (SYN, FIN, RST, PSH, ACK, URG, ECE, CWR) |
| `retransmission` | NO | Not computed in current flow features. |
| `network_state` / `S_t` | YES | `src/shield_rca/network_state/`, `state_schema.py` (280-d), `state_encoder.py` |
| `future_dataset` | YES | `src/shield_rca/network_state/future_dataset.py`, `test_future_dataset.py` |
| `world_model` | YES | `src/shield_rca/world_model/`, `model.py`, `trainer.py`, `evaluator.py`, `test_world_model.py` |
| `forecast` / `rollout` / `horizon` | YES | `future_dataset.py`, `time_conditioning.py`, `inference.py`, `WORLD_MODEL.md` |
| `k-step` / `future state` | YES | `future_dataset.py`, `model.py`, `FUTURE_STATE_DATASET.md` |
| `persistence` / `linear baseline` | YES | `src/shield_rca/world_model/baselines.py`, `test_world_model.py` |
| `risk` / `infiltration` | PARTIAL | Mathematical deviation in `evaluator.py` & docs; standalone risk engine missing. |
| `confidence` / `calibration` | PARTIAL | `confidence_score` in `AgentState` and Guardian check ($< 0.70$); calibration missing. |
| `MITRE` / `ATT&CK` / `attack stage` | PARTIAL | In `AUDIT.md`, `REUSE_PLAN.md`, `label_maps.py`; runtime stage mapper missing. |
| `SHAP` / `explainability` / `attribution` | PARTIAL | Sliced group MAE in `loss.py` & `evaluator.py`; runtime feature attribution missing. |
| `trigger` / `guardian` / `policy` | YES | `GuardianAgent` in `langgraph_core.py`; predictive trigger gate missing. |
| `CyberHeal` / `agent` | YES | `ShieldLangGraph` in `langgraph_core.py` (Intake, Sentinel, ThreatIntel, Causor, Planner, Guardian, Executor, Verifier, RLM, Scribe, Escalator). |
| `remediation` / `verification` | YES | `PlannerAgent`, `ExecutorAgent`, `VerifierAgent` in `langgraph_core.py`. |
| `logistic regression` | NO | Not used (Linear Ridge regression is used for baseline dynamics). |
| `precision` / `recall` / `F1` / `ROC` / `PR-AUC` | YES | `src/shield_rca/evaluation/metrics.py`, `test_metrics.py`. |
| `unseen attack` / `held-out attack` | NO | Protocol documented in audit; evaluation code missing. |
| `offline` / `demo` | YES | `DEMO_MODE=True` in `langgraph_core.py`, local `blockchain.json`, mock fallback in React UI. |
| `Streamlit` | NO | Not present. Frontend is React + Vite + Tailwind CSS. |
| `CLI` | YES | `scripts/run_pipeline.py`, `scripts/train_world_model.py`. |

---

## 4. Current Test Suite Baseline

Executed test command:
```powershell
pytest -v
```

Execution environment:
- Platform: Windows (win32)
- Python Version: 3.12.10
- Pytest Version: 9.1.1
- Pluggy: 1.6.0
- Plugins: anyio-4.14.2

### Test Suite Results:
- **Phase 0 Baseline Tests:** 62
- **Phase 1-8 Progressive Tests:** 151
- **Phase 9 Real Benchmark Tests:** 20
- **Total Tests Passing:** 233
- **Failed:** 0
- **Skipped:** 0
- **Pass Rate:** 100%

---

## 5. Phase 9 SIH Compliance Status Update

- **Real Dataset Benchmark Infrastructure:** **IMPLEMENTED & VERIFIED**
  - Dataset discovery, automatic format detection (`csv`, `pcap`, `auto`).
  - Adapters for CSE-CIC-IDS2018 (`CicIds2018Adapter`) and CTU-13 (`Ctu13Adapter`).
  - Cryptographic provenance tracking with SHA-256 file hashing (`compute_file_sha256`).
  - Automated Data Quality Audit (`DataQualityReport`).
  - Leakage-safe chronological splitting ($70/15/15$) preserving temporal ordering.
  - Aligned Logistic Regression supervised baseline (`LogisticRegressionBaseline`).
  - Multi-horizon World Model continuous forecasting ($k \in \{1, 3, 5\}$) vs Persistence and Linear Dynamics baselines.
  - Zero-day holdout unseen attack generalization protocol.
  - Deterministic state delta attribution explainability.
  - Zero-risk simulated dry-run CyberHeal remediation gating.
  - Production model checkpoint protection (`artifacts/world_model/best_model.pt` strictly untouched).
- **Real Dataset Benchmark Execution on Real Local Files:** **NOT VERIFIED (PENDING LOCAL DATA PLACEMENT)**
  - Local dataset files are not present in workspace (`DATASET STATUS: NOT AVAILABLE LOCALLY`).
  - In strict compliance with the anti-fabrication mandate, metrics on real data were NOT fabricated.
  - One-command execution ready via `python scripts/run_real_data_benchmark.py`.

