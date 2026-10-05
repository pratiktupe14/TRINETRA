# STEP 1 — TEMPORARY RESEARCH REPOSITORY & CYBERHEAL AI AUDIT

**Date:** September 2026  
**Auditor:** Antigravity AI Pair Programmer  
**Primary Repository:** `CyberHeal` (Location: `d:\sih 2026\CyberHeal`)  
**Temporary Research Repositories:** `network-attack-research` & `AI-NIDS-Transformer-XGBoost` (Extracted in temporary scratch environment)  

---

## Executive Summary & Boundary Notice

1. **Isolation Guarantee:** The research repository is strictly an offline source of algorithms, data structures, and feature engineering patterns. It will **never** be merged, copied wholesale, or linked as a submodule/dependency into CyberHeal.
2. **Target Destination:** All productionized components will reside natively inside `d:\sih 2026\CyberHeal` under modular packages (e.g., `src/shield_rca/telemetry/` and `src/shield_rca/models/`).
3. **No CyberHeal Source Modifications in Step 1:** No implementation code has been touched in CyberHeal. This document establishes the factual technical baseline and feasibility analysis.

---

## Comprehensive Component Audit Table

| Component | Source Repository | Source File | What It Does | Can Reuse? | How It Should Be Integrated Into CyberHeal | Changes Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PCAP Packet Parsing** | `network-attack-research` | `src/data/pcap_parser.py` | Streams raw PCAP bytes using Scapy `PcapReader`, filters protocols (TCP, UDP, ICMP), decodes headers into memory-efficient `RawPacketRecord` slots dataclasses. | **YES (High Value)** | Place in `src/shield_rca/telemetry/pcap_parser.py`. Used by Sentinel to ingest real network capture files or live packet buffers uploaded via API/dashboard. | Decouple from CLI scripts; wrap Scapy dependency in try/except fallback; add live socket/interface sniffing support; remove hardcoded CIC capture assumptions. |
| **Flow Aggregation & Timeouts** | `network-attack-research` | `src/data/flow_aggregator.py` | Groups packets by 5-tuple into bidirectional flows with active (120s) and idle (30s) timeouts, TCP RST/FIN close policies, and active/idle duration tracking. | **YES (High Value)** | Place in `src/shield_rca/telemetry/flow_aggregator.py`. Acts as the stateful flow accumulator between packet reception and anomaly detection. | Add thread-safe flow cache for real-time packet-by-packet streaming; allow configurable sliding timeouts per interface; export clean dicts or Pydantic flow schemas. |
| **Statistical Feature Extraction** | `network-attack-research` | `src/data/flow_features.py` | Computes ~80 statistical features matching CICFlowMeter (packet counts, inter-arrival times, length stats, TCP flags, byte/packet rates, bulk metrics). | **YES (Critical Core)** | Place in `src/shield_rca/telemetry/flow_features.py`. Directly feeds real network flow feature vectors into Sentinel's ML models. | Replace ad-hoc dict output with structured float arrays aligned to trained model feature orders; eliminate unneeded identity string fields during inference; add vectorised batching. |
| **Attack Label Mapping & Resolution** | `network-attack-research` | `src/data/labeling.py`, `src/data/label_merger.py` | Resolves microsecond vs second timestamps; performs 5-tuple exact matching, NAT translation adjustment, and integer encoding across 15 attack classes. | **YES (Selective)** | Place canonical label mapping into `src/shield_rca/models/label_maps.py`. Used during offline evaluation, model fine-tuning, and for mapping model class IDs to human names in RCA. | Strip offline CSV merge logic (`by_timestamp`, `by_source_ip`); retain the canonical 15-class dictionary and integer-to-name decoder for the SOC dashboard. |
| **Attack Schedule Label Generator** | `network-attack-research` | `scripts/generate_cicids2017_labels.py` | Creates attack labels from official CIC-IDS2017 timing docs. | **NO (Research-only)** | Do not import into CyberHeal runtime. Retain in scratch research environment for dataset regeneration only. | None. Excluded from production codebase. |
| **Temporal Sliding Windowing** | `network-attack-research` | `src/data/windows.py` | Constructs `[N, T, F]` 3D tensors from flow DataFrames with window size (e.g. 64), stride (16), and seq2seq per-flow label targets (`[N, T]`). | **YES (Medium Value)** | Adapt into `src/shield_rca/telemetry/window_buffer.py` for multi-flow sequence and temporal anomaly detection. | Simplify into an in-memory ring buffer (FIFO queue of the last $W$ flows per host/subnet) rather than a heavy disk-backed Parquet pipeline. |
| **Sessionizer Windowing** | `AI-NIDS` | `src/data/sessionizer.py` | Groups consecutive flows into 20-flow sessions per source host, labels sessions, and flags attack participation. | **YES (High Value)** | Combine with `windows.py` inside `src/shield_rca/telemetry/window_buffer.py` for session-level temporal context. | Adapt from static DataFrame slicing to an event-driven rolling buffer suitable for live API ingestion. |
| **Zeek Telemetry Mapper** | `AI-NIDS` | `src/data/zeek_mapper.py` | Maps raw flows into 5 Zeek protocol views (conn, dns, http, ssl, files) and infers connection state (SF, S0, RSTO, OTH). | **YES (High Value)** | Place in `src/shield_rca/telemetry/zeek_mapper.py`. Enriches raw flows with cross-protocol contextual tags for Causor and Sentinel. | Pure python function, lightweight and zero external dependencies. Convert pandas-only logic to accept single flow dictionaries as well as batches. |
| **Temporal Transformer Models** | `network-attack-research` | `src/timeseries/models_transformer.py` | Implements `PatchTST`, `ITransformer`, and `TransformerSeq` (`(B, T, F) -> (B, T, C)`) for sequence intrusion detection. | **YES (Selective)** | Place `TransformerSeq` or `PatchTST` inference runtime into `src/shield_rca/models/transformer_detector.py`. | Strip training/loss boilerplate; provide clean `.predict_proba()` method; load lightweight ONNX or PyTorch weights; provide CPU inference fallback. |
| **Self-Supervised Zeek Transformer + XGBoost Hybrid** | `AI-NIDS` | `src/models/hybrid.py`, `src/models/transformer.py` | Pre-trained Transformer encoder extracts 256-d CLS session embeddings; XGBoost performs multi-class classification (Config C: embeddings + stats). | **YES (Recommended Architecture)** | Integrate model loader & inference engine into `src/shield_rca/models/hybrid_detector.py`. | Package pre-trained weights and scaler into artifacts; write lightweight inference class with zero training dependencies. |
| **Baseline Time-Series Models** | `network-attack-research` | `src/timeseries/models_baselines.py` | InceptionTime and TCN (Temporal Convolutional Networks). | **NO (Research Baseline)** | Not needed in CyberHeal. The Transformer or XGBoost models achieve superior detection. | Exclude to prevent dependency bloat. |
| **Tabular Classifiers & Stacking** | `network-attack-research` | `src/tabular/models.py` | Dynamic wrappers for XGBoost, LightGBM, CatBoost, Random Forest, and StackingClassifier. | **YES (Selective - XGBoost)** | Keep XGBoost classifier interface inside `src/shield_rca/models/xgboost_detector.py`. | Isolate XGBoost runtime; remove dependencies on CatBoost/TabNet/LightGBM unless explicitly selected. |
| **Training Pipelines** | `network-attack-research` | `src/tabular/train.py`, `src/timeseries/train.py` | Full PyTorch & Scikit-learn training loops with mixed precision, AdamW, and cosine annealing. | **NO (Offline Research Only)** | Keep offline in training scripts directory if custom retraining is ever needed, but NOT in main agent execution path. | Decouple completely from FastAPI/LangGraph runtime. |
| **Multiclass Evaluation Metrics** | `network-attack-research` | `src/evaluation/metrics.py` | Computes Macro-F1, Weighted-F1, per-class P/R/F1, AUROC, AUPR, and Confusion Matrix. | **YES (High Value for Verifier/QA)** | Place in `src/shield_rca/evaluation/metrics.py`. Used by VerifierAgent to objectively score model confidence and validation passes. | Keep pure stateless NumPy/Scikit-learn math; remove heavy plotting dependencies from runtime path. |
| **Evaluation Reporting & Plotting** | `network-attack-research` | `src/evaluation/reporting.py` | Generates matplotlib/seaborn confusion matrices, ROC/PR curves, and LaTeX/Markdown model cards. | **NO (Heavy UI Dependencies)** | Do not put Matplotlib/Seaborn in production backend. CyberHeal frontend renders charts natively via React. | Exclude plotting logic. |
| **Synthetic PCAP Generator** | `network-attack-research` | `scripts/generate_test_pcap.py` | Creates a small deterministic PCAP containing TCP, UDP, and ICMP traffic anchored to 2017-07-04. | **YES (High Value for CI/Testing)** | Move to `tests/fixtures/generate_test_pcap.py` in CyberHeal for regression and unit testing of the ingestion engine. | Convert script arguments into a callable test fixture function. |
| **Threat Knowledge Base & RAG** | `AI-NIDS` | `src/rag/threat_rag.py`, `src/rag/knowledge_base.py` | ChromaDB vector store of MITRE ATT&CK tactics/techniques with Groq Llama-3.3 narrative generation. | **YES (High Value for Causor & ThreatIntel)** | Integrate into `src/shield_rca/agents.py` (ThreatIntelAgent / CausorAgent) to enrich detections with real MITRE ATT&CK techniques. | Decouple ChromaDB (allow lightweight in-memory cosine search or Chroma persistent client); configure Groq API key alongside existing LLM configs. |
| **CSV Replay & Streaming Simulator** | `AI-NIDS` | `src/streaming/csv_replay.py` | Replays real flow CSVs row-by-row into the detection pipeline, measuring throughput, TPR, and FPR. | **YES (High Value for Demo & Testing)** | Place in `src/shield_rca/telemetry/stream_replay.py` to power the dashboard simulation engine (`/simulate/replay-stream`). | Bind to FastAPI background tasks / WebSockets for live frontend streaming visual updates. |
| **Existing CyberHeal Backend API** | `CyberHeal` | `configs/api/main.py`, `configs/api/routes.py` | FastAPI application serving `/run`, `/incidents`, `/logs`, `/simulate/*`, `/trace/*`, and `/webhook/security-event`. | **RETAIN & EXTEND (Core)** | Central entry point. Add `/upload/pcap` and `/stream/network` endpoints. | Move from `configs/api/` into standard `src/cyberheal/api/` or keep in `configs/api/` while updating imports. |
| **Existing Log Ingestor & Simulation** | `CyberHeal` | `configs/api/ingestion.py` | Polls Windows EventLog ID 4625 / Linux `/var/log/auth.log`, manages incident trigger lifecycle, emits Omium traces. | **RETAIN & EXTEND (Core)** | Integrate `FlowAggregator` and `PCAPParser` so it triggers on network telemetry in addition to OS log events. | Unify OS log ingestion with network packet ingestion under a polymorphic TelemetryIngestor. |
| **Existing Celery/Threading Worker** | `CyberHeal` | `configs/api/worker.py` | Dispatches incident investigation jobs to Celery Redis queue with synchronous thread fallback. | **RETAIN (Core)** | Unchanged. Handles pipeline queueing reliably across operating systems. | None. |
| **Existing Omium Distributed Tracing** | `CyberHeal` | `configs/api/tracing.py` | Emits spans to local JSON `traces/` and sends payloads to Omium Cloud API. | **RETAIN (Core)** | Add trace spans for PCAP parsing, feature extraction, and ML model inference. | Add span types: `PCAP_INGESTION`, `FLOW_EXTRACTION`, `MODEL_INFERENCE`. |
| **Existing Blockchain Audit Log** | `CyberHeal` | `src/shield_rca/blockchain.py` | SHA-256 block hashing and persistent tamper-evident blockchain stored in `blockchain.json`. | **RETAIN (Core)** | Cryptographically anchors fix execution and incident resolution hashes. | None. |
| **Existing Agent Workflow (LangGraph Core)** | `CyberHeal` | `src/shield_rca/langgraph_core.py` | 10-step autonomous agent loop: Intake -> Sentinel -> ThreatIntel -> Causor -> Planner -> Guardian -> Executor -> Verifier -> RLM -> Scribe. | **RETAIN & ENHANCE (Core)** | Replace mock Sentinel/Causor logic with real ML model inference and MITRE ATT&CK contextualization. | Update `SentinelAgent` to run feature extraction + ML classification; update `AgentState` to hold network flow features and model confidence distributions. |
| **Existing Sentinel Agent** | `CyberHeal` | `src/shield_rca/langgraph_core.py`, `src/shield_rca/agents.py` | Detects anomalies. Currently hardcoded (`traffic_volume: 1500.5, value: 99.9`). | **MODIFY & UPGRADE (Critical)** | Upgrade `SentinelAgent` to execute real-time model scoring (XGBoost / Transformer) on network features. | Wire in `HybridDetector` or `ZeekXGBoost` model weights; output genuine class probabilities and feature attributions. |
| **Existing Causor Agent** | `CyberHeal` | `src/shield_rca/langgraph_core.py`, `src/shield_rca/agents.py` | Performs RCA on dependency graph with basic IP reputation lookup. | **MODIFY & UPGRADE (High Value)** | Wire in MITRE ATT&CK RAG knowledge base from research repo to attribute exact attack techniques and affected protocol layers. | Connect to `ThreatRAG` / Groq analyst for deep root cause explanations. |
| **Existing Guardian Agent** | `CyberHeal` | `src/shield_rca/langgraph_core.py` | Evaluates blast-radius and confidence threshold ($< 0.70 \to \text{HIGH RISK}$). | **RETAIN (Core Safety)** | Leverages calibrated model prediction probabilities from the ML pipeline to make mathematically grounded safety gating decisions. | Calibrate probability inputs using research evaluation metrics. |
| **Existing Executor Agent** | `CyberHeal` | `src/shield_rca/langgraph_core.py` | Executes `tool_block_ip` (`netsh`/`iptables`) and `tool_restart_service`, mines hashes to blockchain. | **RETAIN (Core Action)** | Target IP is extracted directly from malicious network flow 5-tuple identified by Sentinel. | Ensure clean extraction of attacker IP from flow key metadata. |
| **Existing Verifier Agent** | `CyberHeal` | `src/shield_rca/langgraph_core.py` | Evaluates post-fix telemetry to determine if threat subsided. Currently toggles on retry count. | **MODIFY & UPGRADE** | Use temporal flow buffer to verify that packet/flow volume from the flagged attacker drops to zero post-firewall rule. | Connect Verifier to live `FlowAggregator` statistics. |
| **Existing Reflective Learning (RLM)** | `CyberHeal` | `src/shield_rca/langgraph_core.py` | Dynamic strategy pivoting upon failure (`EXPAND_BLOCK_RANGE`, `ADD_RATE_LIMIT`, `INCREASE_THRESHOLD`). | **RETAIN & ENHANCE** | Adjust ML detection sensitivity / classification thresholds dynamically based on RLM retry directives. | Add threshold adjustment hook into Sentinel's ML inference. |
| **Existing Scribe Agent** | `CyberHeal` | `src/shield_rca/langgraph_core.py` | Compiles audit evidence and records final resolution hash to blockchain. | **RETAIN (Core Audit)** | Incorporate ML feature importance, model version, and MITRE IDs into final audit bundle. | Add model metadata into Scribe's final audit JSON. |
| **Existing Frontend Dashboard** | `CyberHeal` | `src/App.jsx`, `src/components/*` | React + Vite + Tailwind UI visualizing pipeline cards, active logs, Supabase incidents, and manual triggers. | **RETAIN & POLISH** | Visualizes live network telemetry, flow rates, top attacking IPs, and ML confidence bars. | Add PCAP file dropzone and live network flow monitor widget to UI. |
| **Existing CyberHeal Tests** | `CyberHeal` | `test_omium.py`, `scripts/run_pipeline.py` | Smoke scripts for pipeline and tracing. No unit test harness. | **AUGMENT WITH PYTEST** | Adopt pytest test suite inspired by research repo's testing patterns. | Add `tests/test_flow_pipeline.py` and `tests/test_agent_workflow.py`. |

---

## Detailed Audit of network-attack-research Components (Items 1 - 10)

### 1. Real Network Data
- **Artifacts:** Raw PCAP files from Canadian Institute for Cybersecurity (CIC-IDS2017 & CSE-CIC-IDS2018).
- **Structure:** 5 daily PCAP captures covering benign traffic and 14 distinct attack classes (FTP/SSH Patator, DoS Slowloris, Slowhttptest, Hulk, GoldenEye, Heartbleed, Web Attacks, Infiltration, Botnet, PortScan, DDoS).
- **Download Automation:** `scripts/download_datasets.sh` automates fetching raw data.
- **Deterministic Fixture:** `scripts/generate_test_pcap.py` produces a synthetic PCAP with TCP, UDP, ICMP packet streams for zero-download local regression tests.

### 2. PCAP Parsing
- **Module:** `src/data/pcap_parser.py`
- **Methodology:** Scapy `PcapReader` used as an iterative generator (`iter_packets`). Decodes Layer 2/3/4 headers into compact `RawPacketRecord` slots dataclasses (`timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `ip_length`, `payload_length`, `tcp_flags`, `ip_version`).
- **Characteristics:** Memory footprint is $O(1)$ with respect to PCAP file size. Filters out non-IP packets and handles malformed frames gracefully without throwing unhandled exceptions.

### 3. Flow Aggregation
- **Module:** `src/data/flow_aggregator.py`
- **Methodology:** Implements 5-tuple bidirectional aggregation (`FlowKey = (src_ip, dst_ip, src_port, dst_port, protocol)`).
- **Timeouts:**
  - Active timeout: 120.0s (forces flow termination for long-lived streams).
  - Idle timeout: 30.0s (closes flow when packet inter-arrival exceeds threshold).
- **State Handling:** Tracks forward and backward packet sequences, computes active and idle durations, detects TCP FIN/RST termination flags, and outputs `FlowRecord` objects.

### 4. Label Processing
- **Modules:** `src/data/labeling.py`, `src/data/label_merger.py`, `scripts/generate_cicids2017_labels.py`
- **Methodology:**
  - `generate_cicids2017_labels.py` builds canonical CSV labels directly from published attack timetables.
  - `label_merger.py` addresses the microsecond ($10^{15}$) vs second ($10^9$) timestamp mismatch inherent in CIC datasets.
  - Supports 3 merge strategies: `by_timestamp`, `by_source_ip`, and `by_exact_match` (5-tuple match accounting for victim IPs, attacker subnets `172.16.0.1`, and selective port/protocol rules).
  - `labeling.py` encodes string labels to integers using `data/metadata/label_maps.json`.

### 5. Feature Extraction
- **Module:** `src/data/flow_features.py`
- **Methodology:** Computes ~80 statistical features compliant with the CICFlowMeter standard:
  1. *Counts & Bytes:* `tot_fwd_pkts`, `tot_bwd_pkts`, `tot_fwd_bytes`, `tot_bwd_bytes`.
  2. *Inter-Arrival Times (IAT):* Mean, standard deviation, max, and min for forward, backward, and combined flow directions.
  3. *Packet Length Stats:* Mean, standard deviation, variance, max, min, skewness.
  4. *TCP Flags:* Individual counts and aggregate distribution of FIN, SYN, RST, PSH, ACK, URG, ECE, CWR flags.
  5. *Rates & Ratios:* `flow_bytes_per_s`, `flow_packets_per_s`, `fwd_bytes_per_s`, `bwd_bytes_per_s`, `down_up_ratio`.
  6. *Header Lengths:* Forward and backward header byte sums.
  7. *Subflows & Bulk Transfer:* Subflow packet/byte averages and bulk rate estimates.
  8. *Active/Idle Segment Stats:* Mean, std, max, min of active burst durations and idle intervals.

### 6. Temporal Windowing
- **Modules:** `network-attack-research/src/data/windows.py` and `AI-NIDS/src/data/sessionizer.py`
- **Methodology:**
  - `windows.py`: Slices chronologically sorted flows into sliding 3D tensor windows of shape `[N, T, F]` ($N$ windows, $T$ timesteps, $F$ features). Provides window-level labeling (`majority`, `last`, `any_attack`) or per-flow sequence labeling (`seq2seq: true`, shape `[N, T]` with `-100` padding mask).
  - `sessionizer.py`: Organizes flows into fixed-size 20-flow sessions per host, tracking attack presence and cross-flow transition patterns.

### 7. Existing Temporal ML Models
- **Modules:** `src/timeseries/models_transformer.py`, `src/timeseries/models_baselines.py`, `AI-NIDS/src/models/transformer.py`, `AI-NIDS/src/models/hybrid.py`
- **Architectures:**
  - `PatchTST`: Patched time-series transformer (Nie et al. 2023) using tokenized temporal patches with self-attention.
  - `ITransformer`: Inverted transformer (Liu et al. 2024) computing self-attention over the feature variate dimension.
  - `TransformerSeq`: Encoder-only transformer outputting per-flow sequence predictions `(B, T, num_classes)`.
  - `ZeekTransformerEncoder`: 4-layer, 8-head self-supervised transformer pre-trained on masked feature modeling, producing 256-d CLS session embeddings.
  - `HybridIDS`: Fuses transformer CLS embeddings with statistical session metrics and classifies with an XGBoost head.

### 8. Training Utilities
- **Modules:** `src/tabular/train.py`, `src/timeseries/train.py`, `scripts/run_train_tabular.py`, `scripts/run_train_timeseries.py`
- **Methodology:** Automated stratified splitting, class weight rebalancing for rare attacks (e.g. Heartbleed, Infiltration), learning rate scheduling (OneCycleLR / CosineAnnealing), mixed precision FP16/AMP, early stopping based on validation macro-F1, and model checkpointing.

### 9. Evaluation Utilities
- **Modules:** `src/evaluation/metrics.py`, `src/evaluation/reporting.py`
- **Methodology:** Full multi-class evaluation computing Macro-F1, Weighted-F1, per-class Precision/Recall/F1/Support, One-vs-Rest AUROC, and Precision-Recall AUC (AUPR). Reporting module provides confusion matrix visualization and model card generation.

### 10. Dataset / Configuration Utilities
- **Modules:** `configs/data.yaml`, `configs/features.yaml`, `configs/split.yaml`, `configs/tabular_models.yaml`, `configs/timeseries_models.yaml`, `src/common/`
- **Methodology:** Structured YAML configurations via OmegaConf/Hydra, deterministic random seeding (`seed_everything`), custom loggers, and type contracts (`FlowKey`, `WindowDataset`, `SplitData`).

---

## Detailed Audit of CyberHeal AI Components (Items 1 - 15)

### 1. Existing Backend
- **Files:** `configs/api/main.py`, `configs/api/routes.py`, `configs/api/models.py`, `configs/api/auth.py`, `run_backend.py`, `backend/main.py`
- **Role:** High-performance asynchronous REST API powered by FastAPI.
- **Functionality:**
  - Mounts API router at root with CORS wildcard for local development.
  - Lifecycle hook `@app.on_event("startup")` starts background log ingestion daemon and runs crash recovery check on `pipeline_state.json`.
  - Exposes endpoints: `/run` (trigger pipeline), `/incidents` (list), `/incidents/{id}` (full graph details), `/logs/{id}` (log trail), `/simulate/*` (brute force, service outage, edge anomaly), `/trace/*` (export/replay), `/webhook/security-event` (external alerts).
  - Secondary `backend/main.py` provides real-time Windows event polling over WebSockets.

### 2. Existing Frontend
- **Files:** `src/App.jsx`, `src/components/AgentCard.jsx`, `src/components/LogConsole.jsx`, `src/index.css`, `tailwind.config.js`, `src/supabaseClient.js`
- **Role:** Interactive Cyber SOC Operational Console.
- **Functionality:**
  - Modern dark-mode interface built with React, Vite, and Tailwind CSS.
  - Visualizes the 8-stage multi-agent pipeline with step cards showing execution status, latencies, and active agent badges.
  - Interactive Log Console displaying color-coded agent logs with real-time autoscroll, search, and severity filters.
  - Trigger controls for manual simulation injections (Brute Force, Service Outage, Edge Anomaly) and trace file replay.
  - Connects to Supabase for live incident state and falls back to mock demo incident if offline.

### 3. Existing Agent Workflow
- **Files:** `src/shield_rca/langgraph_core.py`, `src/shield_rca/agents.py`
- **Role:** Orchestrates multi-agent execution using state machine patterns.
- **State Machine Schema (`AgentState`):**
  - Holds `incident_id`, `alert_payload`, `normalized_alert`, `anomalies`, `root_cause`, `confidence_score`, `remediation_plan`, `guardian_approved`, `risk_score`, `execution_results`, `verification_passed`, `status`, `retry_count`, `max_retries`, `escalated`, `rlm_history`, `threat_intel`, `current_step`.
- **Workflow Order:**
  `Intake` $\to$ `Sentinel` $\parallel$ `ThreatIntel` $\to$ `Causor` $\to$ `Planner` $\to$ `Guardian` $\to$ `Executor` $\to$ `Verifier` $\to$ `(RLM Retry Loop)` $\to$ `Scribe` / `Escalator`.
- **Persistence:** Serializes intermediate state to `pipeline_state.json` at every phase boundary to permit transparent crash recovery.

### 4. Trigger Event
- **Files:** `configs/api/routes.py`, `configs/api/ingestion.py`
- **Role:** Entry point into incident processing.
- **Mechanisms:**
  - Webhook: `POST /webhook/security-event` receives JSON payloads (e.g. `{ "ip": "1.2.3.4", "failedAttempts": 10 }`).
  - OS Polling: `LogIngestor.poll_logs()` queries Windows Security Log Event ID 4625 (failed logons) or Linux `/var/log/auth.log`.
  - Simulation: `SimulationEngine` generates simulated brute force, outage, and edge anomalies.
  - Trace Replay: Replays archived JSON run traces.
  - Calls `LogIngestor._trigger_incident()`, creates record in Supabase `incidents`, traces `WEBHOOK_RECEIVED` / `ALERT_DETECTED` / `TICKET_CREATED`, and spawns asynchronous worker thread.

### 5. Alert Handling
- **Files:** `src/shield_rca/langgraph_core.py` (`IntakeAgent`)
- **Role:** Intake, normalization, and deduplication.
- **Functionality:** Ingests raw heterogeneous trigger payloads and formats them into a normalized dictionary (`source`, `metric`, `severity`, `ip`). Assigns initial severity and passes normalized alert to Sentinel.

### 6. Diagnosis
- **Files:** `src/shield_rca/langgraph_core.py` (`SentinelAgent`), `src/shield_rca/agents.py`
- **Role:** Telemetry anomaly detection.
- **Current State:** Placeholder logic. Produces a hardcoded anomaly (`metric: traffic`, `value: 99.9`, `source: node-A`) and logs to Supabase `anomalies` table.
- **Integration Target:** This is the primary insertion point for the research repository's feature extraction and ML model scoring engine.

### 7. RCA (Root-Cause Analysis)
- **Files:** `src/shield_rca/langgraph_core.py` (`CausorAgent`, `ThreatIntelAgent`)
- **Role:** Dependency graph root-cause isolation.
- **Functionality:** Queries `ThreatIntelAgent` (which calls `http://ip-api.com/json/{ip}` to retrieve ISP and country data), calculates a heuristic confidence score, and formulates root-cause descriptions (e.g. "High-risk attack verified from [country] ([isp])"). Records finding in Supabase `root_causes`.

### 8. Guardian
- **Files:** `src/shield_rca/langgraph_core.py` (`GuardianAgent`)
- **Role:** Autonomous safety gating and blast-radius verification.
- **Functionality:** Assesses the proposed remediation plan before execution. If `confidence_score < 0.70`, flags `risk_score = "HIGH"`, sets `guardian_approved = False`, rejects the plan, and diverts the execution flow into the Reflective Learning Module (`RLM`). If approved, authorizes execution.

### 9. Fix / Remediation
- **Files:** `src/shield_rca/langgraph_core.py` (`PlannerAgent`, `ExecutorAgent`), `src/shield_rca/blockchain.py`
- **Role:** Plan generation and execution.
- **Functionality:**
  - `PlannerAgent` creates targeted actions: `block_ip` on target IP or `restart_service` on affected node.
  - `ExecutorAgent` invokes tool registry:
    - `tool_block_ip`: Executes `netsh advfirewall firewall add rule` on Windows or `iptables -A INPUT -s ... -j DROP` on Linux. Falls back to writing to `blocked_ips.txt`.
    - `tool_restart_service`: Generates local incident report.
  - Cryptographic Verification: For each executed action, calls `blockchain_sys.store_event_hash()` to append an immutable record to `blockchain.json`. Updates Supabase `actions` table.

### 10. Verification
- **Files:** `src/shield_rca/langgraph_core.py` (`VerifierAgent`)
- **Role:** Post-execution telemetry validation.
- **Current State:** Emulates an initial failure (`retry_count == 0` $\to$ FAILED) to exercise the RLM retry loop, followed by passing on attempt 1.
- **Integration Target:** Connect to real post-fix flow metrics from `FlowAggregator` to verify packet drops.

### 11. Resolution
- **Files:** `src/shield_rca/langgraph_core.py` (`ScribeAgent`, `EscalationAgent`)
- **Role:** Incident finalization and audit packaging.
- **Functionality:**
  - If verified, `ScribeAgent` marks incident `RESOLVED` in Supabase, anchors the resolution hash into `blockchain.json`, and outputs the complete evidence trail.
  - If retries exceed maximum limit (`max_retries = 3`), `EscalationAgent` marks incident `ESCALATED` for human operator intervention.

### 12. RLM / Self-Learning
- **Files:** `src/shield_rca/langgraph_core.py` (`ReflectiveLearningAgent`)
- **Role:** Autonomous reflection, adaptive re-planning, and retry loop.
- **Functionality:** Triggered whenever Guardian rejects a plan or Verifier detects active anomalies. Increments retry counter, analyzes previous failure reason, and generates an updated mitigation strategy (e.g. `block_ip` $\to$ `EXPAND_BLOCK_RANGE` $\to$ `ADD_RATE_LIMIT` $\to$ `INCREASE_THRESHOLD`). Resets the pipeline step to `DETECTION` and re-executes.

### 13. API Integration Points
- **External Integrations:**
  - Supabase Database: Remote PostgreSQL tables `incidents`, `anomalies`, `root_causes`, `actions`, `logs`.
  - Omium Tracing SDK: Emits distributed tracing spans to `https://api.omium.ai/api/v1/traces/ingest` and stores local JSON files in `traces/`.
  - IP Reputation: Calls `http://ip-api.com/json/{ip}`.
  - Celery / Redis: Background distributed task queue on `redis://localhost:6379/0` with local thread fallback.
  - Operating System: Win32 Event Log API, `netsh advfirewall`, Linux `iptables`.

### 14. Configuration System
- **Files:** `configs/default.yaml`, `.env`
- **Structure:** YAML-based configuration controlling agent parameters (thresholds, graph depth, policy strictness), dataset paths, output directories, and environment keys (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `OMIUM_API_KEY`).

### 15. Existing Tests
- **Files:** `scripts/run_pipeline.py --mode smoke`, `test_omium.py`, `test_omium2.py`, `test_decorator.py`
- **Current Coverage:** Manual scripts verifying Omium SDK span emission and a CLI sequential smoke test of the agent pipeline.
- **Gap:** Lacks automated regression tests (pytest), mock DB fixtures, and state machine validation.

---

## Technical Feasibility Assessment & Risk Matrix

| Area | Feasibility | Key Risk | Mitigation |
| :--- | :--- | :--- | :--- |
| **PCAP Parsing & Scapy** | High | Scapy can be slow on very large (>1GB) files and requires Npcap on Windows. | Use chunked generator streaming via `PcapReader`; provide synthetic CSV/socket stream fallbacks for testing. |
| **80-Feature Extraction** | High | High computational cost per flow if implemented inefficiently. | Keep functions stateless with NumPy vectorisation; extract only the core feature subset required by the trained model. |
| **Model Serving in FastAPI** | High | Loading heavy PyTorch models can increase backend startup time and memory footprint. | Default to trained XGBoost tabular model (compact, sub-millisecond CPU inference); load PyTorch Transformer lazily on demand. |
| **State Machine Continuity** | High | Changing `AgentState` schema might break frontend serialization or checkpoint recovery. | Ensure all new network fields (`flow_features`, `mitre_attack_id`, `pcap_filename`) have default values (`Optional`). |
| **No Repository Bleed** | Guaranteed | Research codebase accidentally commingling with production CyberHeal. | Temporary scratch extraction directory only; clean manual port of isolated pure Python functions into CyberHeal namespaces. |
