# STEP 1 — REUSE & INTEGRATION PLAN: CYBERHEAL AI

**Date:** September 2026  
**Status:** Architectural Plan (Pre-Implementation Audit)  
**Target Repository:** `CyberHeal` (Location: `d:\sih 2026\CyberHeal`)  
**Temporary Source:** `network-attack-research` & `AI-NIDS-Transformer-XGBoost` (Scratch space)  

---

## Architectural Principle & Isolation Boundary

The temporary research repository serves **solely** as an algorithm and data processing reference. Under no circumstances will the research repository be merged into git history, copied as a directory, or imported as an external package. 

All production components will be cleanly integrated directly into the `CyberHeal` namespace (`src/shield_rca/` and `configs/api/`).

```
[Raw Network Traffic / PCAP] 
          │
          ▼
┌────────────────────────────────────────┐
│  src/shield_rca/telemetry/             │
│  ├── pcap_parser.py    (Scapy stream)  │
│  ├── flow_aggregator.py (5-tuple/state)│
│  └── flow_features.py  (80-stats)      │
└──────────────────┬─────────────────────┘
                   │ Extracted Feature Vector [80-d]
                   ▼
┌────────────────────────────────────────┐
│  src/shield_rca/models/                │
│  ├── xgboost_detector.py / hybrid.py   │
│  └── label_maps.py (15 attack classes) │
└──────────────────┬─────────────────────┘
                   │ Real Anomaly & Threat Classification + Probabilities
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  src/shield_rca/langgraph_core.py (ShieldLangGraph)         │
│  ├── IntakeAgent    (Normalizes Network & Host Telemetry)   │
│  ├── SentinelAgent  (Real ML Inference & Severity Scoring)   │
│  ├── CausorAgent    (MITRE ATT&CK Attribution & Graph RCA)   │
│  ├── GuardianAgent  (Blast Radius & Calibrated Risk Gating) │
│  ├── PlannerAgent   (Automated Mitigation Planning)          │
│  ├── ExecutorAgent  (OS Firewall Netsh/Iptables Enforcement)│
│  ├── VerifierAgent  (Live Flow Delta Verification)          │
│  ├── RLM Agent      (Reflective Learning & Threshold Tuning)│
│  └── ScribeAgent    (Tamper-Evident Blockchain Proof Mining)│
└─────────────────────────────────────────────────────────────┘
```

---

## A. CODE TO REUSE (Minimal Modification)

These modules contain pure algorithms, mathematically rigorous feature extraction, or canonical data mappings that can be integrated directly into CyberHeal with only namespace and import adjustments.

### 1. `src/data/flow_features.py` (from `network-attack-research`)
- **Target Location in CyberHeal:** `src/shield_rca/telemetry/flow_features.py`
- **What It Does:** Calculates ~80 statistical flow features (packet counts, packet length distribution, inter-arrival times, TCP flag counts, byte/packet rates, bulk transfer statistics, active/idle period metrics) compliant with the CICFlowMeter standard.
- **Why Reuse:** Pure mathematical functions with no external dependencies other than NumPy. Rewriting this from scratch would risk subtle errors in statistical formulas (e.g. sample variance vs population variance in packet lengths).
- **Adaptation Needed:** Ensure input accepts CyberHeal's `FlowRecord` dataclass and returns a clean NumPy array formatted for model input.

### 2. `data/metadata/label_maps.json` (from `network-attack-research`)
- **Target Location in CyberHeal:** `src/shield_rca/models/label_maps.py`
- **What It Does:** Canonical taxonomy mapping integer prediction indices ($0 \dots 14$) to human-readable attack labels (`BENIGN`, `DoS Hulk`, `PortScan`, `DDoS`, `FTP-Patator`, `SSH-Patator`, `Botnet`, etc.).
- **Why Reuse:** Guarantees that predictions from pre-trained weights accurately translate to standardized attack descriptions across the dashboard and incident tickets.
- **Adaptation Needed:** Convert JSON into a static Python dictionary with helper functions (`get_attack_name(id)`, `is_attack(id)`).

### 3. `scripts/generate_test_pcap.py` (from `network-attack-research`)
- **Target Location in CyberHeal:** `tests/fixtures/generate_test_pcap.py`
- **What It Does:** Generates a lightweight, deterministic synthetic PCAP containing valid TCP, UDP, and ICMP frames without requiring multi-gigabyte external downloads.
- **Why Reuse:** Essential for automated integration tests in CyberHeal CI/CD to verify that the packet decoder, flow aggregator, and feature extraction pipeline work end-to-end on any operating system.
- **Adaptation Needed:** Refactor from a CLI script into a reusable pytest fixture module.

### 4. `src/evaluation/metrics.py` (from `network-attack-research`)
- **Target Location in CyberHeal:** `src/shield_rca/evaluation/metrics.py`
- **What It Does:** Computes Macro-F1, Weighted-F1, per-class Precision/Recall, AUROC (One-vs-Rest), and AUPR without any plotting or GUI side-effects.
- **Why Reuse:** Used by the `VerifierAgent` and offline model evaluation scripts to benchmark detection quality objectively.
- **Adaptation Needed:** Retain only the pure metric calculation functions; remove dependencies on external reporting scripts.

### 5. `src/data/zeek_mapper.py` (from `AI-NIDS-Transformer-XGBoost`)
- **Target Location in CyberHeal:** `src/shield_rca/telemetry/zeek_mapper.py`
- **What It Does:** Enriches raw flow statistics with Zeek protocol family classifications (`conn`, `dns`, `http`, `ssl`, `files`) and connection state heuristics (`SF`, `S0`, `RSTO`, `OTH`).
- **Why Reuse:** Lightweight ($<150$ lines), pure Python logic that gives Causor and Sentinel semantic protocol context from raw numeric port numbers.
- **Adaptation Needed:** Provide support for single flow dictionaries in addition to batch pandas Series.

---

## B. CODE TO MODIFY (Refactor & Interface Alignment)

These modules contain valuable architecture but require refactoring to fit CyberHeal's asynchronous multi-agent framework, Pydantic state schemas, and production deployment constraints.

### 1. `src/data/pcap_parser.py` (from `network-attack-research`)
- **Target Location in CyberHeal:** `src/shield_rca/telemetry/pcap_parser.py`
- **Modifications Required:**
  - Decouple from CLI scripts and relative path imports.
  - Wrap Scapy import in an optional conditional block with a clear user diagnostic if Npcap/libpcap is missing on Windows.
  - Add an in-memory byte-stream parser (`parse_pcap_bytes`) so FastAPI can accept direct file uploads via `POST /telemetry/upload-pcap` without first requiring disk writes.
  - Expose packet yield callbacks for real-time progress tracing in the UI.

### 2. `src/data/flow_aggregator.py` (from `network-attack-research`)
- **Target Location in CyberHeal:** `src/shield_rca/telemetry/flow_aggregator.py`
- **Modifications Required:**
  - Add a thread-safe sliding window cache (`LiveFlowTable`) to support continuous live packet streaming where packets arrive asynchronously over WebSockets or network interfaces.
  - Add eviction callbacks so completed/timed-out flows automatically trigger Sentinel evaluation.
  - Convert `FlowRecord` dataclasses into Pydantic models compatible with FastAPI JSON serialization.

### 3. `src/tabular/models.py` (from `network-attack-research`) & `src/models/hybrid.py` (from `AI-NIDS`)
- **Target Location in CyberHeal:** `src/shield_rca/models/detector.py`
- **Modifications Required:**
  - Consolidate model serving into a single, clean `NetworkAttackDetector` class.
  - Strip training dependencies (PyTorch Lightning, TabNet, CatBoost) to keep CyberHeal's production environment fast and lean.
  - Provide a lightweight inference runtime utilizing serialized XGBoost models (`.json` or `.joblib`) with calibrated probability output.
  - Provide an optional PyTorch Transformer embedding extractor if GPU/deep learning dependencies are available, falling back gracefully to pure XGBoost when on CPU.

### 4. `src/rag/threat_rag.py` & `knowledge_base.py` (from `AI-NIDS`)
- **Target Location in CyberHeal:** `src/shield_rca/telemetry/mitre_enricher.py`
- **Modifications Required:**
  - Adapt the MITRE ATT&CK knowledge base to directly feed `CausorAgent` and `ThreatIntelAgent`.
  - Replace mandatory ChromaDB dependency with a lightweight in-memory cosine similarity search or local ChromaDB fallback so the system functions even in low-resource environments.
  - Connect threat narrative generation to CyberHeal's existing LLM provider configurations.

### 5. `src/shield_rca/langgraph_core.py` (inside `CyberHeal`)
- **Target Location in CyberHeal:** `src/shield_rca/langgraph_core.py`
- **Modifications Required:**
  - Extend `AgentState` schema to include `network_flow_features`, `predicted_attack_family`, `attack_probabilities`, and `mitre_technique_id`.
  - Upgrade `SentinelAgent`: Replace the placeholder mock anomaly with real inference calls to `NetworkAttackDetector`.
  - Upgrade `CausorAgent`: Incorporate model feature importances (e.g. high SYN flag count $\to$ SYN flood / PortScan) and MITRE ATT&CK technique IDs into root-cause findings.
  - Upgrade `VerifierAgent`: Query `FlowAggregator` post-mitigation to confirm packet rates from the blocked IP drop to zero.

### 6. `configs/api/ingestion.py` & `configs/api/routes.py` (inside `CyberHeal`)
- **Target Location in CyberHeal:** `configs/api/ingestion.py` & `configs/api/routes.py`
- **Modifications Required:**
  - Add PCAP ingestion route: `POST /api/telemetry/upload-pcap`.
  - Add live replay simulation route: `POST /api/simulate/network-attack` which streams real flows from a benchmark sample into the agent pipeline.
  - Update `LogIngestor` to monitor network telemetry alongside OS event logs.

---

## C. CODE TO BUILD FROM SCRATCH

These are the new integration components necessary to bind the network detection engine into CyberHeal's multi-agent autonomous framework.

### 1. `src/shield_rca/telemetry/network_sensor.py` (Live Telemetry Bridge)
- **Role:** Bridges raw packet reception to `FlowAggregator` and notifies `SentinelAgent`.
- **Capabilities:**
  - Manages active sniffing on local network interfaces (optional live mode via Scapy `sniff()`).
  - Implements a thread-safe ring buffer of active flows.
  - Exposes an event-driven hook that fires `on_flow_complete(flow)` to push flows into the detection pipeline without polling.

### 2. `src/shield_rca/models/model_loader.py` (Safe Artifact Manager)
- **Role:** Manages loading, caching, and versioning of trained model weights, feature scalers, and label encoders.
- **Capabilities:**
  - Loads pre-trained model artifacts from a dedicated `artifacts/models/` directory inside CyberHeal.
  - Verifies artifact hashes and schema dimensions before executing inference to prevent runtime crashes.
  - Implements mock fallback predictors if model weight files are absent, allowing the system to run seamlessly in development or test modes.

### 3. `src/shield_rca/telemetry/pcap_endpoint.py` (FastAPI PCAP Ingestion Handler)
- **Role:** Handles multipart PCAP file uploads from the SOC dashboard.
- **Capabilities:**
  - Validates uploaded `.pcap` / `.pcapng` headers.
  - Invokes `PCAPParser` and `FlowAggregator` in background tasks.
  - Generates real-time SSE / WebSocket progress events to update the frontend progress bar during large PCAP analyses.

### 4. `tests/test_flow_pipeline.py` (Automated Pipeline Test Suite)
- **Role:** Automated unit and regression test suite using pytest.
- **Capabilities:**
  - Tests synthetic packet parsing $\to$ flow aggregation $\to$ feature extraction $\to$ model prediction.
  - Validates that `SentinelAgent` produces correct alert structures on malicious flows.
  - Tests Guardian risk gating on low vs high confidence classifications.

---

## D. CODE NOT TO USE (Exclude from CyberHeal)

The following files and directories from the research repository are experimental, baseline comparisons, documentation for papers, or heavy training pipelines. They **must be excluded** to prevent technical debt and dependency bloat in CyberHeal:

| Path in Research Repo | Reason for Exclusion |
| :--- | :--- |
| `src/tabular/train.py`, `src/timeseries/train.py` | CyberHeal is an operational runtime platform, not a training orchestrator. Retraining belongs in offline experimental environments. |
| `src/timeseries/models_baselines.py` (InceptionTime, TCN) | Academic baseline models used solely for comparative thesis benchmarks. |
| `src/evaluation/reporting.py` | Contains heavy matplotlib/seaborn plotting code. CyberHeal's frontend renders all charts dynamically using React and Tailwind. |
| `scripts/run_cross_dataset_eval.py`, `run_train_*.py` | Ad-hoc CLI scripts for running thesis grid-searches. |
| `docs/*.md` (LYCOS analysis, Distrinet notes, thesis writeups) | Research notes specific to academic dataset comparisons. |
| `reports/`, `reports/paper/` (LaTeX paper drafts) | Academic publication materials irrelevant to CyberHeal's production codebase. |
| `notebooks/` (Jupyter Exploratory Data Analysis notebooks) | Unstructured experimentation artifacts. |
| `push_and_update_pr.sh`, `scripts/pr_automerge.sh` | Research repo maintenance shell scripts. |
| Heavy dependencies (`pytorch-tabnet`, `catboost`, `optuna`, `hydra-core`) | Keep CyberHeal's dependency tree lightweight and stable. |

---

## Proposed Package Structure in CyberHeal

Upon completion of Step 2 (integration), the new files inside `d:\sih 2026\CyberHeal` will follow this strict structure:

```
CyberHeal/
├── AUDIT.md                          <-- Created in Step 1
├── REUSE_PLAN.md                     <-- Created in Step 1
├── configs/
│   ├── api/
│   │   ├── routes.py                 <-- Extended with PCAP/streaming endpoints
│   │   ├── ingestion.py              <-- Extended with network telemetry hooks
│   │   └── ...
│   └── default.yaml                  <-- Extended with model paths & feature config
├── src/
│   ├── shield_rca/
│   │   ├── telemetry/                <-- [NEW MODULAR PACKAGE]
│   │   │   ├── __init__.py
│   │   │   ├── pcap_parser.py        <-- Adapted from research
│   │   │   ├── flow_aggregator.py    <-- Adapted from research
│   │   │   ├── flow_features.py      <-- Adapted from research
│   │   │   ├── zeek_mapper.py        <-- Adapted from research
│   │   │   └── stream_replay.py      <-- Adapted for dashboard simulations
│   │   ├── models/                   <-- [NEW MODULAR PACKAGE]
│   │   │   ├── __init__.py
│   │   │   ├── detector.py           <-- Lightweight XGBoost inference runtime
│   │   │   ├── label_maps.py         <-- Canonical 15-class attack taxonomy
│   │   │   └── model_loader.py       <-- Weight loading with safe mock fallback
│   │   ├── agents.py                 <-- Upgraded Sentinel & Causor logic
│   │   ├── langgraph_core.py         <-- Enhanced state schema & real ML loop
│   │   └── blockchain.py             <-- Unchanged (SHA-256 tamper-evident log)
│   └── ...
└── tests/
    ├── fixtures/
    │   └── generate_test_pcap.py     <-- Deterministic synthetic PCAP generator
    └── test_flow_pipeline.py         <-- Automated regression test suite
```

---

## Implementation Readiness Checklist

- [x] Temporary research repositories inspected in isolated scratch directory.
- [x] File-by-file audit completed and documented in `AUDIT.md`.
- [x] Clear 4-tier separation established in `REUSE_PLAN.md`.
- [x] Zero modifications made to CyberHeal source code during Step 1.
- [x] Research repository marked for isolation (no git merge, no submodule).
- [x] Clean architecture ready for user approval before moving to Step 2.
