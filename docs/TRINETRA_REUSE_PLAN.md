# TRINETRA: Architectural Reuse & Integration Plan
**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Internal Response Engine:** CyberHeal — Autonomous Self-Healing Response Engine  
**Problem Statement:** SIH26153 — AI based Network Attack Forecasting from Network Traffic Data  
**Audit Stage:** Phase 0 — Full Repository Audit  
**Date:** October 2026  

---

## 1. Architectural Strategy & Non-Negotiable Boundaries

TRINETRA integrates predictive network attack forecasting with safe autonomous remediation. The end-to-end conceptual flow is:

$$\text{Observe} \longrightarrow \text{Understand} \longrightarrow \text{Predict} \longrightarrow \text{Explain} \longrightarrow \text{Act} \longrightarrow \text{Verify} \longrightarrow \text{Learn}$$

### Core Responsibility Separation
- **Temporal World Model ($\mathcal{W}_\theta$):** Answers *"What may happen next?"* by forecasting future 280-dimensional Network State vectors $\hat{S}_{t+k}$.
- **CyberHeal Response Engine (`ShieldLangGraph`):** Answers *"What should we safely do now?"* by formulating, checking, executing, and verifying constrained remediation plans.
- **Predictive Trigger Gate:** Serves as the critical safety bridge between the two. **Prediction must NEVER directly execute remediation.** All remediation must pass through:
  $$\text{Prediction} \longrightarrow \text{Risk} + \text{Confidence} + \text{Evidence} + \text{Asset Criticality} + \text{Action Safety} \longrightarrow \text{Trigger Gate} \longrightarrow \text{Automatic CyberHeal OR Human Review}$$

### Architecture Exclusions
In accordance with system design constraints, the following generic subsystems will **NOT** be introduced as major architecture blocks:
- No standalone database cluster (SQLite/in-memory/local JSON state is used for offline demo).
- No API Gateway or Redis message broker dependency for offline demo execution (worker thread fallback is preserved).
- No cloud authentication, telemetry shipping, or cloud AI services.
- No separate generic threat intelligence or generic SOC dashboard blocks.
- The demonstration interface is strictly designated: **TRINETRA Offline Demo Interface**.

---

## 2. What Will Be REUSED (Unmodified Preservation)

The following components represent high-value, mathematically verified, working functionality that will be reused directly without altering their public API signatures or internal logic:

### A. Network Telemetry Ingestion & Extraction Core
1. **`src/shield_rca/telemetry/pcap_parser.py`**:
   - `RawPacketRecord` slots dataclass.
   - `iter_packets()` and `iter_packets_from_bytes()` generators using Scapy `PcapReader`.
   - Protocol filtering (TCP, UDP, ICMP) and memory-efficient streaming.
2. **`src/shield_rca/telemetry/flow_aggregator.py`**:
   - `FlowKey` canonical 5-tuple dataclass.
   - `FlowRecord` bidirectional state accumulator.
   - `aggregate_flows()` generator with active (120s) and idle (30s) timeouts, and TCP FIN/RST teardown handling.
3. **`src/shield_rca/telemetry/flow_features.py`**:
   - `extract_flow_features()` computing 78 numerical features matching the CICFlowMeter standard (packet length distributions, inter-arrival times, byte/packet rates, TCP flag counters, active/idle period stats).
   - `extract_feature_vector()` returning deterministic float arrays.
4. **`src/shield_rca/telemetry/feature_schema.py`**:
   - `CANONICAL_FEATURES` ($F=78$) in strict alphabetical order.
   - `CIC_COLUMN_ALIASES` for automated CSV header normalization.
   - `validate_feature_vector()` and `align_dataframe_to_schema()`.
5. **`src/shield_rca/telemetry/validator.py`**:
   - `DataValidator` enforcing non-empty, non-negative, finite, and non-duplicate constraints.
6. **`src/shield_rca/telemetry/split.py`**:
   - `chronological_split()` and `scenario_aware_split()` with zero temporal overlap verification.
7. **`src/shield_rca/telemetry/windows.py`**:
   - `WindowDataset` and `build_timeseries_windows()` constructing 3D tensors $[N, T, F]$.
   - `WindowRingBuffer` for in-memory rolling sequence accumulation.
8. **`src/shield_rca/telemetry/dataset_builder.py`**:
   - Unified ingestion builder converting raw PCAP or flow CSVs into ready `WindowBatch` objects.

### B. Network State Representation ($S_t$)
1. **`src/shield_rca/network_state/state_schema.py`**:
   - `STATE_FEATURE_NAMES` ($S=280$) across 5 groups: Continuous Dynamics (150), TCP Flags (48), Distribution Shape (66), Endpoints (6), and Macro Indicators (10).
2. **`src/shield_rca/network_state/state_encoder.py`**:
   - `NetworkState` and `NetworkStateBatch` containers.
   - `NetworkStateEncoder` producing deterministic 280-d states from historical observation windows $[T \times 78]$ with zero future target leakage.
3. **`src/shield_rca/network_state/future_dataset.py`**:
   - `FutureStateDatasetBuilder` creating non-leaking state-target pairs $(S_t, S_{t+k})$ for $k \in \{1, 3, 5\}$ with exact continuous $\Delta \tau$ wall-clock calculation.
4. **`src/shield_rca/network_state/preprocessor.py`**:
   - `NetworkStatePreprocessor` fitting StandardScaler/RobustScaler strictly on training state vectors.
5. **`src/shield_rca/network_state/quality_report.py`**:
   - Automated dataset health and boundary audit generator.

### C. Temporal World Model Core ($\mathcal{W}_\theta$)
1. **`src/shield_rca/world_model/model.py`**:
   - `StateResidualDynamicsModel` architecture with Pre-LayerNorm residual dynamics blocks and GELU activations.
2. **`src/shield_rca/world_model/time_conditioning.py`**:
   - `TimeHorizonConditioner` projecting discrete horizon embeddings $k \to \mathbb{R}^{16}$ and continuous $\log(1 + \Delta \tau) \to \mathbb{R}^{16}$.
3. **`src/shield_rca/world_model/loss.py`**:
   - `WorldModelLoss` computing robust Huber regression loss with group slice breakdown.
4. **`src/shield_rca/world_model/baselines.py`**:
   - `PersistenceBaseline` ($\hat{S}_{t+k} = S_t$) and `LinearDynamicsBaseline` (closed-form ridge regression).
5. **`src/shield_rca/world_model/checkpoint.py`**:
   - Serialization and deserialization of model weights, architecture configs, and training manifests.
6. **`src/shield_rca/world_model/evaluator.py`**:
   - `WorldModelEvaluator` running multi-horizon test benchmarks against baselines.
7. **`src/shield_rca/world_model/inference.py`**:
   - `WorldModelInference` serving PyTorch inference with input bounds checking.
8. **`artifacts/world_model/`**:
   - Serialized pre-trained model weights (`best_model.pt`), architecture config (`model_config.json`), model manifest (`model_manifest.json`), and fitted scaler (`state_scaler.joblib`).

### D. Auditability, Evaluation & Taxonomy
1. **`src/shield_rca/blockchain.py`**:
   - `SimpleBlockchain` providing SHA-256 block hashing, proof linking, and tamper verification in `blockchain.json`.
2. **`src/shield_rca/evaluation/metrics.py`**:
   - Pure NumPy/Scikit-learn multiclass evaluation (Macro-F1, Weighted-F1, AUROC OVR, AUPR, confusion matrix).
3. **`src/shield_rca/models/label_maps.py`**:
   - Canonical 15-class attack taxonomy and string alias mappings.
4. **`tests/` (All 62 Unit and Integration Tests)**:
   - Full preservation of the existing test suite in `tests/test_*.py` and `tests/fixtures/generate_test_pcap.py`.

---

## 3. What Will Be EXTENDED (Interface Enhancement & Integration)

The following existing modules will be enhanced to bind the predictive World Model to the CyberHeal multi-agent framework without breaking backward compatibility:

### 1. `src/shield_rca/langgraph_core.py` (CyberHeal Multi-Agent Engine)
- **Extend `AgentState` schema:**
  Add optional fields with safe defaults:
  - `forecasted_state: Optional[List[float]] = None`
  - `trajectory_risk_score: float = 0.0`
  - `attack_stage: str = "NORMAL"`
  - `top_predictive_features: List[Dict[str, Any]] = []`
  - `asset_criticality: str = "MEDIUM"`
  - `trigger_gate_decision: str = "UNEVALUATED"`
- **Upgrade `SentinelAgent`:**
  Allow ingesting live/simulated Network State and World Model predictions to compute continuous anomaly scores rather than relying on hardcoded static metrics.
- **Upgrade `CausorAgent`:**
  Incorporate predicted feature group surges (e.g. unique destination port surges $\to$ PortScan/Reconnaissance; SYN/RST ratio surges $\to$ DoS/DDoS) and map to MITRE ATT&CK stages.
- **Upgrade `GuardianAgent`:**
  Incorporate asset criticality, action blast radius, and prediction confidence into the approval gate.
- **Upgrade `ExecutorAgent`:**
  Add `SAFE_DEMO_MODE=True` flag (default True). In Safe Demo Mode, OS commands (`netsh`, `iptables`) are simulated and audited to local files, guaranteeing that running the demo will never disrupt the host operating system's network configuration unless explicitly enabled.
- **Upgrade `VerifierAgent`:**
  Verify mitigation by checking that flow traffic from the target attacker drops to zero in post-remediation telemetry.

### 2. `configs/api/routes.py` (FastAPI Endpoints)
- **Add Predictive Forecasting Endpoints:**
  - `POST /api/forecast/predict`: Accepts a current Network State $[280]$ or computes it from telemetry, runs $\mathcal{W}_\theta$, and returns predicted states $[\hat{S}_{t+1}, \hat{S}_{t+3}, \hat{S}_{t+5}]$, trajectory risk score, attack stage, and top feature drivers.
  - `GET /api/forecast/latest`: Returns the most recent forecasting state for live dashboard polling.
  - `POST /api/telemetry/upload-pcap`: Ingests an uploaded PCAP file, parses packets, aggregates flows, and generates Network States.
- **Add Trigger Gate Management:**
  - `POST /api/gate/override`: Permits human operator review and manual approval/rejection of gated remediation plans.
- **Unify WebSocket Streaming:**
  - Consolidate real-time log and incident WebSocket streaming directly into `configs/api/routes.py` (superseding the experimental `backend/main.py`).

### 3. `configs/api/ingestion.py` (Continuous Streaming Bridge)
- Connect the end-to-end autonomous loop:
  $$\text{PCAP / Flow Replay} \longrightarrow \text{Flow Aggregator} \longrightarrow \text{State Encoder } (S_t) \longrightarrow \text{World Model } (\hat{S}_{t+k}) \longrightarrow \text{Risk Estimator} \longrightarrow \text{Trigger Gate} \longrightarrow \text{CyberHeal}$$
- Support replaying benchmark sample flows in simulated real-time for live demonstrations.

### 4. `src/App.jsx` & Dashboard Components (TRINETRA Offline Demo Interface)
- **Enhance UI to display:**
  1. **Multi-Horizon Forecast Trajectory:** Visual comparison of current state $S_t$ vs predicted future states $\hat{S}_{t+1}, \hat{S}_{t+3}, \hat{S}_{t+5}$.
  2. **Predictive Risk Gauge:** Real-time visual gauge displaying continuous risk score ($0.0 \dots 1.0$) with color-coded severity bands.
  3. **Attack Stage Progression Bar:** Indicating progression through MITRE stages (Normal $\to$ Reconnaissance $\to$ Initial Access $\to$ Infiltration $\to$ Impact).
  4. **Predictive Explainability Card:** Listing top feature drivers contributing to the forecasted anomaly (e.g. `syn_flag_cnt`, `endpoint_unique_dst_ports`, `macro_mean_byte_rate`).
  5. **Predictive Trigger Gate Modal:** Providing human-in-the-loop inspection and manual override for high blast-radius remediation plans.
  6. **100% Offline Resilience:** Clean fallback when Supabase is not configured, ensuring seamless offline local presentation.

---

## 4. What Will Be NEWLY CREATED

The following new modules will be authored from scratch to complete the predictive intelligence and gating requirements:

### 1. `src/shield_rca/world_model/risk_estimator.py`
- **Purpose:** Converts future Network State trajectories into continuous, bounded risk scores without fabricating probabilities.
- **Functions:**
  - `compute_trajectory_drift(current_state, predicted_state)`: Calculates normalized Euclidean and Huber deviations across state groups.
  - `estimate_multi_horizon_risk(current_state, predicted_states_dict, horizons)`: Combines multi-step trajectory deviations into an aggregate risk index $\in [0.0, 1.0]$.
  - `estimate_infiltration_probability(drift_score, baseline_variance)`: Sigmoidal mapping calibrated to baseline variance.

### 2. `src/shield_rca/world_model/stage_mapper.py`
- **Purpose:** Maps forecasted state dynamics to standardized attack progression stages aligned with MITRE ATT&CK.
- **Stages:**
  - `STAGE_NORMAL`: Steady-state baseline traffic.
  - `STAGE_RECONNAISSANCE`: Surges in port dispersion, unique source/destination ports, connection resets (PortScan, Network Discovery).
  - `STAGE_INITIAL_ACCESS`: Surges in failed login attempts, authentication burst patterns, unusual external endpoints (Brute Force).
  - `STAGE_INFILTRATION`: Unusual payload sizes, asymmetric down/up ratios, abnormal protocol distributions (Botnet, C2, Lateral Movement).
  - `STAGE_IMPACT`: Massive spikes in byte rates, packet rates, SYN flag ratios, or connection teardowns (DDoS, DoS Hulk, Slowloris).

### 3. `src/shield_rca/world_model/explainability.py`
- **Purpose:** Provides SOC analyst transparency into *why* the World Model forecasted an attack surge.
- **Functions:**
  - `explain_forecast_drift(current_state, predicted_state, top_k=5)`:
    - Calculates absolute and relative deviations for all 280 state dimensions.
    - Aggregates attribution into the 5 feature groups (Continuous Dynamics, TCP Flags, Distribution Shape, Endpoints, Macro Indicators).
    - Identifies top-$K$ individual state features driving the predicted surge.

### 4. `src/shield_rca/world_model/trigger_gate.py`
- **Purpose:** Authoritative safety gate controlling the transition between prediction and autonomous remediation.
- **Logic:**
  ```python
  class PredictiveTriggerGate:
      def evaluate(self, risk_score, confidence, evidence_bundle, asset_criticality, action_type):
          # Safety Rule 1: Low risk -> Suppress
          if risk_score < self.threshold_low:
              return GateDecision(status="SUPPRESS", reason="Risk below trigger threshold")
          
          # Safety Rule 2: High blast radius action on critical asset -> Require Human Review
          if asset_criticality == "CRITICAL" and action_type in ["isolate_subnet", "block_gateway"]:
              return GateDecision(status="REQUIRE_HUMAN_REVIEW", reason="High blast-radius action on critical asset")
          
          # Safety Rule 3: High risk + High confidence + Contained blast radius -> Automatic CyberHeal
          if risk_score >= self.threshold_high and confidence >= self.min_confidence:
              return GateDecision(status="AUTOMATIC_CYBERHEAL", reason="High risk verified with sufficient confidence")
          
          # Default fallback: Human Review
          return GateDecision(status="REQUIRE_HUMAN_REVIEW", reason="Ambiguous or moderate risk requires analyst approval")
  ```

### 5. `src/shield_rca/evaluation/unseen_eval.py`
- **Purpose:** Rigorous offline evaluation harness for zero-day / held-out attack generalization without fabricating results.

### 6. New Automated Tests (Expanding Coverage)
- `tests/test_risk_estimator.py`: Unit tests for trajectory drift math, multi-horizon risk bounds, and edge cases.
- `tests/test_stage_mapper.py`: Unit tests verifying correct attack stage transitions from state signatures.
- `tests/test_explainability.py`: Unit tests verifying group attribution sums and top-$K$ feature extraction.
- `tests/test_trigger_gate.py`: Comprehensive test suite testing all gating rules, blast-radius checks, and overrides.
- `tests/test_langgraph_workflow.py`: End-to-end integration tests of `ShieldLangGraph` with mock DB and simulated telemetry.
- `tests/test_blockchain.py`: Unit tests for `SimpleBlockchain` hashing, chain validation, and tamper detection.
- `tests/test_api_endpoints.py`: FastAPI endpoint tests for `/api/forecast/*`, `/incidents`, `/simulate/*`.

### 7. `scripts/run_trinetra_demo.py`
- **Purpose:** Standalone launcher that starts the FastAPI backend, serves the TRINETRA Offline Demo Interface, and initiates a controlled telemetry replay with zero external dependencies.

---

## 5. What Must NOT Be Touched (Strict Preservation)

To guarantee scientific reproducibility, backward compatibility, and system stability, the following must remain unmodified:

1. **Canonical Flow Feature Formulas (`src/shield_rca/telemetry/flow_features.py`)**:
   - The ~80 mathematical calculations (Fisher-Pearson skewness, sample variances, inter-arrival times, TCP flag bitmasks) must not be refactored or renamed.
2. **Canonical Feature Schema & Ordering (`src/shield_rca/telemetry/feature_schema.py`)**:
   - The $F=78$ feature list and alphabetical order must remain completely intact. Changing feature ordering would invalidate pre-trained weights.
3. **Network State Schema Dimensions (`src/shield_rca/network_state/state_schema.py`)**:
   - The $S=280$ dimension and grouping definitions (150, 48, 66, 6, 10) must remain constant.
4. **World Model Architecture (`src/shield_rca/world_model/model.py`)**:
   - The `StateResidualDynamicsModel` parameter shapes (input 280, hidden 128, 2 residual blocks, output 280) must not be changed.
5. **Pre-Trained Model Weights (`artifacts/world_model/best_model.pt`)**:
   - The existing checkpoint must not be deleted or overwritten.
6. **Existing Test Suite (`tests/test_*.py`)**:
   - All 62 existing tests must continue to pass without modification.
7. **Blockchain Cryptographic Integrity (`src/shield_rca/blockchain.py`)**:
   - The SHA-256 block hashing and chain linkage logic must remain deterministic.
8. **No Cloud/LLM Block Proliferation**:
   - Do not replace the deterministic logic with opaque cloud API calls. The entire system must execute autonomously and offline.
