# TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System

[![CI Tests](https://img.shields.io/badge/pytest-285%20passed-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-World%20Model%20%2B%20CyberHeal-orange.svg)]()
[![Benchmark Datasets](https://img.shields.io/badge/datasets-CIC--IDS2018%20%7C%20CTU--13-informational.svg)]()
[![Safe Mode](https://img.shields.io/badge/remediation-simulation%20only-yellow.svg)]()

> **"Instead of only classifying whether current traffic is malicious, TRINETRA models network behaviour as an evolving temporal state and forecasts future network states to estimate attack progression."**

---

## 1. System Vision & Architecture

Traditional Intrusion Detection Systems (IDS) are fundamentally **reactive**: they flag an incident only after malicious signatures or anomalous traffic thresholds have already impacted the infrastructure. 

**TRINETRA** (Self-Healing Autonomous Predictive Shield) shifts the cyber defence paradigm from reactive detection to **predictive preemption**:

```text
 ┌──────────┐      ┌────────────┐      ┌─────────┐      ┌─────────┐
 │ Observe  │ ───► │ Understand │ ───► │ Predict │ ───► │ Explain │
 └──────────┘      └────────────┘      └─────────┘      └─────────┘
   Telemetry         280-D State         World Model      State Delta
   NetFlow/PCAP      Representation      Forecasting      Attribution
                                                               │
                                                               ▼
 ┌─────────┐       ┌────────────┐                       ┌─────────┐
 │  Learn  │ ◄──── │   Verify   │ ◄──────────────────── │   Act   │
 └─────────┘       └────────────┘                       └─────────┘
   Offline RLM       Post-Action                          Trigger Gate
   Audit Bundle      Verification                         Safe-Mode
```

### Core Components
1. **Telemetry & Flow Aggregator**: Ingests raw NetFlow/PCAP streams, parses flow metrics into canonical 78-D flow feature spaces, and aggregates them over sliding temporal windows ($W=10$s, $S=5$s).
2. **Network State Representation ($S_t \in \mathbb{R}^{280}$)**: Encodes temporal flow statistics into a 280-dimensional composite network state across 5 distinct topological groups:
   - **Continuous Traffic Dynamics** (features 0–59): Rate dynamics, packet/byte distributions, burstiness.
   - **TCP & Connection Signatures** (features 60–119): TCP flag ratios (SYN/ACK/RST), half-open connections, state transitions.
   - **Distribution & Higher-Order Statistics** (features 120–179): Skewness, kurtosis, inter-arrival time quantiles.
   - **Endpoint & Protocol Diversity** (features 180–239): Port dispersion, unique endpoints, protocol entropy.
   - **Macro Indicators** (features 240–279): Aggregate volumes, bandwidth asymmetry, global network pressure.
3. **Temporal World Model ($W_\theta$)**: An isolated dynamics neural network forecasting future continuous state vectors $\hat{S}_{t+k} = S_t + \Delta \hat{S}(S_t, k, \Delta \tau)$ across discrete horizons $k \in \{1, 3, 5\}$ and continuous wall-clock time $\Delta \tau$.
4. **Predictive Risk & Attack Stage Mapper**: Computes bounded asymptotic risk scores ($r \in [0, 1]$) and maps trajectory drift to progression stages (`RECONNAISSANCE`, `INFILTRATION`, `IMPACT`).
5. **State Delta Attribution**: Real-time explainability attributing forecast drift directly to shifting network feature groups (explicitly **State Delta Attribution**, not post-hoc SHAP approximations).
6. **Predictive Trigger Gate & CyberHeal**: Multi-dimensional policy guard evaluating risk, model confidence, evidence strength, action safety, and asset criticality. Executes strictly in **SAFE_MODE** (simulation) preventing unauthorized live host/firewall modifications.
7. **Post-Action Verification & Offline RLM**: Verifies whether predicted remediation normalizes network state trajectories and logs immutable learning records for offline policy calibration.

---

## 2. Real Datasets & Benchmark Methodology

TRINETRA has been rigorously benchmarked on **real, un-fabricated public intrusion datasets**:

* **CSE-CIC-IDS2018**: Evaluated on multi-day enterprise captures (Brute Force, DoS, DDoS, Botnet) using memory-safe chunked streaming.
* **CTU-13 (Stratosphere IPS Project)**: Complete evaluation across all 13 official NetFlow scenarios (19,976,700 real flows):
  - **Train**: Scenarios 1, 2, 3, 6, 7, 8, 9
  - **Validation**: Scenarios 11, 12
  - **Known Test**: Scenarios 4, 5, 13
  - **Unseen Out-of-Distribution Test**: Scenario 10 ONLY (`capture20110818.binetflow`, Rbot ICMP/UDP storm flood)

> **Important Dataset Disclosure**:  
> Benchmark datasets (`.binetflow`, `.pcap`, raw `.csv`) are **NOT** committed to this repository to comply with size restrictions and reproducible open-science guidelines. Data loaders stream directly from local storage paths.

---

## 3. Quick Start & Installation

### Requirements
- Python 3.10, 3.11, or 3.12 (64-bit)
- PyTorch 2.0+ (CPU or CUDA)
- scikit-learn, numpy, pandas, joblib

### Setup
```bash
# 1. Clone repository
git clone https://github.com/pratiktupe14/TRINETRA.git
cd TRINETRA

# 2. Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 4. Running the Test Suite

The test suite validates end-to-end telemetry extraction, 280-D state encoding, World Model dynamics, risk estimation, trigger gating, data adapters, and leakage isolation:

```bash
pytest -q
```
*Current Suite Status: **285 passed, 0 failed**.*

---

## 5. Reproducing Benchmarks

All evaluation phases are fully automated via standalone CLI scripts:

### A. Phase 11B — CTU-13 Real-Data Benchmark & Unseen Evaluation
```bash
python scripts/run_phase11b_ctu13_benchmark.py
```
*Generates full 13-scenario quality audit, strict split verification, and baseline comparison in `artifacts/phase11b/`.*

### B. Phase 12 — Real-Data Model Improvement & State Normalization
```bash
python scripts/run_phase12_benchmark.py
```
*Runs root-cause pipeline audits, 280-D feature dominance analysis, log1p feature stabilization, Candidate A-F forecasting benchmarks, and validation-tuned threshold calibration in `artifacts/phase12/`.*

---

## 6. Real-Data Benchmark Highlights

Empirical findings from the Phase 11B and Phase 12 benchmarks on official CTU-13 data:

| Metric / Pipeline | Phase 11B Baseline | Phase 12 Improved | Impact / Finding |
|:---|:---:|:---:|:---|
| **Known Test F1-Score** | `0.0114` | **`0.3960`** | **34.7x improvement** via log1p scaling & active-window extraction |
| **Known Test PR-AUC** | `0.0093` | **`0.3554`** | Strong precision-recall trade-off on imbalanced real traffic |
| **Known Test ROC-AUC** | `0.6395` | **`0.7526`** | High discrimination between benign and attack state windows |
| **Linear Baseline MAE ($k=1$)** | `43,713,804.0` | **`0.2892`** | Covariance ill-conditioning eliminated; **15.5% better than Persistence** |
| **Production Model Status** | Unchanged | **Unchanged** | Hash verified invariant (`78ed7a...`) |
| **Unseen Scenario 10 Generalization** | F1: `0.0000` | ROC-AUC: `0.540` | Genuine OOD volumetric flood signature exposed; zero data leakage |

---

## 7. SIH Claim Boundaries & Safety Disclosures

In accordance with strict technical integrity guidelines for the Smart India Hackathon (SIH):

1. **Flow vs Packet Level**: Real-data CTU-13 evaluations were conducted using official Bidirectional NetFlow (`.binetflow`) records. Packet-level deep inspection metrics are not claimed for CTU-13.
2. **Explainability**: Model attribution is computed deterministically via **State Delta Attribution** (measuring normalized feature group displacement). We do not claim post-hoc SHAP values.
3. **MITRE ATT&CK Attribution**: `mitre_technique` is explicitly set to `None` when offline ground-truth telemetry labels lack cryptographically verified technique mappings.
4. **Out-of-Distribution Generalization**: Performance on Scenario 10 demonstrates genuine distribution shift under extreme volumetric DDoS traffic. We do not claim 100% zero-day prevention.
5. **Autonomous Remediation**: CyberHeal operates with `SAFE_MODE=True` enabled by default (simulation mode). Host firewall alterations, routing table changes, or process terminations require explicit administrative governance.

---

## 8. Repository Layout

```text
CyberHeal/
├── api/                    # REST API endpoints & schemas
├── artifacts/              # Benchmark reports, split definitions & audit manifests
│   ├── phase11b/           # Phase 11B real CTU-13 benchmark outputs
│   ├── phase12/            # Phase 12 audits, normalization & candidate evaluations
│   ├── phase13/            # Phase 13 final integration & SIH readiness reports
│   └── world_model/        # Production World Model checkpoint (best_model.pt)
├── configs/                # Pipeline & safety policy configurations
├── docs/                   # Architectural specifications & technical whitepapers
├── scripts/                # Execution & reproduction CLIs
├── src/                    # Core TRINETRA Python package
│   └── shield_rca/
│       ├── evaluation/     # Real data adapters & benchmark runners
│       ├── models/         # Classification & graph models
│       ├── network_state/  # 280-D state encoder, schema & preprocessor
│       ├── telemetry/      # Flow aggregation & feature extraction
│       └── world_model/    # Residual & GRU dynamics, risk estimation, trigger gate
└── tests/                  # Pytest verification & regression suites
```

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
