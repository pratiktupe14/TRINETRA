# TRINETRA Attack Stage Mapping Specification

## 1. Purpose
The Attack Stage Progression Mapper translates continuous Network State trajectory deviations into defensive lifecycle attack stages aligned with the MITRE ATT&CK framework and the canonical 15-class attack taxonomy in `src/shield_rca/models/label_maps.py`.

It answers: *"What phase of an attack is unfolding according to network telemetry dynamics?"* without fabricating granular technique identifiers.

---

## 2. Inputs
- `current_state`: 280-dimensional Network State vector $S_t \in \mathbb{R}^{280}$.
- `predicted_state`: 280-dimensional predicted Network State vector $\hat{S}_{t+k} \in \mathbb{R}^{280}$.
- `horizon`: Integer future prediction step $k \ge 1$.
- `risk_score`: Optional normalized risk score $\in [0.0, 1.0]$.
- `known_label`: Optional ground-truth attack label supplied **strictly for offline evaluation/benchmarking**. Never leaked into runtime inference.

---

## 3. Processing Pipeline & Anti-Fabrication Guardrails
1. **Validation:** Checks vector dimensions ($S=280$) and confirms finite values.
2. **Quiescent Baseline Check:** If mean absolute drift across all 280 features is $\le \theta_{\text{normal}}$ (default 0.05), maps immediately to `AttackStage.NORMAL` with no attack alerts.
3. **Domain Metric Extraction:** Evaluates specific diagnostic features in the 280-d state vector:
   - Port dispersion delta: $|\Delta \text{macro\_port\_dispersion}|$
   - Unique destination ports delta: $|\Delta \text{endpoint\_unique\_dst\_ports}|$
   - Connection reset ratio delta: $|\Delta \text{macro\_rst\_ratio}|$
   - SYN/ACK asymmetry delta: $|\Delta \text{macro\_syn\_ack\_ratio}|$
   - Byte/packet transfer rate surges: $|\Delta \text{macro\_mean\_byte\_rate}|$ and $|\Delta \text{macro\_mean\_packet\_rate}|$
   - Download/upload volume asymmetry: $|\Delta \text{macro\_down\_up\_ratio}|$
4. **Candidate Stage Scoring:** Computes signature match scores across supported network-observable stages:
   - `RECONNAISSANCE` score: Weighted sum of port dispersion, unique destination ports, and RST ratio deltas.
   - `IMPACT` score: Weighted sum of byte rate surge, packet rate surge, and SYN/ACK ratio deltas.
   - `INFILTRATION` score: Weighted sum of down/up ratio drift, protocol shift, and RST ratio.
5. **Threshold Gating:** If the dominant score fails to exceed the minimum threshold, categorizes the anomaly as `AttackStage.UNKNOWN` with documented evidence of ambiguity.
6. **Strict MITRE Technique Boundary:** `mitre_technique` is explicitly set to `None` during standard prediction. It is never guessed or fabricated.

---

## 4. Supported vs. Unsupported Stages

### Supported Stages (Network-Observable)
| Attack Stage | Characteristic State Dynamics | Example Correlated Attacks |
|---|---|---|
| **`NORMAL`** | Drift $\le \theta_{\text{normal}}$; quiescent steady-state baseline. | Benign traffic |
| **`RECONNAISSANCE`** | Elevated port dispersion, unique endpoints, high reset ratios. | PortScan, Network Host Discovery |
| **`INITIAL_ACCESS`** | Rapid bursts in authentication endpoints, connection bursts. | SSH-Patator, FTP-Patator, Brute Force |
| **`INFILTRATION`** | Payload distribution shifts, asymmetric volume ratios, abnormal protocol ratios. | Infiltration, Botnet C2 channels |
| **`IMPACT`** | Volumetric surges in byte/packet rates, massive SYN flag floods. | DDoS, DoS Hulk, DoS GoldenEye, Slowloris |
| **`UNKNOWN`** | High overall state drift without a dominant signature. | Ambiguous / multi-vector anomalies |

### Unsupported Stages (Excluded from Network Inference)
The following MITRE ATT&CK stages require host-internal monitoring (OS process creation, registry modifications, kernel hooks) and cannot be defensibly inferred from network flow telemetry alone:
- *Execution*
- *Persistence*
- *Privilege Escalation*
- *Defense Evasion*
- *Credential Access (Host-Internal Memory/SAM)*
- *Collection (Local File System)*

Attempting to infer these stages from network state vectors would constitute architectural fabrication.

---

## 5. Outputs
- `StageMappingResult`:
  - `stage`: `AttackStage` enum (`NORMAL`, `RECONNAISSANCE`, `INITIAL_ACCESS`, `INFILTRATION`, `IMPACT`, `UNKNOWN`)
  - `confidence`: float in $[0.0, 1.0]$ or `None`
  - `supporting_features`: list of top drifting features with current, predicted, and delta values
  - `supporting_groups`: list of state groups triggering the classification
  - `evidence`: dictionary containing dominant score, candidate scores, and reasoning text
  - `horizon`: int ($k$)
  - `mitre_technique`: `None` (or documented ground-truth string if `known_label` was supplied)

---

## 6. Files
- Implementation: `src/shield_rca/world_model/stage_mapper.py`
- Tests: `tests/test_stage_mapper.py`

---

## 7. Configuration
Configurable via `AttackStageMapper` constructor arguments:
- `normal_drift_threshold`: Maximum drift considered baseline (default `0.05`).
- `recon_port_dispersion_threshold`: Sensitivity for reconnaissance detection (default `0.20`).
- `impact_rate_surge_threshold`: Sensitivity for volumetric impact detection (default `0.50`).
- `syn_flood_ratio_threshold`: Sensitivity for SYN flood detection (default `0.40`).

---

## 8. Tests
Covered by 9 unit tests in `tests/test_stage_mapper.py`:
- `test_normal_stage_on_low_drift`: Verifies low drift maps to `NORMAL` with `mitre_technique=None`.
- `test_reconnaissance_stage_mapping`: Verifies port dispersion surges map to `RECONNAISSANCE`.
- `test_impact_stage_mapping`: Verifies rate spikes and SYN floods map to `IMPACT`.
- `test_unknown_stage_on_unstructured_drift`: Verifies ambiguous drift does not guess false stages.
- `test_no_fabricated_mitre_technique`: Strictly verifies `mitre_technique is None` on normal inference.
- `test_offline_ground_truth_label_preservation`: Verifies offline evaluation ground truth is preserved.
- `test_deterministic_mapping`: Verifies identical inputs yield identical outputs.
- `test_invalid_input_rejections`: Rejects None, wrong shapes, and NaN/Inf.
- `test_unsupported_stages_documentation`: Confirms excluded host stages are not part of `AttackStage`.

---

## 9. Limitations
1. **Network Scope Only:** Cannot detect host-internal privilege escalation or local malware execution without host-agent integration.
2. **Coarse-Grained Stages:** Focuses on lifecycle phases rather than specific sub-technique permutations.
3. **Threshold Sensitivity:** In environments with naturally high port churn, thresholds must be calibrated to network baseline.

---

## 10. Implementation Status
`IMPLEMENTED` — Fully implemented, type-hinted, and validated with 9 passing unit tests.
