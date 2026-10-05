# CyberHeal AI: K-Step Future-State Dataset ($\mathcal{D}_k$)

## 1. Formal Mathematical Foundations

In the predictive CyberHeal architecture:

$$\text{Real Telemetry Window } X_{t-T+1:t} \in \mathbb{R}^{T \times 78} \longrightarrow \mathbf{Network\ State\ S_t \in \mathbb{R}^{280}} \longrightarrow \mathbf{Future\ Target\ S_{t+k} \in \mathbb{R}^{280}} \longrightarrow \text{World Model}$$

### Definitions
1. **Discrete Window Index ($t$)**:
   $t \in \{0, 1, 2, \dots, M-1\}$ denotes the discrete sequence index of an observation window. Window $t$ spans historical flow observations from $t - T + 1$ to $t$, with effective timestamp $\tau_t$ (the ending timestamp of window $t$).

2. **Current Network State ($S_t$)**:
   The deterministic 280-dimensional summary of historical telemetry up to time $\tau_t$:
   $$S_t = f\left(X_{t-T+1:t}\right) \in \mathbb{R}^{280}$$

3. **Future Prediction Target ($Y_t^{(k)}$)**:
   The true future Network State $k$ observation windows ahead in time:
   $$Y_t^{(k)} = S_{t+k} = f\left(X_{t+k-T+1:t+k}\right) \in \mathbb{R}^{280}$$

4. **Prediction Dataset ($\mathcal{D}_k$)**:
   The set of all valid, non-leaking state-target pairs for horizon $k$:
   $$\mathcal{D}_k = \left\{ \left(S_t, S_{t+k}, \tau_t, \tau_{t+k}, \Delta \tau_{t, k}\right) \;\middle|\; t+k < M, \; \text{scenario}(t) = \text{scenario}(t+k), \; \text{split}(t) = \text{split}(t+k) \right\}$$

---

## 2. Horizon Specification ($k \in \{1, 3, 5\}$)

The prediction horizon $k$ represents discrete temporal observation steps:

| Horizon | Definition | Physical Meaning |
| :--- | :--- | :--- |
| **$k = 1$** | Immediate next window ($S_t \to S_{t+1}$) | Short-term network state transition (drift detection, immediate onset). |
| **$k = 3$** | 3 windows ahead ($S_t \to S_{t+3}$) | Medium-term trajectory (escalation of DDoS/Slowloris bursts, scan expansion). |
| **$k = 5$** | 5 windows ahead ($S_t \to S_{t+5}$) | Multi-step rollout horizon (sustained exfiltration, full attack saturation). |

### Wall-Clock Time Interpretation
Because flows arrive at dynamic rates, $k=1$ does **not** denote a static "1 second" or "1 minute". Instead, the exact elapsed wall-clock time is explicitly recorded for every pair:

$$\Delta \tau = \tau_{t+k} - \tau_t \ge 0$$

Downstream World Models can optionally ingest $\Delta \tau$ to perform continuous-time state transitions:
$$\hat{S}_{t+k} = \mathcal{W}_\theta\left(S_t, \Delta \tau\right)$$

---

## 3. Network State Semantics & Raw Telemetry Guarantee

A critical architectural finding from our Step 5 audit:
- **Macro Features Require Raw Physical Telemetry**:
  Quantities such as `macro_total_bytes`, `macro_total_packets`, `macro_syn_ack_ratio`, and `macro_rst_ratio` must be computed from **raw, unscaled telemetry** ($W \in \mathbb{R}^{T \times 78}$).
  If computed from standardized z-scores, byte sums and flag counts become negative, corrupting physical meaning and inverting ratios.
- **State-Level Normalization**:
  Model-oriented normalization is performed **after** state encoding on $S_t \in \mathbb{R}^{280}$ via `NetworkStatePreprocessor`, fit strictly on the training partition.

---

## 4. Strict Temporal Boundary & Split Policy (Zero Leakage)

To guarantee that no future target state leaks across evaluation boundaries:

```
[--- TRAIN PARTITION ---]       [-- VALIDATION PARTITION --]       [---- TEST PARTITION ----]
   S_0 ... S_m-1                   S_m ... S_m+n-1                    S_m+n ... S_total-1
           ▲       ▲                       ▲
           │       │                       │
      (S_i, S_i+k) │                       │
      KEPT IN TRAIN│                       │
                   ▼                       ▼
           (S_m-1, S_m) ──────────────► DROPPED!
           Crosses boundary: S_m is in VAL!
```

### Boundary Enforcement Rules
1. **TRAIN Pairs**: Kept if and only if **both** $S_t \in \text{TRAIN}$ and $S_{t+k} \in \text{TRAIN}$.
2. **VALIDATION Pairs**: Kept if and only if **both** $S_t \in \text{VAL}$ and $S_{t+k} \in \text{VAL}$.
3. **TEST Pairs**: Kept if and only if **both** $S_t \in \text{TEST}$ and $S_{t+k} \in \text{TEST}$.
4. **Boundary Drops**: If $S_t$ is in TRAIN and $S_{t+k}$ falls in VAL, the candidate pair is **strictly dropped**.
   - Dropped pairs are counted and reported in quality metrics.
   - We **never** compromise evaluation integrity to inflate sample counts.

---

## 5. Multi-Scenario Isolation Policy

In multi-scenario, multi-host, or multi-day datasets:
- State sequences are grouped strictly by `source_id` / `scenario_id`.
- States from Scenario A are **never** paired with states from Scenario B:
  $$S_t^{(A)} \longrightarrow S_{t+k}^{(A)} \quad \text{only!}$$
- At the tail of Scenario A, the final $k$ states cannot form complete pairs and are cleanly dropped.

---

## 6. Overlapping Window Semantics

When windows are formed with window size $T$ (e.g. 20) and stride $s$ (e.g. 10):
- Adjacent windows share $T - s$ flows of historical context.
- For $k = 1$, the future window $X_{t+1}$ advances by $s$ flows into new telemetry.
- **Why Chronological Evaluation Remains Rigorous**:
  Because TRAIN, VAL, and TEST are partitioned with non-overlapping temporal cutoffs, no flow appearing in a TEST window ever appeared in any TRAIN window. Overlap exists only within localized sequence dynamics, accurately reflecting live SOC streaming telemetry.

---

## 7. Data Quality & Audit Report

The `generate_future_dataset_quality_report` utility validates:
- Usable pair counts per horizon $k$.
- Total dropped boundary and tail pairs.
- Minimum, maximum, mean, and median elapsed wall-clock times $\Delta \tau$.
- Absence of negative delta times ($\Delta \tau \ge 0$).
- Exact dimension consistency ($S = 280$).
- Complete absence of attack labels, risk scores, or MITRE categories in state targets.

---

## 8. Summary of Interface for Future Step 6 (World Model)

The resulting `MultiHorizonFutureDataset` provides model-ready tensors:
- $X \in \mathbb{R}^{N \times 280}$: Current Network States $S_t$.
- $Y \in \mathbb{R}^{N \times K \times 280}$: Multi-step future state trajectories $[S_{t+1}, S_{t+3}, S_{t+5}]$.
- $\Delta T \in \mathbb{R}^{N \times K}$: Elapsed wall-clock time intervals.

No World Model is trained in Step 5.
