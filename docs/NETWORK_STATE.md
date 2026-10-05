# CyberHeal AI: Network State Representation ($S_t$)

## 1. Concept and Purpose

In the predictive CyberHeal architecture:

$$\text{Real Network Data} \longrightarrow \text{Feature Extraction } (F=78) \longrightarrow \text{Temporal Windowing } (T) \longrightarrow \mathbf{Network\ State\ (S_t)} \longrightarrow \text{World Model}$$

The **Network State** $S_t$ is the mathematical bridge between raw temporal telemetry streams and the future-state **World Model**.

Instead of feeding raw, uncompressed, high-dimensional matrices $[T \times 78]$ directly into transition models, $S_t$ extracts a **principled, compact, and deterministic** snapshot of the current network condition at observation time $t$:

$$S_t = f\left(X_{t-T+1:t}\right) \in \mathbb{R}^{280}$$

where $f$ is a deterministic mathematical operator strictly utilizing only historical telemetry observations up to time $t$.

---

## 2. Mathematical Definition & Zero Leakage

Given a discrete sequence of flow records $\{x_k\}_{k=1}^{t}$ where each flow vector $x_k \in \mathbb{R}^{78}$, an observation window of length $T$ ending at time $t$ is:

$$X_{t-T+1:t} = \begin{bmatrix} x_{t-T+1, 1} & \dots & x_{t-T+1, 78} \\ \vdots & \ddots & \vdots \\ x_{t, 1} & \dots & x_{t, 78} \end{bmatrix} \in \mathbb{R}^{T \times 78}$$

### Fundamental Non-Leakage Theorems
1. **Strictly Retrospective**: $S_t$ depends **only** on $X_{t-T+1:t}$.
   $$S_t \perp X_{t+k} \quad \forall k \ge 1$$
   Modifying or appending any future packet, flow, or window after time $t$ has **zero mathematical effect** on $S_t$.
2. **Label Isolation**: Ground-truth labels $y_t$ (e.g. `attack_class`, `label_int`) are strictly excluded from $S_t$.
3. **No Future Targets**: Future prediction horizons $[t+1, \dots, t+K]$ are not part of state generation.

---

## 3. Dimensionality and Structured Groupings ($S = 280$)

Naive flattening of $X$ would yield $20 \times 78 = 1,560$ coefficients, causing the curse of dimensionality, temporal phase sensitivity, and overfitting.

Instead, $S_t$ condenses the window into **280 domain-grounded state features** partitioned into five distinct mathematical groups:

```
+-------------------------------------------------------------------------------+
|                        Network State S_t in R^280                              |
+-------------------------------------------------------------------------------+
| 1. Continuous Traffic Dynamics      | 30 features x 5 statistics = 150 feats |
| 2. TCP Flag & Connection Signatures | 12 features x 4 statistics = 48 feats  |
| 3. Distribution Shape & Auxiliary   | 33 features x 2 statistics = 66 feats  |
| 4. Endpoint & Protocol Diversity    | 6 structural metrics       = 6 feats   |
| 5. Macro-Network Indicators         | 10 holistic metrics        = 10 feats  |
+-------------------------------------------------------------------------------+
| Total Dimension S                   |                              280 feats  |
+-------------------------------------------------------------------------------+
```

---

## 4. Detailed Feature Group Formulations

### Group 1: Continuous Traffic Dynamics (150 Features)
Applies to the 30 continuous rate, byte, IAT, length, and duration metrics (e.g., `flow_bytes_per_s`, `flow_duration_us`, `pkt_len_mean`, `tot_fwd_bytes`, `flow_iat_mean`).

For each feature $j \in \{1, \dots, 30\}$ across window timesteps $\tau \in \{0, \dots, T-1\}$:
1. **Window Central Tendency (Magnitude)**:
   $$\mu_j = \frac{1}{T} \sum_{\tau=0}^{T-1} X_{\tau, j}$$
2. **Window Volatility / Burstiness**:
   $$\sigma_j = \sqrt{\frac{1}{T} \sum_{\tau=0}^{T-1} (X_{\tau, j} - \mu_j)^2}$$
3. **Window Peak Surge**:
   $$M_j = \max_{\tau \in [0, T-1]} X_{\tau, j}$$
4. **Current Instantaneous State**:
   $$L_j = X_{T-1, j}$$
5. **Dimensionless Trend (Z-score Drift Velocity)**:
   $$\text{slope}_j = \frac{\sum_{\tau=0}^{T-1} (\tau - \bar{\tau}) (X_{\tau, j} - \mu_j)}{\sum_{\tau=0}^{T-1} (\tau - \bar{\tau})^2 + \epsilon}, \quad \text{trend}_j = \frac{\text{slope}_j}{\sigma_j + \epsilon}$$

### Group 2: TCP Flags & Connection Signatures (48 Features)
Applies to the 12 discrete TCP flags and counters (e.g., `syn_flag_cnt`, `rst_flag_cnt`, `fin_flag_cnt`, `psh_flag_cnt`, `fwd_urg_flags`).
1. **Flag Mean Frequency**: $\mu_j$ (average flag presence across flows)
2. **Flag Max Burst**: $M_j$ (highest single-flow flag count)
3. **Instantaneous Flag**: $L_j = X_{T-1, j}$ (flag state in current flow)
4. **Flag Momentum (Delta)**: $\Delta_j = X_{T-1, j} - X_{0, j}$ (sudden surge in RST/SYN)

### Group 3: Distribution Shape & Auxiliary Metrics (66 Features)
Applies to 33 higher-order distribution metrics (skewness, variance, minimum packet lengths, header lengths, bulk transfer stats).
1. **Window Mean**: $\mu_j$
2. **Current Instantaneous Value**: $L_j = X_{T-1, j}$

### Group 4: Network Endpoint & Protocol Diversity (6 Features)
Rather than computing meaningless arithmetic averages on port numbers (e.g. $(80 + 443)/2 = 261.5$), this group extracts structural telemetry:
1. `endpoint_last_dst_port`: Most recent targeted destination port.
2. `endpoint_last_protocol`: Most recent transport protocol (TCP=6, UDP=17, ICMP=1).
3. `endpoint_unique_dst_ports`: Count of distinct destination ports targeted within the window (surges during horizontal port scans).
4. `endpoint_unique_src_ports`: Count of distinct source ports observed (detects spoofing or ephemeral exhaustion).
5. `endpoint_tcp_ratio`: Fraction of window flows using TCP.
6. `endpoint_udp_ratio`: Fraction of window flows using UDP.

### Group 5: Macro-Network Indicators (10 Features)
Holistic cross-flow aggregations capturing network-wide health:
1. `macro_total_bytes`: Aggregate bytes transferred ($\sum \text{fwd} + \text{bwd}$).
2. `macro_total_packets`: Aggregate packets transferred.
3. `macro_duration_sum_s`: Cumulative active flow duration.
4. `macro_mean_byte_rate`: Window-wide transfer rate (bytes/s).
5. `macro_mean_packet_rate`: Window-wide packet rate (packets/s).
6. `macro_down_up_ratio`: Asymmetry ratio of download to upload bytes.
7. `macro_syn_ack_ratio`: Ratio of SYN flags to ACK flags (elevated in SYN Flood / DDoS attacks).
8. `macro_rst_ratio`: Ratio of connection resets to total packets (elevated in failed scans and aborted connections).
9. `macro_port_dispersion`: Unique destination ports divided by $T$.
10. `macro_flow_density`: Flow concurrency rate ($T / \text{duration}$).

---

## 5. Container & Batch Interface

The `NetworkStateEncoder` exposes:

1. **Single Window Encoding**:
   ```python
   encoder = NetworkStateEncoder()
   state: NetworkState = encoder.encode_window(window_matrix_Tx78, timestamp=t, source_id="node-A")
   # state.state_vector is 1D np.ndarray of shape (280,)
   ```
2. **Batch Window Encoding**:
   ```python
   batch_state: NetworkStateBatch = encoder.encode_batch(window_batch)
   # batch_state.states is 2D np.ndarray of shape (N, 280)
   ```

---

## 6. How the Future World Model Will Consume $S_t$

In Step 5 and beyond, the World Model models future network dynamics:

$$\hat{S}_{t+k} = \mathcal{W}_\theta\left(S_t, A_t\right)$$

where:
- $S_t \in \mathbb{R}^{280}$ is the current encoded Network State.
- $A_t$ represents a candidate remediation action (e.g. rate limit, IP block, container quarantine, or null action).
- $\hat{S}_{t+k} \in \mathbb{R}^{280}$ is the forecasted future state $k$ steps ahead.

Because $S_t$ preserves exact physical scales, rates, and flag proportions without lossy or stochastic components, the World Model can reliably predict whether anomalous traffic surges will escalate, decay, or be mitigated by proposed actions.
