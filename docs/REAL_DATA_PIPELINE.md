# CyberHeal AI: Real Data → Feature Pipeline

## 1. Overview and Purpose

The Real Data Pipeline in CyberHeal transforms raw, heterogeneous network telemetry into mathematically standardized, chronologically sorted, and leakage-safe temporal window tensors $[N, T, F]$.

This pipeline forms the scientific foundation that feeds into the **Network State** representation and subsequent **World Model** rollouts, without fabricating synthetic attack probabilities or relying on hardcoded anomaly scores.

```
+-------------------+
| Raw Network Data  |  (PCAP file, in-memory byte buffer, or CIC-IDS CSV)
+---------+---------+
          |
          v
+-------------------+
| Packet Ingestion  |  (Scapy streaming reader, protocol filtering: TCP/UDP/ICMP)
+---------+---------+
          |
          v
+-------------------+
| Flow Aggregator   |  (Stateful 5-tuple bidirectional aggregation, active/idle timeouts)
+---------+---------+
          |
          v
+-------------------+
| Feature Extractor |  (Pure statistical extraction, ~80 raw network metrics)
+---------+---------+
          |
          v
+-------------------+
| Canonical Schema  |  (F = 78 numerical features, schema version 1.0.0)
+---------+---------+
          |
          v
+-------------------+
|  Data Validator   |  (Physical bound checks, NaN/Inf rejection, deduplication)
+---------+---------+
          |
          v
+-------------------+
| Temporal Windows  |  (Sliding observation windows [N, T, F] + horizon hooks [t+1, ..., t+K])
+---------+---------+
```

---

## 2. Supported Input Types

The pipeline provides unified ingestion via `DatasetBuilder`:

1. **Raw PCAP / PCAPNG Files**: On-disk capture files recorded via tcpdump, Wireshark, or Zeek.
2. **In-Memory Packet Bytes**: Memory-buffered PCAP byte payloads uploaded via API endpoints (`iter_packets_from_bytes`), enabling zero-disk live ingestion.
3. **Flow CSV Datasets**: Tabular CSV datasets conforming to the CIC-IDS2017, CSE-CIC-IDS2018, or custom SOC flow logger formats. Standard column headers are automatically mapped using `CIC_COLUMN_ALIASES`.

---

## 3. Packet → Flow → Feature Process

### Packet Decoding (`pcap_parser.py`)
Streams raw packets into lightweight, slotted `RawPacketRecord` dataclasses containing:
- High-resolution POSIX timestamp
- 5-tuple: Source IP, Destination IP, Source Port, Destination Port, Protocol (TCP, UDP, ICMP)
- Packet lengths: Total IP length and TCP/UDP payload length
- TCP flags: Bitmask byte (SYN, ACK, FIN, RST, PSH, URG, ECE, CWR)

### Stateful Flow Aggregation (`flow_aggregator.py`)
Packets are grouped into bidirectional flows using a canonical `FlowKey`:
- Forward (`fwd`) direction: Flow initiator (SYN sender or first packet).
- Backward (`bwd`) direction: Reverse response direction.
- Flows close on:
  - TCP RST or FIN-ACK teardown sequence.
  - Idle timeout: 30 seconds of inactivity.
  - Active timeout: 120 seconds maximum lifetime.

### Statistical Feature Extraction (`flow_features.py`)
Pure mathematical extraction matching the CICFlowMeter standard:
- Packet length distributions: Min, max, mean, std, variance, Fisher-Pearson skewness (forward, backward, and combined).
- Inter-arrival times (IAT): Mean, std, max, min, total duration.
- Transfer rates: Flow bytes/sec, flow packets/sec, forward bytes/sec, backward bytes/sec.
- TCP flag counters: SYN, FIN, RST, PSH, ACK, URG, ECE, CWR counts.
- Active & Idle times: Statistical distribution of active burst durations and inter-burst idle times.

---

## 4. Canonical Feature Schema ($F = 78$)

To guarantee strict consistency across training, validation, testing, and production inference, the schema is locked to **$F = 78$** features in deterministic alphabetical order (`feature_schema.py`).

| Category | Dimension | Example Features |
| :--- | :--- | :--- |
| **Packet Lengths** | 18 | `pkt_len_mean`, `fwd_pkt_len_std`, `bwd_pkt_len_max`, `pkt_len_skew` |
| **Inter-Arrival Times** | 15 | `flow_iat_mean`, `fwd_iat_max`, `bwd_iat_std`, `flow_iat_total` |
| **TCP Flags** | 12 | `syn_flag_cnt`, `ack_flag_cnt`, `rst_flag_cnt`, `fwd_psh_flags` |
| **Active / Idle** | 8 | `active_mean`, `active_std`, `idle_max`, `idle_min` |
| **Rates & Ratios** | 5 | `flow_bytes_per_s`, `flow_packets_per_s`, `down_up_ratio` |
| **Bulk Metrics** | 6 | `fwd_avg_bulk_rate`, `bwd_avg_bytes_per_bulk`, `bwd_avg_pkts_per_bulk` |
| **Subflows & Counts** | 8 | `tot_fwd_pkts`, `tot_bwd_bytes`, `subflow_fwd_bytes`, `subflow_bwd_pkts` |
| **Network Endpoints** | 3 | `src_port`, `dst_port`, `protocol` |
| **Header Lengths** | 2 | `fwd_header_length`, `bwd_header_length` |
| **Duration** | 1 | `flow_duration_us` |
| **Total** | **78** | |

*Metadata fields (`flow_id`, `flow_start_time`, `src_ip`, `dst_ip`, `scenario_id`, `attack_class`, `label_int`) are strictly isolated from the numerical feature tensor.*

---

## 5. Temporal Ordering and Semantics

Temporal ordering is strictly chronological:
- Records are sorted ascending by `flow_start_time`.
- Random shuffling of temporal data is **strictly prohibited**.
- Each window slice $X[i]$ represents the **historical observation window**:
  $$\text{Observation: } [t - T + 1, \dots, t]$$
- No future flow information can leak into an observation window.

---

## 6. Window Definition (`WindowBatch` $[N, T, F]$)

The sliding window generator produces a `WindowBatch` dataclass:
- **$X$**: 3D float32 tensor of shape `[N, T, F]` ($T = \text{window size}$, default 20; $F = 78$).
- **$y$**: 1D integer array of shape `[N]` representing the window-level sequence label (computed via majority voting, last-step label, or any-attack flag).
- **`window_start_times`**: Array of start POSIX timestamps (length $N$).
- **`window_end_times`**: Array of end POSIX timestamps (length $N$).
- **`source_ids`**: Scenario or host identifiers for each window (length $N$).
- **`schema_version`**: `"1.0.0"`.

---

## 7. Leakage-Safe Splitting Strategy

The pipeline implements two chronological partitioning modes in `split.py`:

1. **Pure Chronological Partitioning**:
   - **TRAIN** (first 70%): Earlier time period $[\tau_0, \tau_{\text{train\_end}}]$.
   - **VALIDATION** (next 15%): Middle time period $[\tau_{\text{val\_start}}, \tau_{\text{val\_end}}]$.
   - **TEST** (final 15%): Latest time period $[\tau_{\text{test\_start}}, \tau_{\text{test\_end}}]$.
   - Strict boundary verification: $\max(\text{train}) \le \min(\text{val})$ and $\max(\text{val}) \le \min(\text{test})$.

2. **Scenario-Aware Partitioning**:
   - When multiple capture days, sessions, or scenarios exist, partitions are allocated by distinct scenario IDs.
   - Prevents the same attack session from appearing in both train and test splits.

---

## 8. Normalization and Scaling Policy

To prevent data snooping and target leakage:
1. **Scaler Isolation**: Scalers (`StandardScaler` or `RobustScaler`) are fit **strictly** on the training partition.
2. **Transform Only**: Validation, test, and streaming inference sets are transformed using the fitted training scaler without refitting.
3. **Artifact Serialization**: The fitted scaler is serialized alongside the feature schema metadata (`scaler.joblib` and `preprocessor_meta.json`).
4. **Label Protection**: Labels (`label_int`, `attack_class`) and timestamps are never modified or normalized.

---

## 9. Data Validation and Fail-Loud Rules

The `DataValidator` enforces strict data integrity rules:
- **Empty Dataset Check**: Rejects empty inputs immediately.
- **Timestamp Integrity**: Rejects null, NaN, or non-positive timestamps.
- **Physical Feasibility**: Rejects impossible negative values in non-negative metrics (durations, byte counts, packet counts, rates).
- **NaN / Infinity Handling**: Fails loudly in strict mode if non-finite numbers appear in numerical features.
- **Duplicate Removal**: Identifies and drops duplicate flow records based on flow ID or 5-tuple timestamp matching.
- **Dimension Check**: Ensures exact conformity to $F = 78$.

---

## 10. Label Handling and Taxonomy

Labels follow the canonical 15-class taxonomy defined in `shield_rca.models.label_maps`:
- `0: BENIGN`
- `1: Bot`, `2: DDoS`, `3: DoS GoldenEye`, `4: DoS Hulk`, `5: DoS Slowhttptest`, `6: DoS slowloris`, `7: FTP-Patator`, `8: Heartbleed`, `9: Infiltration`, `10: PortScan`, `11: SSH-Patator`, `12: Web Attack - Brute Force`, `13: Web Attack - Sql Injection`, `14: Web Attack - XSS`.
- **Unknown Label Policy**: Unrecognized labels are explicitly reported in validation logs and filtered according to the configured `unknown_strategy` (`"drop"` or `"error"`), rather than being silently converted to benign.

---

## 11. Synthetic Test Fixtures vs. Real Data

- **Synthetic Test Fixtures**: `tests/fixtures/generate_test_pcap.py` produces small, fully-deterministic synthetic PCAPs (TCP HTTP, SSH, UDP DNS, ICMP echo) strictly for unit testing and CI verification.
- **Real Production Data**: Production datasets (e.g. CIC-IDS2017/2018 CSVs or PCAPs) are read directly via `DatasetBuilder.from_csv()` or `DatasetBuilder.from_pcap()` without hardcoded paths.
- No synthetic performance scores or artificial detection numbers are claimed.

---

## 12. How Future K-Step Targets Attach to This Pipeline

The `WindowBatch` structure contains a reserved metadata attribute:
`target_horizon_indices: list[tuple[int, int]]`

When `future_horizon_k > 0`:
- For each historical window $[t - T + 1, \dots, t]$, the slice $(t + 1, \min(t + 1 + K, N))$ is stored in metadata.
- In Step 4/5 (World Model), the prediction head can directly index the true future state trajectory $S_{t+1:t+K}$ corresponding to each observation window without refactoring the telemetry pipeline.
