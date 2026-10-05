# TRINETRA: Packet-Level Features & Signal Enrichment Specification
**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Internal Response Engine:** CyberHeal — Autonomous Self-Healing Response Engine  
**SIH Problem Statement:** SIH26153 — AI based Network Attack Forecasting from Network Traffic Data  
**Phase:** Phase 8 — Packet-Level Features & Traffic-Level Signal Enrichment  
**Date:** October 2026  

---

> [!IMPORTANT]
> **Phase 8 Scope Notice:**  
> Phase 8 adds packet-level feature extraction and validation. It does not claim that the existing production World Model has been retrained on these new packet features. The production 78-dimensional flow feature schema and 280-dimensional Network State representation remain strictly unchanged to preserve backward compatibility with existing trained model checkpoints.

---

## 1. Objective

The primary objective of Phase 8 is to implement packet-derived network traffic signals to achieve strict compliance with the Smart India Hackathon (SIH26153) mandate:
$$\text{"Use both flow-level AND packet-level features from open-source traffic data."}$$

The TRINETRA system already features a high-grade 78-dimensional flow feature extraction engine compliant with the CICFlowMeter standard. Phase 8 enriches this telemetry pipeline with fine-grained packet-level signals (TTL statistics, TCP retransmissions, inter-arrival time percentiles, and packet jitter) in a modular, backward-compatible, and leakage-safe manner.

---

## 2. Existing 78-Feature Coverage vs. Packet-Level Enrichment

The 78 canonical flow features in `src/shield_rca/telemetry/feature_schema.py` capture statistical aggregates over entire 5-tuple conversations:
- **Packet Length Distribution:** `pkt_len_mean`, `pkt_len_std`, `pkt_len_min`, `pkt_len_max`, `pkt_len_var`, `pkt_len_skew`, `fwd_pkt_len_*`, `bwd_pkt_len_*`
- **Volume & Byte Totals:** `tot_fwd_pkts`, `tot_bwd_pkts`, `tot_fwd_bytes`, `tot_bwd_bytes`, `subflow_*`
- **Flow Rates:** `flow_bytes_per_s`, `flow_packets_per_s`, `fwd_bytes_per_s`, `bwd_bytes_per_s`, `down_up_ratio`
- **TCP Control Flags:** `fin_flag_cnt`, `syn_flag_cnt`, `rst_flag_cnt`, `psh_flag_cnt`, `ack_flag_cnt`, `urg_flag_cnt`, `ece_flag_cnt`, `cwr_flag_cnt`
- **Coarse Flow Timing:** `flow_duration_us`, `flow_iat_*`, `fwd_iat_*`, `bwd_iat_*`, `active_*`, `idle_*`

**Clarification on Packet Count & Packet Size:**  
Packet count and size distributions are already comprehensively represented by existing flow-level features (`tot_fwd_pkts`, `tot_bwd_pkts`, `pkt_len_mean`, etc.). Phase 8 does not create duplicate representations of packet count or packet size; instead, Phase 8 extracts previously missing packet-specific indicators:
1. IP Time-to-Live (TTL) & IPv6 Hop-Limit distributions.
2. Directional TCP sequence-overlap retransmission heuristics.
3. High-resolution packet inter-arrival percentiles ($p_{50}, p_{95}$) and inter-arrival jitter.

---

## 3. Newly Added Packet Feature Schema ($F_{pkt} = 18$)

All packet-level features are declared in `src/shield_rca/telemetry/packet_feature_schema.py` under `PACKET_FEATURE_NAMES` (strictly ordered alphabetically):

| Index | Feature Name | Type | Physical Bounds | Missing Value Sentinel | Description |
|---|---|---|---|---|---|
| 0 | `bwd_tcp_retrans_count` | `float` | $[0, \infty)$ | `0.0` | Backward direction TCP retransmitted packet count |
| 1 | `bwd_ttl_mean` | `float` | $[0, 255]$ or $-1.0$ | `-1.0` | Mean IPv4 TTL / IPv6 hop limit for backward packets |
| 2 | `fwd_tcp_retrans_count` | `float` | $[0, \infty)$ | `0.0` | Forward direction TCP retransmitted packet count |
| 3 | `fwd_ttl_mean` | `float` | $[0, 255]$ or $-1.0$ | `-1.0` | Mean IPv4 TTL / IPv6 hop limit for forward packets |
| 4 | `packet_iat_jitter` | `float` | $[0, \infty)$ | `0.0` | Mean absolute difference between consecutive IATs ($\text{s}$) |
| 5 | `packet_iat_max` | `float` | $[0, \infty)$ | `0.0` | Maximum packet inter-arrival time ($\text{s}$) |
| 6 | `packet_iat_mean` | `float` | $[0, \infty)$ | `0.0` | Mean packet inter-arrival time ($\text{s}$) |
| 7 | `packet_iat_min` | `float` | $[0, \infty)$ | `0.0` | Minimum packet inter-arrival time ($\text{s}$) |
| 8 | `packet_iat_p50` | `float` | $[0, \infty)$ | `0.0` | Median ($50\text{th}$ percentile) packet inter-arrival time ($\text{s}$) |
| 9 | `packet_iat_p95` | `float` | $[0, \infty)$ | `0.0` | $95\text{th}$ percentile packet inter-arrival time ($\text{s}$) |
| 10 | `packet_iat_std` | `float` | $[0, \infty)$ | `0.0` | Standard deviation of packet inter-arrival times ($\text{s}$) |
| 11 | `tcp_retransmission_count` | `float` | $[0, \infty)$ | `0.0` | Total TCP retransmissions in flow ($\text{fwd} + \text{bwd}$) |
| 12 | `tcp_retransmission_rate` | `float` | $[0.0, 1.0]$ | `0.0` | Ratio of retransmissions to total observed TCP packets |
| 13 | `ttl_max` | `float` | $[0, 255]$ or $-1.0$ | `-1.0` | Maximum TTL observed across all packets |
| 14 | `ttl_mean` | `float` | $[0, 255]$ or $-1.0$ | `-1.0` | Arithmetic mean of TTL observed across all packets |
| 15 | `ttl_min` | `float` | $[0, 255]$ or $-1.0$ | `-1.0` | Minimum TTL observed across all packets |
| 16 | `ttl_range` | `float` | $[0, 255]$ or $-1.0$ | `-1.0` | Range of TTL ($\text{ttl\_max} - \text{ttl\_min}$) |
| 17 | `ttl_std` | `float` | $[0, \infty)$ or $-1.0$ | `-1.0` | Sample standard deviation of TTL across all packets |

---

## 4. Mathematical Definitions & Methodologies

### 4.1. IP Time-To-Live (TTL) & IPv6 Hop Limit
For a packet stream $P = \{p_1, \dots, p_N\}$:
- IPv4 TTL is extracted from the IPv4 datagram header `ip_layer.ttl` ($\in [0, 255]$).
- IPv6 Hop Limit is extracted from the IPv6 fixed header `ip_layer.hlim` ($\in [0, 255]$) and treated as the equivalent hop-limit signal.
- Packets lacking IP headers or with corrupted/missing TTL fields are ignored during statistical aggregation.
- Let $V_{ttl} = \{p.ttl \mid p \in P, p.ttl \neq \text{None}, 0 \le p.ttl \le 255\}$.
  - If $|V_{ttl}| = 0$: all TTL metrics evaluate to $\text{MISSING\_TTL\_SENTINEL} = -1.0$.
  - If $|V_{ttl}| > 0$:
    $$\text{ttl\_min} = \min(V_{ttl}), \quad \text{ttl\_max} = \max(V_{ttl}), \quad \text{ttl\_mean} = \frac{1}{|V_{ttl}|} \sum_{v \in V_{ttl}} v$$
    $$\text{ttl\_std} = \sqrt{\frac{1}{|V_{ttl}|} \sum_{v \in V_{ttl}} (v - \text{ttl\_mean})^2}, \quad \text{ttl\_range} = \text{ttl\_max} - \text{ttl\_min}$$
- Directional averages $\text{fwd\_ttl\_mean}$ and $\text{bwd\_ttl\_mean}$ are computed analogously over forward and backward packet sets.

> [!CAUTION]
> **TTL Interpretation Rule:**  
> TTL varies by operating system default (e.g., Linux=64, Windows=128, Cisco=255) and routing path length. Low or anomalous TTL does not alone indicate malicious activity. Phase 8 strictly extracts numeric TTL distributions; semantic threat classification is left to downstream models.

---

### 4.2. TCP Retransmission Detection: Sequence-Overlap Heuristic

#### Detection Rule
A TCP packet $p$ in direction $d \in \{\text{forward}, \text{backward}\}$ is identified as a retransmission under the **sequence-overlap heuristic** if:
1. $p.\text{protocol} == 6$ (TCP).
2. The packet carries sequence space:
   $$\text{effective\_len} = \begin{cases} \max(p.\text{payload\_length}, 1), & \text{if } p.\text{tcp\_flags} \& (\text{SYN} \mid \text{FIN}) \\ p.\text{payload\_length}, & \text{otherwise} \end{cases}$$
   Pure ACKs with $\text{effective\_len} = 0$ are **strictly excluded** to prevent false positives on legitimate cumulative acknowledgment progressions.
3. The half-open sequence interval $[S, S + \text{effective\_len})$ where $S = p.\text{tcp\_seq}$ intersects any sequence interval $[a, b)$ previously observed in direction $d$:
   $$\max(a, S) < \min(b, S + \text{effective\_len})$$

#### Retransmission Rate
$$\text{tcp\_retransmission\_rate} = \begin{cases} \frac{\text{tcp\_retransmission\_count}}{N_{tcp}}, & \text{if } N_{tcp} > 0 \\ 0.0, & \text{if } N_{tcp} = 0 \end{cases}$$
where $N_{tcp}$ is the total number of TCP packets observed in the flow.

#### Heuristic Limitations & Edge Cases
- **No Ground Truth Claim:** This metric is explicitly a *sequence-overlap heuristic*, not a hardware-verified ground-truth detector.
- **Sequence Wrapping:** Sequence space is modeled modulo $2^{32}$. Because standard flow timeouts are bounded (120 seconds active timeout), sequence wrap-around within a single flow is exceptionally rare.
- **Out-of-Order Packets:** The interval tracker maintains disjoint sets of transmitted sequence intervals. Out-of-order packets filling previously unobserved sequence gaps are correctly recognized as new data rather than retransmissions.
- **SYN/FIN Tracking:** SYN and FIN control packets consume 1 sequence byte and are properly tracked when retransmitted during connection setup or teardown.

---

### 4.3. Packet Inter-Arrival Time (IAT) & Jitter

#### IAT Extraction
Let timestamps sorted chronologically be $t_0 \le t_1 \le \dots \le t_{N-1}$ (in fractional POSIX seconds).
If $N < 2$:
$$\text{packet\_iat\_*} = 0.0, \quad \text{packet\_iat\_jitter} = 0.0$$
If $N \ge 2$:
$$\Delta t_i = t_i - t_{i-1}, \quad \forall i \in \{1, \dots, N-1\}$$
$$\text{packet\_iat\_mean} = \frac{1}{N-1} \sum_{i=1}^{N-1} \Delta t_i$$
$$\text{packet\_iat\_std} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (\Delta t_i - \mu)^2}$$
$$\text{packet\_iat\_min} = \min(\{\Delta t_i\}), \quad \text{packet\_iat\_max} = \max(\{\Delta t_i\})$$
$$\text{packet\_iat\_p50} = \text{percentile}_{50}(\{\Delta t_i\}), \quad \text{packet\_iat\_p95} = \text{percentile}_{95}(\{\Delta t_i\})$$

#### Packet Inter-Arrival Jitter Formula
Jitter is defined deterministically as the mean absolute change between consecutive inter-arrival times:
$$\text{packet\_iat\_jitter} = \begin{cases} \frac{1}{N-2} \sum_{i=2}^{N-1} |\Delta t_i - \Delta t_{i-1}|, & \text{if } N \ge 3 \\ 0.0, & \text{if } N < 3 \end{cases}$$

> [!NOTE]
> This metric is termed **packet inter-arrival jitter** to distinguish it from protocol-specific metrics (such as RFC 3550 RTP jitter or RFC 4689 IPDV).

---

## 5. Missing Data & Protocol Robustness

The feature extraction pipeline is hardened against malformed, fragmented, or incomplete captures:
1. **Non-IP Packets (ARP, Ethernet):** Skipped by `_decode_packet` without exceptions.
2. **Missing TTL:** Recorded as `-1.0` (missing sentinel).
3. **Missing / Non-Finite Timestamps:** Filtered prior to IAT computation; if fewer than 2 valid timestamps remain, IAT metrics evaluate to `0.0`.
4. **UDP & ICMP Flows:** Retransmission count and rate evaluate strictly to `0.0`.
5. **Zero-Packet Flows:** Handled gracefully via `PacketFeatureVector.empty()`.

---

## 6. Algorithmic Complexity & Performance

- **Sequence Interval Merging:** Interval tracking uses sorted disjoint intervals. Insertion and overlap checks require $O(\log M)$ time per packet using binary search, where $M \le N_{pkts}$ is the number of contiguous data bursts.
- **Timing & IAT:** Single-pass sorting and NumPy vectorized differences require $O(N \log N)$ time and $O(N)$ memory.
- **Empirical Benchmark:** A flow of 5,000 packets processes in under 50 milliseconds on standard x86_64 hardware (well within the required 1.0 second threshold).

---

## 7. Integration Architecture & Backward Compatibility

```
                      RAW PCAP / PCAPNG / STREAM
                                  │
                                  ▼
                             Packet Parser
                       (RawPacketRecord: TTL, Seq, Ack)
                                  │
                                  ▼
                            Flow Aggregator
                             (FlowRecord)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      Canonical Flow Extractor         Packet Feature Extractor
       (78 Canonical Features)             (18 Packet Signals)
                 │                                 │
                 │                                 ▼
                 │                      PacketFeatureVector (F_pkt=18)
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                        build_enriched_features()
                 ┌─────────────────────────────────┐
                 │  - 78 Canonical Features        │
                 │  - metadata                     │
                 │  - "packet_features": dict      │
                 │  - "pkt_<name>": flattened      │
                 └────────────────┬────────────────┘
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       ▼                                                     ▼
Existing Production Path                              Enriched Path
78-D -> 280-D Network State -> World Model            (Future Phase 9 Retraining)
```

- **Canonical 78-D Features:** Completely untouched in `feature_schema.py`.
- **280-D Network State:** Completely untouched in `state_schema.py`.
- **World Model:** Retains exact $(B, 280)$ input tensor compatibility; no weights modified.
- **Opt-in Enrichment:** Downstream components access packet features via `extract_packet_features()`, `extract_packet_feature_vector()`, or `build_enriched_features()`.

---

## 8. SIH Gap Resolution

| Component | Status | Verification |
|---|---|---|
| Flow-level features (78 canonical) | IMPLEMENTED | Verified by 181 regression tests |
| Packet-level features (TTL, Retransmissions, Jitter) | IMPLEMENTED | Verified by 32 dedicated unit tests |
| Scapy PCAP/PCAPNG parsing | IMPLEMENTED | Verified on synthetic PCAPs |
| Production World Model weights | PRESERVED | No weights modified |
| Real CIC-IDS2018 / CTU-13 Benchmark | NOT VERIFIED | Deferred to Phase 9 |
| Offline TRINETRA Demo | PENDING | Phase 10 |
| Dashboard UI | PENDING | Phase 11 |
