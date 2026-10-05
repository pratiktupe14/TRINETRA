# TRINETRA Phase 9A — CSE-CIC-IDS2018 Dataset Provenance & Execution Guide

**Project:** TRINETRA — Predictive Autonomous Cyber Defence & Self-Healing System  
**Internal Engine:** CyberHeal AI  
**SIH Problem Statement:** SIH26153  
**Target Dataset:** Communications Security Establishment & Canadian Institute for Cybersecurity IDS 2018 (CSE-CIC-IDS2018)  
**Execution Phase:** Phase 9A — Real CSE-CIC-IDS2018 Benchmark Execution  

---

## 1. Dataset Availability Status
**STATUS: NOT AVAILABLE LOCALLY**

In strict adherence to the **TRINETRA anti-fabrication mandate**:
- A comprehensive recursive search of `d:\sih 2026\CyberHeal`, `d:\sih 2026\`, and local drives verified that no official CSE-CIC-IDS2018 files (CSV flow records or raw PCAP files) exist locally.
- Automated internet downloads were strictly prohibited.
- **Zero real benchmark metrics have been fabricated, guessed, or inferred.**
- All infrastructure, dataset adapters, and evaluation pipelines stand verified and ready for immediate one-command execution upon local dataset placement.

---

## 2. Official Dataset Provenance & Metadata Specifications

| Attribute | Official Specification |
|---|---|
| **Dataset Name** | CSE-CIC-IDS2018 (Communications Security Establishment & Canadian Institute for Cybersecurity) |
| **Origin / Providers** | Canadian Institute for Cybersecurity (CIC) & Communications Security Establishment (CSE) |
| **Year / Capture Period** | February 14, 2018 – March 2, 2018 (10 capture days) |
| **Infrastructure / Topology** | AWS cloud infrastructure: 50 victim servers, 420 PCs, 30 victim servers across 5 departments |
| **Total Record Volume** | ~16,233,002 bidirectional flow records |
| **Raw Volume** | ~400 GB raw PCAPs; ~10 GB processed CSV flow records |
| **Expected Local Location** | `d:\sih 2026\CyberHeal\data\cic_ids2018\` |
| **Input Formats Supported** | CSV flow records (`*.csv`) and raw packet captures (`*.pcap`, `*.pcapng`) |
| **Timestamp Availability** | `Timestamp` column with format `dd/MM/yyyy HH:mm:ss` (converted to Unix epoch seconds) |
| **Packet Data Availability** | Available when PCAP archives are provided; unavailable on flow-only CSVs |

---

## 3. Official File Breakdown & Attack Taxonomy

The complete official CSE-CIC-IDS2018 corpus comprises 10 daily capture sessions:

| Target File | Attack Scenario | Canonical Category | Expected Labels |
|---|---|---|---|
| `02-14-2018.csv` | FTP & SSH Brute Force | `FTP-Patator`, `SSH-Patator` | `Benign`, `FTP-BruteForce`, `SSH-Bruteforce` |
| `02-15-2018.csv` | DoS GoldenEye & Slowloris | `DoS GoldenEye`, `DoS slowloris` | `Benign`, `DoS attacks-GoldenEye`, `DoS attacks-Slowloris` |
| `02-16-2018.csv` | DoS SlowHTTPTest & Hulk | `DoS Slowhttptest`, `DoS Hulk` | `Benign`, `DoS attacks-SlowHTTPTest`, `DoS attacks-Hulk` |
| `02-20-2018.csv` | DDoS LOIC-HTTP | `DDoS` | `Benign`, `DDoS attacks-LOIC-HTTP` |
| `02-21-2018.csv` | DDoS LOIC-UDP & HOIC | `DDoS` | `Benign`, `DDOS attack-LOIC-UDP`, `DDOS attack-HOIC` |
| `02-22-2018.csv` | Web Attacks (Brute Force, XSS) | `Web Attack - Brute Force`, `Web Attack - XSS` | `Benign`, `Brute Force -Web`, `Brute Force -XSS` |
| `02-23-2018.csv` | Web Attacks (SQL Injection, XSS) | `Web Attack - Sql Injection`, `Web Attack - XSS` | `Benign`, `SQL Injection`, `Brute Force -Web`, `Brute Force -XSS` |
| `02-28-2018.csv` | Infiltration | `Infiltration` | `Benign`, `Infiltration` |
| `03-01-2018.csv` | Infiltration | `Infiltration` | `Benign`, `Infiltration` |
| `03-02-2018.csv` | Botnet Traffic (ARES) | `Bot` | `Benign`, `Bot` |

---

## 4. Pipeline Execution Protocol (When Dataset is Placed)

### Step 1: Place Dataset Files
Copy or symlink official CSV or PCAP files into:
```
d:\sih 2026\CyberHeal\data\cic_ids2018\
```

### Step 2: Run Unified Benchmark CLI
Execute the deterministic benchmark CLI:
```powershell
python scripts/run_real_data_benchmark.py `
    --dataset cic-ids2018 `
    --data-dir data/cic_ids2018 `
    --output-dir artifacts/real_data_benchmark/cic_ids2018 `
    --input-format auto
```

### Step 3: Automated Pipeline Execution
The runner will autonomously:
1. **Compute Cryptographic Hashes:** Calculate SHA-256 for all ingested files into `provenance.json`.
2. **Audit Data Quality:** Detect/clean infinite values, NaN values, duplicate rows, and header repeats into `data_quality_report.json`.
3. **Partition Chronologically:** Split strictly 70% Train, 15% Validation, 15% Test without temporal leakage into `split_summary.json`.
4. **Align Features:** Align 78 canonical flow features and invoke Phase 8 packet feature extraction if PCAP files exist (`feature_report.json`).
5. **Evaluate Logistic Baseline:** Fit on train, tune threshold on validation, test on test split (`logistic_baseline_metrics.json`).
6. **Evaluate World Model:** Compute k=1, 3, 5 multi-horizon forecasting MAE/MSE vs Persistence and Linear baselines (`world_model_forecasting_metrics.json`).
7. **Evaluate Unseen Attacks:** Isolate an attack category (e.g. `Bot`) and compute degradation $\Delta F1$ (`unseen_attack_metrics.json`).
8. **Simulate CyberHeal Intake:** Verify dry-run safe remediation handoff with zero real firewall mutations (`cyberheal_simulation_summary.json`).
9. **Protect Production Model:** Verify `artifacts/world_model/best_model.pt` hash is unchanged.
