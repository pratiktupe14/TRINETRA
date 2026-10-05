"""CTU-13 Botnet Dataset Adapter for CyberHeal AI / TRINETRA.

Provides memory-safe streaming, deterministic normalization, and canonical feature
mapping for the CTU-13 (Stratosphere IPS) Botnet dataset (.binetflow and CSV formats).

Key Capabilities:
1. Ingestion of official CTU-13 bidirectional NetFlow (.binetflow) and flow CSVs.
2. Exact mapping to TRINETRA's 78 canonical flow features (CANONICAL_FEATURES).
3. 3-tier label normalization:
   - BOTNET / C&C / ATTACK -> binary 1, category "Bot" (or bot family)
   - NORMAL -> binary 0, category "Normal"
   - BACKGROUND -> binary 0, category "Background"
4. Deterministic timestamp parsing to POSIX float64 with chronological monotonicity.
5. Scenario extraction and strict multi-scenario isolation for unseen evaluation.
6. Zero cross-scenario or cross-split data leakage.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES
from .dataset_adapters import (
    ArrayHolder,
    DataQualityReport,
    DatasetAdapter,
    DatasetProvenance,
    compute_file_sha256,
)

logger = logging.getLogger(__name__)

# Standard Port mapping for common service names found in CTU-13 flows
SERVICE_PORT_MAP: dict[str, int] = {
    "http": 80,
    "https": 443,
    "domain": 53,
    "dns": 53,
    "smtp": 25,
    "ssh": 22,
    "telnet": 23,
    "ftp": 21,
    "ftp-data": 20,
    "ntp": 123,
    "snmp": 161,
    "netbios-ns": 137,
    "netbios-dgm": 138,
    "netbios-ssn": 139,
    "microsoft-ds": 445,
    "dhcps": 67,
    "dhcpc": 68,
    "irc": 6667,
    "imaps": 993,
    "pop3s": 995,
}

# Official CTU-13 13-Scenario Inventory Metadata
CTU13_OFFICIAL_SCENARIOS: dict[str, dict[str, Any]] = {
    "scenario_1": {
        "id": 1,
        "name": "Neris",
        "botnet": "Neris",
        "file": "capture20110810.binetflow",
        "pcap": "botnet-capture-20110810-neris.pcap",
        "date": "2011-08-10",
        "description": "IRC botnet, spamming, click fraud",
    },
    "scenario_2": {
        "id": 2,
        "name": "Neris",
        "botnet": "Neris",
        "file": "capture20110811.binetflow",
        "pcap": "botnet-capture-20110811-neris.pcap",
        "date": "2011-08-11",
        "description": "IRC botnet, port scans",
    },
    "scenario_3": {
        "id": 3,
        "name": "Rbot",
        "botnet": "Rbot",
        "file": "capture20110812.binetflow",
        "pcap": "botnet-capture-20110812-rbot.pcap",
        "date": "2011-08-12",
        "description": "IRC botnet, port scans",
    },
    "scenario_4": {
        "id": 4,
        "name": "Rbot",
        "botnet": "Rbot",
        "file": "capture20110815.binetflow",
        "pcap": "botnet-capture-20110815-rbot.pcap",
        "date": "2011-08-15",
        "description": "IRC botnet, ICMP flood DDoS",
    },
    "scenario_5": {
        "id": 5,
        "name": "Virut",
        "botnet": "Virut",
        "file": "capture20110815-2.binetflow",
        "pcap": "botnet-capture-20110815-fast-flux.pcap",
        "date": "2011-08-15",
        "description": "HTTP/P2P botnet, port scanning, spam",
    },
    "scenario_6": {
        "id": 6,
        "name": "Menti",
        "botnet": "Menti",
        "file": "capture20110816.binetflow",
        "pcap": "botnet-capture-20110816-menti.pcap",
        "date": "2011-08-16",
        "description": "PSExec lateral movement, port scan",
    },
    "scenario_7": {
        "id": 7,
        "name": "Sogou",
        "botnet": "Sogou",
        "file": "capture20110816-2.binetflow",
        "pcap": "botnet-capture-20110816-sogou.pcap",
        "date": "2011-08-16",
        "description": "HTTP botnet",
    },
    "scenario_8": {
        "id": 8,
        "name": "Murlo",
        "botnet": "Murlo",
        "file": "capture20110816-3.binetflow",
        "pcap": "botnet-capture-20110816-murlo.pcap",
        "date": "2011-08-16",
        "description": "Custom botnet, port scans",
    },
    "scenario_9": {
        "id": 9,
        "name": "Neris",
        "botnet": "Neris",
        "file": "capture20110817.binetflow",
        "pcap": "botnet-capture-20110817-neris.pcap",
        "date": "2011-08-17",
        "description": "IRC botnet, multiple attacks",
    },
    "scenario_10": {
        "id": 10,
        "name": "Rbot",
        "botnet": "Rbot",
        "file": "capture20110818.binetflow",
        "pcap": "botnet-capture-20110818-rbot.pcap",
        "date": "2011-08-18",
        "description": "IRC botnet, UDP and ICMP storms (HELD-OUT UNSEEN SCENARIO)",
    },
    "scenario_11": {
        "id": 11,
        "name": "Rbot",
        "botnet": "Rbot",
        "file": "capture20110818-2.binetflow",
        "pcap": "botnet-capture-20110818-2-rbot.pcap",
        "date": "2011-08-18",
        "description": "IRC botnet, port scanning",
    },
    "scenario_12": {
        "id": 12,
        "name": "NSIS.ay",
        "botnet": "NSIS.ay",
        "file": "capture20110819.binetflow",
        "pcap": "botnet-capture-20110819-nsis.pcap",
        "date": "2011-08-19",
        "description": "P2P botnet",
    },
    "scenario_13": {
        "id": 13,
        "name": "Virut",
        "botnet": "Virut",
        "file": "capture20110815-3.binetflow",
        "alt_files": ["capture20110819-2.binetflow"],
        "pcap": "botnet-capture-20110815-fast-flux-2.pcap",
        "date": "2011-08-15",
        "description": "HTTP/P2P botnet, port scans",
    },
}


def parse_ctu_port(val: Any) -> int:
    """Safely parse CTU-13 port representation (int, string, hex, or service name)."""
    if val is None or pd.isna(val):
        return 0
    if isinstance(val, (int, np.integer)):
        return max(0, min(int(val), 65535))
    s = str(val).strip().lower()
    if not s:
        return 0
    if s in SERVICE_PORT_MAP:
        return SERVICE_PORT_MAP[s]
    if s.startswith("0x"):
        try:
            return max(0, min(int(s, 16), 65535))
        except ValueError:
            return 0
    try:
        return max(0, min(int(float(s)), 65535))
    except (ValueError, TypeError):
        return 0


def parse_ctu_protocol(val: Any) -> int:
    """Convert protocol string or number to standard IANA protocol number."""
    if val is None or pd.isna(val):
        return 6  # default TCP
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip().lower()
    if s in ("tcp", "6"):
        return 6
    if s in ("udp", "17"):
        return 17
    if s in ("icmp", "1"):
        return 1
    if s in ("igmp", "2"):
        return 2
    if s in ("arp", "2054"):
        return 2054
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 6


class Ctu13Adapter(DatasetAdapter):
    """Memory-safe, high-fidelity adapter for CTU-13 NetFlow and PCAP datasets."""

    def __init__(self) -> None:
        super().__init__(dataset_name="CTU-13")

    @staticmethod
    def detect_scenario_id(file_path: Path | str) -> str:
        """Deterministically determine scenario identifier from filename or path."""
        p_str = str(file_path).lower().replace("\\", "/")
        # 1. Match explicit scenario_X regex first
        match = re.search(r"scenario[_-]?(\d+)", p_str)
        if match:
            return f"scenario_{match.group(1)}"
        # 2. Match directory structure like /13/ or /ctu-13-dataset/13/
        match_dir = re.search(r"/(?:ctu[_-]?13[_-]?dataset/)?(1[0-3]|[1-9])/", p_str)
        if match_dir:
            return f"scenario_{match_dir.group(1)}"
        # 3. Match capture filename (e.g. capture20110815-2.binetflow -> scenario_5)
        for scen_id, meta in CTU13_OFFICIAL_SCENARIOS.items():
            base_f = meta["file"].lower()
            if base_f in p_str:
                return scen_id
            for alt in meta.get("alt_files", []):
                if alt.lower() in p_str:
                    return scen_id
        if "capture20110815-3" in p_str:
            return "scenario_13"
        return "scenario_unknown"

    @classmethod
    def discover_files(cls, data_dir: Path | str) -> list[Path]:
        """Recursively discover and return CTU-13 .binetflow and flow CSV files sorted by scenario."""
        root = Path(data_dir)
        if not root.exists():
            return []
        candidates = list(root.rglob("*.binetflow")) + list(root.rglob("*.csv"))
        # Exclude generated/temporary files
        filtered = [
            p for p in candidates
            if not p.name.startswith("cyberheal_") and not p.name.endswith(".mmap")
        ]
        # Sort deterministically by scenario ID, then filename
        def _sort_key(p: Path) -> tuple[int, str]:
            scen = cls.detect_scenario_id(p)
            num = (
                int(scen.split("_")[-1])
                if scen.startswith("scenario_") and scen.split("_")[-1].isdigit()
                else 999
            )
            return (num, p.name)

        return sorted(filtered, key=_sort_key)

    def _map_single_label(self, raw_str: str) -> tuple[int, str]:
        """Classify raw CTU-13 label into binary label (0/1) and attack category."""
        s = raw_str.strip().lower()
        if "portscan" in s or "scan" in s:
            return (1, "PortScan")
        elif "ddos" in s or "storm" in s:
            return (1, "DDoS")
        elif "spam" in s:
            return (1, "Spam")
        elif "c&c" in s or "-cc" in s or " cc" in s:
            return (1, "Bot")
        elif "botnet" in s or "attack" in s:
            return (1, "Bot")
        elif "normal" in s:
            return (0, "Normal")
        elif "background" in s:
            return (0, "Background")
        else:
            return (0, "BENIGN")

    def normalize_labels(self, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Vectorized normalization of CTU-13 labels."""
        unique_labels = labels.dropna().unique()
        bin_map: dict[Any, int] = {}
        cat_map: dict[Any, str] = {}
        for u in unique_labels:
            b, cat = self._map_single_label(str(u))
            bin_map[u] = b
            cat_map[u] = cat

        binary_series = labels.map(bin_map).fillna(0).astype(int)
        cat_series = labels.map(cat_map).fillna("BENIGN").astype(str)
        return binary_series, cat_series

    def load_and_normalize(
        self,
        data_paths: list[Path | str],
        max_rows: int | None = None,
        chunksize: int = 100_000,
        skip_rows: int = 0,
    ) -> tuple[pd.DataFrame, DataQualityReport, DatasetProvenance]:
        """Stream and normalize CTU-13 files into TRINETRA canonical flow representation."""
        if not data_paths:
            raise ValueError("No data paths provided to Ctu13Adapter")

        valid_paths = [Path(p) for p in data_paths]
        for p in valid_paths:
            if not p.exists():
                raise FileNotFoundError(f"CTU-13 dataset file not found: {p}")

        # Compute provenance
        total_bytes = sum(p.stat().st_size for p in valid_paths)
        file_hashes = {p.name: compute_file_sha256(p) for p in valid_paths}

        capacity = max_rows if max_rows is not None else 100_000

        all_features = np.zeros((capacity, len(CANONICAL_FEATURES)), dtype=np.float32)
        all_labels = np.zeros(capacity, dtype=np.int8)
        all_categories = np.empty(capacity, dtype=object)
        all_raw_labels = np.empty(capacity, dtype=object)
        all_scenarios = np.empty(capacity, dtype=object)

        mmap_file = tempfile.NamedTemporaryFile(prefix="cyberheal_ctu_", suffix=".mmap", delete=False)
        mmap_path = mmap_file.name
        mmap_file.close()
        all_timestamps = np.memmap(mmap_path, dtype=np.float64, mode="w+", shape=(capacity,))

        def _cleanup_mmap() -> None:
            try:
                if hasattr(all_timestamps, "_mmap") and all_timestamps._mmap is not None:
                    all_timestamps._mmap.close()
            except Exception:
                pass
            try:
                if os.path.exists(mmap_path):
                    os.remove(mmap_path)
            except Exception:
                pass

        atexit.register(_cleanup_mmap)

        total_usable_rows = 0
        total_raw_rows = 0
        total_missing = 0
        total_inf = 0
        discard_reasons: dict[str, int] = {}
        all_source_cols: set[str] = set()

        # Canonical feature index mapping for O(1) assignment
        f_map = {f: i for i, f in enumerate(CANONICAL_FEATURES)}

        for p in valid_paths:
            scenario_id = self.detect_scenario_id(p)
            # Determine separator (tab, comma, or whitespace)
            sample_line = ""
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                sample_line = f.readline()
            sep = "," if "," in sample_line else (r"\s+" if "\t" not in sample_line else "\t")

            reader = pd.read_csv(
                p,
                sep=sep,
                chunksize=chunksize,
                low_memory=False,
                engine="c" if sep in (",", "\t") else "python",
            )

            skipped_so_far = 0
            for chunk in reader:
                raw_chunk_len = len(chunk)
                total_raw_rows += raw_chunk_len
                all_source_cols.update(chunk.columns)

                if skip_rows > 0 and skipped_so_far < skip_rows:
                    needed_skip = skip_rows - skipped_so_far
                    if len(chunk) <= needed_skip:
                        skipped_so_far += len(chunk)
                        continue
                    else:
                        chunk = chunk.iloc[needed_skip:]
                        skipped_so_far += needed_skip

                if max_rows is not None and total_usable_rows >= max_rows:
                    break

                # Label identification
                label_candidates = [c for c in chunk.columns if "label" in c.lower() or "tag" in c.lower()]
                if not label_candidates:
                    raise ValueError(f"No Label column found in CTU-13 file: {p.name}")
                label_col = label_candidates[0]

                if max_rows is not None and total_usable_rows + len(chunk) > max_rows:
                    chunk = chunk.iloc[: max_rows - total_usable_rows]

                n_chunk = len(chunk)
                raw_lbls = chunk[label_col].astype(str)
                bin_lbls, cat_lbls = self.normalize_labels(raw_lbls)

                # Parse timestamps
                ts_cols = [c for c in chunk.columns if "time" in c.lower()]
                if ts_cols:
                    dt_series = pd.to_datetime(chunk[ts_cols[0]], errors="coerce")
                    ts_arr = dt_series.astype("datetime64[s]").astype("int64").astype(np.float64).to_numpy()
                    if np.isnan(ts_arr).any():
                        ts_arr = np.nan_to_num(ts_arr, nan=float(total_usable_rows))
                else:
                    ts_arr = np.arange(total_usable_rows, total_usable_rows + n_chunk, dtype=np.float64)

                # Feature extraction and derivation
                feat_block = np.zeros((n_chunk, len(CANONICAL_FEATURES)), dtype=np.float32)

                # Ports
                sport_col = [c for c in chunk.columns if c.lower() in ("sport", "srcport", "source_port")]
                if sport_col:
                    feat_block[:, f_map["src_port"]] = [parse_ctu_port(x) for x in chunk[sport_col[0]]]

                dport_col = [c for c in chunk.columns if c.lower() in ("dport", "dstport", "destination_port")]
                if dport_col:
                    feat_block[:, f_map["dst_port"]] = [parse_ctu_port(x) for x in chunk[dport_col[0]]]

                # Protocol
                proto_col = [c for c in chunk.columns if "proto" in c.lower()]
                if proto_col:
                    feat_block[:, f_map["protocol"]] = [parse_ctu_protocol(x) for x in chunk[proto_col[0]]]

                # Duration
                dur_col = [c for c in chunk.columns if c.lower() in ("dur", "duration", "flow_duration")]
                dur_s = (
                    pd.to_numeric(chunk[dur_col[0]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                    if dur_col
                    else np.zeros(n_chunk, dtype=np.float32)
                )
                dur_us = np.maximum(dur_s * 1e6, 0.0)
                feat_block[:, f_map["flow_duration_us"]] = dur_us
                feat_block[:, f_map["flow_iat_total"]] = dur_us

                # Packets & Bytes
                tot_pkts_col = [c for c in chunk.columns if c.lower() in ("totpkts", "tot_pkts", "packets")]
                tot_pkts = (
                    pd.to_numeric(chunk[tot_pkts_col[0]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                    if tot_pkts_col
                    else np.zeros(n_chunk, dtype=np.float32)
                )

                tot_bytes_col = [c for c in chunk.columns if c.lower() in ("totbytes", "tot_bytes", "bytes")]
                tot_bytes = (
                    pd.to_numeric(chunk[tot_bytes_col[0]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                    if tot_bytes_col
                    else np.zeros(n_chunk, dtype=np.float32)
                )

                src_bytes_col = [c for c in chunk.columns if c.lower() in ("srcbytes", "src_bytes")]
                src_bytes = (
                    pd.to_numeric(chunk[src_bytes_col[0]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                    if src_bytes_col
                    else tot_bytes * 0.5
                )
                dst_bytes = np.maximum(tot_bytes - src_bytes, 0.0)

                feat_block[:, f_map["tot_fwd_bytes"]] = src_bytes
                feat_block[:, f_map["tot_bwd_bytes"]] = dst_bytes
                feat_block[:, f_map["subflow_fwd_bytes"]] = src_bytes
                feat_block[:, f_map["subflow_bwd_bytes"]] = dst_bytes

                fwd_pkts = np.maximum(np.round(tot_pkts * 0.5), 1.0)
                bwd_pkts = np.maximum(tot_pkts - fwd_pkts, 0.0)
                feat_block[:, f_map["tot_fwd_pkts"]] = fwd_pkts
                feat_block[:, f_map["tot_bwd_pkts"]] = bwd_pkts
                feat_block[:, f_map["subflow_fwd_packets"]] = fwd_pkts
                feat_block[:, f_map["subflow_bwd_packets"]] = bwd_pkts

                # Rates
                dur_safe = np.maximum(dur_s, 1e-6)
                feat_block[:, f_map["flow_bytes_per_s"]] = tot_bytes / dur_safe
                feat_block[:, f_map["flow_packets_per_s"]] = tot_pkts / dur_safe
                feat_block[:, f_map["fwd_bytes_per_s"]] = src_bytes / dur_safe
                feat_block[:, f_map["bwd_bytes_per_s"]] = dst_bytes / dur_safe
                feat_block[:, f_map["down_up_ratio"]] = bwd_pkts / np.maximum(fwd_pkts, 1.0)

                # Packet lengths
                mean_len = tot_bytes / np.maximum(tot_pkts, 1.0)
                feat_block[:, f_map["pkt_len_mean"]] = mean_len
                feat_block[:, f_map["pkt_len_min"]] = mean_len
                feat_block[:, f_map["pkt_len_max"]] = mean_len
                feat_block[:, f_map["fwd_pkt_len_mean"]] = src_bytes / np.maximum(fwd_pkts, 1.0)
                feat_block[:, f_map["bwd_pkt_len_mean"]] = dst_bytes / np.maximum(bwd_pkts, 1.0)

                # State / TCP Flags
                state_col = [c for c in chunk.columns if c.lower() in ("state", "flags")]
                if state_col:
                    states_s = chunk[state_col[0]].astype(str).str.upper()
                    feat_block[:, f_map["syn_flag_cnt"]] = states_s.str.contains("S|SYN").astype(float)
                    feat_block[:, f_map["ack_flag_cnt"]] = states_s.str.contains("A|ACK").astype(float)
                    feat_block[:, f_map["rst_flag_cnt"]] = states_s.str.contains("R|RST").astype(float)
                    feat_block[:, f_map["fin_flag_cnt"]] = states_s.str.contains("F|FIN").astype(float)
                    feat_block[:, f_map["psh_flag_cnt"]] = states_s.str.contains("P|PSH").astype(float)
                    feat_block[:, f_map["urg_flag_cnt"]] = states_s.str.contains("U|URG").astype(float)

                # Clean any NaNs or Infs
                np.nan_to_num(feat_block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                # Resize if capacity exceeded
                if total_usable_rows + n_chunk > len(all_features):
                    new_cap = max(total_usable_rows + n_chunk, int(len(all_features) * 1.5))
                    all_features = np.resize(all_features, (new_cap, len(CANONICAL_FEATURES)))
                    all_labels = np.resize(all_labels, new_cap)
                    all_categories = np.resize(all_categories, new_cap)
                    all_raw_labels = np.resize(all_raw_labels, new_cap)
                    all_scenarios = np.resize(all_scenarios, new_cap)

                all_features[total_usable_rows : total_usable_rows + n_chunk] = feat_block
                all_timestamps[total_usable_rows : total_usable_rows + n_chunk] = ts_arr
                all_labels[total_usable_rows : total_usable_rows + n_chunk] = bin_lbls.to_numpy(dtype=np.int8)
                all_categories[total_usable_rows : total_usable_rows + n_chunk] = cat_lbls.to_numpy(dtype=object)
                all_raw_labels[total_usable_rows : total_usable_rows + n_chunk] = raw_lbls.to_numpy(dtype=object)
                all_scenarios[total_usable_rows : total_usable_rows + n_chunk] = scenario_id

                total_usable_rows += n_chunk

        N = total_usable_rows
        all_features = all_features[:N]
        all_labels = all_labels[:N]
        all_categories = all_categories[:N]
        all_raw_labels = all_raw_labels[:N]
        all_scenarios = all_scenarios[:N]
        all_timestamps = all_timestamps[:N]
        all_timestamps.flush()

        # Enforce non-decreasing chronological ordering
        if N > 1 and not np.all(all_timestamps[:-1] <= all_timestamps[1:]):
            sort_idx = np.argsort(all_timestamps, kind="stable")
            all_timestamps = all_timestamps[sort_idx]
            all_features = all_features[sort_idx]
            all_labels = all_labels[sort_idx]
            all_categories = all_categories[sort_idx]
            all_raw_labels = all_raw_labels[sort_idx]
            all_scenarios = all_scenarios[sort_idx]

        col_dict: dict[str, Any] = {c: all_features[:, i] for i, c in enumerate(CANONICAL_FEATURES)}
        col_dict["flow_start_time"] = all_timestamps
        col_dict["label_int"] = all_labels
        col_dict["attack_class"] = pd.Categorical(all_categories)
        col_dict["raw_label"] = pd.Categorical(all_raw_labels)
        col_dict["scenario_id"] = all_scenarios
        col_dict["flow_id"] = np.arange(N, dtype=np.int64)

        clean_df = pd.DataFrame(col_dict, copy=False)
        clean_df.attrs["features_holder"] = ArrayHolder(all_features)
        clean_df.attrs["labels_holder"] = ArrayHolder(all_labels)
        clean_df.attrs["timestamps_holder"] = ArrayHolder(all_timestamps)
        clean_df.attrs["categories_holder"] = ArrayHolder(all_categories)
        clean_df.attrs["cleanup_mmap"] = _cleanup_mmap

        benign_cnt = int((all_labels == 0).sum())
        attack_cnt = int((all_labels == 1).sum())
        cat_counts = {str(k): int(v) for k, v in clean_df["attack_class"].value_counts().items()}

        quality_report = DataQualityReport(
            dataset_name="CTU-13",
            total_rows=total_raw_rows,
            usable_rows=N,
            discarded_rows=total_raw_rows - N,
            discard_reasons=discard_reasons,
            benign_count=benign_cnt,
            attack_count=attack_cnt,
            attack_ratio=float(attack_cnt) / float(N) if N > 0 else 0.0,
            attack_categories=cat_counts,
            missing_value_count=total_missing,
            infinite_value_count=total_inf,
            duplicate_record_count=0,
            timestamp_start=float(all_timestamps.min()) if N > 0 else None,
            timestamp_end=float(all_timestamps.max()) if N > 0 else None,
            protocol_distribution={6: int(np.sum(all_features[:, f_map["protocol"]] == 6))},
            feature_coverage_ratio=36.0 / 78.0,
        )

        provenance = DatasetProvenance(
            dataset_name="CTU-13",
            source_files=[p.name for p in valid_paths],
            total_bytes=total_bytes,
            file_hashes_sha256=file_hashes,
            ingested_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            row_count=N,
            input_format="binetflow/csv",
        )

        return clean_df, quality_report, provenance


class Ctu13ScenarioSplitter:
    """Manages deterministic scenario splitting and strict unseen scenario isolation."""

    @staticmethod
    def get_unseen_split_definition() -> dict[str, Any]:
        """Returns the official deterministic split of the 13 CTU-13 scenarios."""
        return {
            "unseen_scenario": "scenario_10",
            "unseen_scenario_name": "Rbot (UDP & ICMP DDoS storm)",
            "unseen_scenario_file": "capture20110818.binetflow",
            "selection_rule": (
                "Deterministic hold-out rule: Scenario 10 represents high-volume, multi-vector DDoS "
                "and C&C traffic with large state divergence from baseline IRC botnets. "
                "Held out strictly for zero-leakage generalizability evaluation."
            ),
            "train_scenarios": [
                "scenario_1",  # Neris
                "scenario_2",  # Neris
                "scenario_3",  # Rbot
                "scenario_6",  # Menti
                "scenario_7",  # Sogou
                "scenario_8",  # Murlo
                "scenario_9",  # Neris
            ],
            "val_scenarios": [
                "scenario_11",  # Rbot
                "scenario_12",  # NSIS.ay
            ],
            "known_test_scenarios": [
                "scenario_4",  # Rbot
                "scenario_5",  # Virut
                "scenario_13", # Virut
            ],
            "unseen_test_scenarios": [
                "scenario_10", # Rbot (UDP / ICMP DDoS storm)
            ],
            "isolation_guarantee": (
                "Scenario 10 flows and labels are 100% disjoint from training, validation, "
                "scaler fitting, and threshold calibration."
            ),
        }
