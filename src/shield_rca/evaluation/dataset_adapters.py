"""Dataset Ingestion Adapters and Provenance Tracking for TRINETRA / CyberHeal AI.

Provides adapters for open-source cybersecurity datasets:
1. CIC-IDS2018 (CSE-CIC-IDS2018) CSV flow records and raw PCAP/PCAPNG captures.
2. CTU-13 Botnet dataset NetFlow / flow records and PCAP captures.

Responsibilities:
- Header normalization: maps heterogeneous source column names to canonical 78 flow features.
- Label normalization: maps diverse attack strings into binary labels (0=BENIGN, 1=ATTACK)
  and canonical attack categories while preserving raw labels.
- Provenance tracking: calculates file sizes, record counts, and SHA-256 cryptographic hashes.
- Data Quality Audit: identifies missing/NaN values, infinite values, duplicate records,
  protocol distributions, and class imbalance.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import logging
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..telemetry.feature_schema import (
    CANONICAL_FEATURE_DIM,
    CANONICAL_FEATURES,
    CIC_COLUMN_ALIASES,
    align_dataframe_to_schema,
)

logger = logging.getLogger(__name__)

# Standard attack label mappings for CIC-IDS2018
CIC_2018_LABEL_MAP: dict[str, tuple[int, str]] = {
    "benign": (0, "BENIGN"),
    "ddos attacks-loic-http": (1, "DDoS"),
    "ddos attack-hoic": (1, "DDoS"),
    "ddos attack-loic-udp": (1, "DDoS"),
    "dos attacks-hulk": (1, "DoS Hulk"),
    "dos attacks-slowloris": (1, "DoS slowloris"),
    "dos attacks-slowhttptest": (1, "DoS Slowhttptest"),
    "dos attacks-goldeneye": (1, "DoS GoldenEye"),
    "ftp-bruteforce": (1, "FTP-Patator"),
    "ssh-bruteforce": (1, "SSH-Patator"),
    "brute force -web": (1, "Web Attack - Brute Force"),
    "brute force -xss": (1, "Web Attack - XSS"),
    "sql injection": (1, "Web Attack - Sql Injection"),
    "bot": (1, "Bot"),
    "infiltration": (1, "Infiltration"),
    "infilteration": (1, "Infiltration"),
}

# Standard attack label mappings for CTU-13
CTU_13_LABEL_MAP: dict[str, tuple[int, str]] = {
    "normal": (0, "BENIGN"),
    "background": (0, "BENIGN"),
    "botnet": (1, "Bot"),
    "c&c": (1, "Bot"),
    "c&c-heartbeat": (1, "Bot"),
    "c&c-torii": (1, "Bot"),
    "c&c-filedownload": (1, "Bot"),
    "c&c-mirai": (1, "Bot"),
    "attack": (1, "DDoS"),
    "ddos": (1, "DDoS"),
    "portscan": (1, "PortScan"),
}


@dataclass(slots=True, frozen=True)
class DatasetProvenance:
    """Cryptographic and operational provenance metadata for a dataset."""

    dataset_name: str
    source_files: list[str]
    total_bytes: int
    file_hashes_sha256: dict[str, str]
    ingested_at: str
    row_count: int
    input_format: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DataQualityReport:
    """Comprehensive data quality audit of ingested network traffic data."""

    dataset_name: str
    total_rows: int
    usable_rows: int
    discarded_rows: int
    discard_reasons: dict[str, int]
    benign_count: int
    attack_count: int
    attack_ratio: float
    attack_categories: dict[str, int]
    missing_value_count: int
    infinite_value_count: int
    duplicate_record_count: int
    timestamp_start: float | None
    timestamp_end: float | None
    protocol_distribution: dict[int, int]
    feature_coverage_ratio: float  # fraction of 78 canonical features populated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAdapter(ABC):
    """Abstract base adapter for normalizing external network traffic datasets."""

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    @abstractmethod
    def load_and_normalize(
        self,
        data_paths: list[Path | str],
        max_rows: int | None = None,
        chunksize: int = 100_000,
    ) -> tuple[pd.DataFrame, DataQualityReport, DatasetProvenance]:
        """Load data files, normalize schema and labels, and audit quality."""


def compute_file_sha256(path: Path | str, chunk_size: int = 1048576) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _get_file_start_timestamp(path: Path | str) -> float:
    """Read initial rows to determine earliest timestamp for chronological file sequencing."""
    try:
        df_sample = pd.read_csv(path, nrows=5, low_memory=False)
        ts_cols = [c for c in df_sample.columns if "time" in c.lower()]
        if ts_cols:
            s = df_sample[ts_cols[0]].dropna()
            if len(s) > 0:
                sample_val = str(s.iloc[0]).strip()
                if "/" in sample_val:
                    dt = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
                    if dt.isna().all():
                        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
                else:
                    dt = pd.to_datetime(s, errors="coerce")
                valid = dt.dropna()
                if len(valid) > 0:
                    return float(valid.iloc[0].timestamp())
    except Exception:
        pass
    return 0.0


def _count_lines(paths: list[Path | str]) -> int:
    """Fast binary line count across multiple files to calculate pre-allocation capacity."""
    total = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                total += sum(buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b""))
        except Exception:
            pass
    return max(1, total)


class ArrayHolder:
    """Wrapper around NumPy arrays to allow safe inclusion in pandas DataFrame.attrs."""
    __slots__ = ("array",)

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __deepcopy__(self, memo: dict[int, Any]) -> ArrayHolder:
        # Crucial for Pandas __finalize__: Pandas deepcopies .attrs on every column indexing!
        # Returning self prevents copy.deepcopy from duplicating the multi-gigabyte array in memory.
        return self

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ArrayHolder) and (self.array is other.array)


class CicIds2018Adapter(DatasetAdapter):
    """Adapter for CSE-CIC-IDS2018 flow CSV and PCAP datasets."""

    def __init__(self) -> None:
        super().__init__(dataset_name="CIC-IDS2018")

    def _map_single_label(self, raw_str: str) -> tuple[int, str]:
        s = raw_str.strip().lower()
        if s in CIC_2018_LABEL_MAP:
            return CIC_2018_LABEL_MAP[s]
        elif "benign" in s:
            return (0, "BENIGN")
        elif "ddos" in s:
            return (1, "DDoS")
        elif "hulk" in s:
            return (1, "DoS Hulk")
        elif "goldeneye" in s:
            return (1, "DoS GoldenEye")
        elif "slowloris" in s:
            return (1, "DoS slowloris")
        elif "slowhttp" in s:
            return (1, "DoS Slowhttptest")
        elif "dos" in s:
            return (1, "DoS Hulk")
        elif "ftp" in s:
            return (1, "FTP-Patator")
        elif "ssh" in s:
            return (1, "SSH-Patator")
        elif "bot" in s:
            return (1, "Bot")
        elif "infilt" in s:
            return (1, "Infiltration")
        elif "sql" in s:
            return (1, "Web Attack - Sql Injection")
        elif "xss" in s:
            return (1, "Web Attack - XSS")
        elif "brute" in s or "web" in s:
            return (1, "Web Attack - Brute Force")
        else:
            return (1, "DDoS")

    def normalize_labels(self, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Normalize raw CIC-IDS2018 labels into binary_label and attack_category."""
        import sys
        unique_labels = labels.dropna().unique()
        bin_map: dict[Any, int] = {}
        cat_map: dict[Any, str] = {}
        for u in unique_labels:
            b, cat = self._map_single_label(str(u))
            bin_map[u] = b
            cat_map[u] = sys.intern(str(cat))

        binary_series = labels.map(bin_map).fillna(0).astype(int)
        cat_series = labels.map(cat_map).fillna("BENIGN")
        return binary_series, cat_series

    def load_and_normalize(
        self,
        data_paths: list[Path | str],
        max_rows: int | None = None,
        chunksize: int = 100_000,
    ) -> tuple[pd.DataFrame, DataQualityReport, DatasetProvenance]:
        """Load and normalize CIC-IDS2018 CSV flow files using memory-safe chunked streaming."""
        if not data_paths:
            raise ValueError("No data paths provided to CicIds2018Adapter")

        for p in data_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                raise FileNotFoundError(f"Dataset file not found: {path_obj}")

        # Chronological sort across distinct files
        sorted_paths = sorted(data_paths, key=_get_file_start_timestamp)

        file_hashes: dict[str, str] = {}
        total_bytes = 0
        for p in sorted_paths:
            path_obj = Path(p)
            total_bytes += path_obj.stat().st_size
            file_hashes[path_obj.name] = compute_file_sha256(path_obj)

        total_raw_rows = 0
        total_usable_rows = 0
        discard_reasons: dict[str, int] = {}
        total_missing_values = 0
        total_inf_values = 0
        protocol_dist: dict[int, int] = {}
        all_source_cols: set[str] = set()

        total_files = len(sorted_paths)
        start_wall_time = time.time()

        # Determine pre-allocation capacity to avoid duplicate array allocations
        if max_rows is not None:
            capacity = max(1, max_rows)
        else:
            print(f"Pre-scanning {total_files} files for row capacity...", flush=True)
            capacity = max(1, _count_lines(sorted_paths) + 10_000)

        all_features = np.empty((capacity, len(CANONICAL_FEATURES)), dtype=np.float32)
        all_labels = np.empty(capacity, dtype=np.int8)
        all_categories = np.empty(capacity, dtype=object)
        all_raw_labels = np.empty(capacity, dtype=object)

        # Temporary disk-backed memmap for timestamps:
        # Eliminates the 124 MiB in-RAM float64 allocation for 16M rows while preserving
        # full double-precision epoch timestamps and zero duplicate feature matrices.
        mmap_file = tempfile.NamedTemporaryFile(prefix="cyberheal_ts_cic_", suffix=".mmap", delete=False)
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

        for f_idx, p in enumerate(sorted_paths):
            path_obj = Path(p)
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            print(
                f"Processing file {f_idx + 1}/{total_files}: {path_obj.name} "
                f"({size_mb:.1f} MB) | rows so far: {total_usable_rows:,} | elapsed: {time.time() - start_wall_time:.1f}s",
                flush=True,
            )
            logger.info(
                "Processing file %d/%d: %s (%.1f MB)",
                f_idx + 1,
                total_files,
                path_obj.name,
                size_mb,
            )

            file_start_offset = total_usable_rows

            try:
                reader = pd.read_csv(
                    path_obj,
                    chunksize=chunksize,
                    low_memory=False,
                )
            except Exception as e:
                logger.error("Failed to read CSV %s: %s", path_obj, e)
                raise

            for chunk in reader:
                raw_chunk_len = len(chunk)
                total_raw_rows += raw_chunk_len
                all_source_cols.update(chunk.columns)

                if max_rows is not None and total_usable_rows >= max_rows:
                    break

                label_cols = [c for c in chunk.columns if "label" in c.lower()]
                if not label_cols:
                    raise ValueError(f"Could not find a 'Label' column in dataset CSV: {path_obj.name}")
                label_col = label_cols[0]

                # Identify repeated header rows
                header_mask = chunk[label_col].astype(str).str.strip().str.lower() == "label"
                n_repeats = int(header_mask.sum())
                if n_repeats > 0:
                    discard_reasons["repeated_header_rows"] = (
                        discard_reasons.get("repeated_header_rows", 0) + n_repeats
                    )
                    chunk = chunk[~header_mask]

                if len(chunk) == 0:
                    continue

                if max_rows is not None and total_usable_rows + len(chunk) > max_rows:
                    chunk = chunk.iloc[: max_rows - total_usable_rows]

                ts_cols = [c for c in chunk.columns if "time" in c.lower()]
                ts_col_raw = ts_cols[0] if ts_cols else None

                raw_labels_raw = chunk[label_col].astype(str).str.strip()
                raw_intern_map = {u: sys.intern(str(u)) for u in raw_labels_raw.unique()}
                raw_labels_s = raw_labels_raw.map(raw_intern_map)
                binary_s, category_s = self.normalize_labels(raw_labels_s)

                renamed = {c: CIC_COLUMN_ALIASES.get(c.strip(), c.strip()) for c in chunk.columns}
                chunk.rename(columns=renamed, inplace=True)

                if "fwd_pkt_len_var" not in chunk.columns and "fwd_pkt_len_std" in chunk.columns:
                    chunk["fwd_pkt_len_var"] = pd.to_numeric(chunk["fwd_pkt_len_std"], errors="coerce").fillna(0.0) ** 2
                if "bwd_pkt_len_var" not in chunk.columns and "bwd_pkt_len_std" in chunk.columns:
                    chunk["bwd_pkt_len_var"] = pd.to_numeric(chunk["bwd_pkt_len_std"], errors="coerce").fillna(0.0) ** 2
                if "pkt_len_var" not in chunk.columns and "pkt_len_std" in chunk.columns:
                    chunk["pkt_len_var"] = pd.to_numeric(chunk["pkt_len_std"], errors="coerce").fillna(0.0) ** 2
                if "flow_iat_total" not in chunk.columns:
                    if "fwd_iat_total" in chunk.columns and "bwd_iat_total" in chunk.columns:
                        chunk["flow_iat_total"] = (
                            pd.to_numeric(chunk["fwd_iat_total"], errors="coerce").fillna(0.0)
                            + pd.to_numeric(chunk["bwd_iat_total"], errors="coerce").fillna(0.0)
                        )
                    elif "flow_duration_us" in chunk.columns:
                        chunk["flow_iat_total"] = pd.to_numeric(chunk["flow_duration_us"], errors="coerce").fillna(0.0)
                if "fwd_bytes_per_s" not in chunk.columns and "tot_fwd_bytes" in chunk.columns and "flow_duration_us" in chunk.columns:
                    dur_s = np.maximum(pd.to_numeric(chunk["flow_duration_us"], errors="coerce").fillna(0.0).to_numpy() / 1e6, 1e-6)
                    chunk["fwd_bytes_per_s"] = pd.to_numeric(chunk["tot_fwd_bytes"], errors="coerce").fillna(0.0).to_numpy() / dur_s
                if "bwd_bytes_per_s" not in chunk.columns and "tot_bwd_bytes" in chunk.columns and "flow_duration_us" in chunk.columns:
                    dur_s = np.maximum(pd.to_numeric(chunk["flow_duration_us"], errors="coerce").fillna(0.0).to_numpy() / 1e6, 1e-6)
                    chunk["bwd_bytes_per_s"] = pd.to_numeric(chunk["tot_bwd_bytes"], errors="coerce").fillna(0.0).to_numpy() / dur_s

                ts_col_name = "flow_start_time" if "flow_start_time" in chunk.columns else ts_col_raw
                if ts_col_name and ts_col_name in chunk.columns:
                    raw_ts = chunk[ts_col_name]
                    num_ts = pd.to_numeric(raw_ts, errors="coerce")
                    if num_ts.notna().sum() > len(raw_ts) * 0.5:
                        parsed_ts = num_ts.fillna(0.0).to_numpy(dtype=np.float64)
                    else:
                        sample_val = str(raw_ts.dropna().iloc[0]) if len(raw_ts.dropna()) > 0 else ""
                        if "/" in sample_val:
                            dt_ts = pd.to_datetime(raw_ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")
                            if dt_ts.isna().sum() > len(raw_ts) * 0.1:
                                dt_ts = pd.to_datetime(raw_ts, dayfirst=True, errors="coerce")
                        else:
                            dt_ts = pd.to_datetime(raw_ts, errors="coerce")
                        parsed_ts = dt_ts.astype("datetime64[s]").astype("int64").astype(np.float64).to_numpy()
                        if np.isnan(parsed_ts).any():
                            parsed_ts = np.nan_to_num(parsed_ts, nan=0.0)
                else:
                    parsed_ts = np.arange(total_usable_rows, total_usable_rows + len(chunk), dtype=np.float64)

                if "protocol" in chunk.columns:
                    p_num = pd.to_numeric(chunk["protocol"], errors="coerce").fillna(6).astype(int)
                    for k, v in p_num.value_counts().items():
                        protocol_dist[int(k)] = protocol_dist.get(int(k), 0) + int(v)
                else:
                    protocol_dist[6] = protocol_dist.get(6, 0) + len(chunk)

                feat_mat = np.zeros((len(chunk), len(CANONICAL_FEATURES)), dtype=np.float32)
                for j, f in enumerate(CANONICAL_FEATURES):
                    if f in chunk.columns:
                        col_num = pd.to_numeric(chunk[f], errors="coerce")
                        nan_cnt = int(col_num.isna().sum())
                        if nan_cnt > 0:
                            total_missing_values += nan_cnt
                        vals = col_num.to_numpy(dtype=np.float32, na_value=0.0)
                        inf_cnt = int(np.isinf(vals).sum())
                        if inf_cnt > 0:
                            total_inf_values += inf_cnt
                        np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        feat_mat[:, j] = vals
                    else:
                        feat_mat[:, j] = 0.0

                chunk_len = len(chunk)
                # Expand capacity if pre-scan undercounted slightly
                if total_usable_rows + chunk_len > len(all_features):
                    new_cap = max(total_usable_rows + chunk_len, int(len(all_features) * 1.25))
                    all_features = np.resize(all_features, (new_cap, len(CANONICAL_FEATURES)))
                    all_labels = np.resize(all_labels, new_cap)
                    all_categories = np.resize(all_categories, new_cap)
                    all_raw_labels = np.resize(all_raw_labels, new_cap)

                all_features[total_usable_rows : total_usable_rows + chunk_len] = feat_mat
                all_timestamps[total_usable_rows : total_usable_rows + chunk_len] = parsed_ts
                all_labels[total_usable_rows : total_usable_rows + chunk_len] = binary_s.to_numpy(dtype=np.int8)
                all_categories[total_usable_rows : total_usable_rows + chunk_len] = category_s.to_numpy(dtype=object)
                all_raw_labels[total_usable_rows : total_usable_rows + chunk_len] = raw_labels_s.to_numpy(dtype=object)

                total_usable_rows += chunk_len

            # Consolidate and sort this file's slice chronologically if out-of-order
            file_end_offset = total_usable_rows
            file_len = file_end_offset - file_start_offset
            if file_len > 1:
                file_ts = all_timestamps[file_start_offset:file_end_offset]
                if not np.all(file_ts[:-1] <= file_ts[1:]):
                    file_idx = np.argsort(file_ts, kind="stable")
                    all_timestamps[file_start_offset:file_end_offset] = file_ts[file_idx]
                    all_features[file_start_offset:file_end_offset] = all_features[file_start_offset:file_end_offset][file_idx]
                    all_labels[file_start_offset:file_end_offset] = all_labels[file_start_offset:file_end_offset][file_idx]
                    all_categories[file_start_offset:file_end_offset] = all_categories[file_start_offset:file_end_offset][file_idx]
                    all_raw_labels[file_start_offset:file_end_offset] = all_raw_labels[file_start_offset:file_end_offset][file_idx]

            if max_rows is not None and total_usable_rows >= max_rows:
                print(f"Reached max_rows budget ({max_rows:,}). Stopping ingestion.", flush=True)
                break

        N = total_usable_rows
        if N == 0:
            raise ValueError("No usable data rows extracted from provided dataset files.")

        all_features = all_features[:N]
        all_labels = all_labels[:N]
        all_categories = all_categories[:N]
        all_raw_labels = all_raw_labels[:N]
        all_timestamps = all_timestamps[:N]
        all_timestamps.flush()

        # Enforce non-decreasing chronological ordering across timestamps without memory duplication
        if N > 1 and not np.all(all_timestamps[:-1] <= all_timestamps[1:]):
            np.maximum.accumulate(all_timestamps, out=all_timestamps)
            all_timestamps.flush()

        # Zero-copy DataFrame construction: passing 1D column views prevents Pandas
        # from transposing and reallocating a second 4.72 GiB 2D block for all_features.
        col_dict: dict[str, Any] = {c: all_features[:, i] for i, c in enumerate(CANONICAL_FEATURES)}
        col_dict["flow_start_time"] = all_timestamps
        col_dict["label_int"] = all_labels
        col_dict["attack_class"] = pd.Categorical(all_categories)
        col_dict["raw_label"] = pd.Categorical(all_raw_labels)
        col_dict["scenario_id"] = "cic_ids_2018"
        if N <= 100_000:
            col_dict["flow_id"] = [f"flow_{i}" for i in range(N)]
        else:
            col_dict["flow_id"] = np.arange(N, dtype=np.int64)

        clean_df = pd.DataFrame(col_dict, copy=False)

        clean_df.attrs["features_holder"] = ArrayHolder(all_features)
        clean_df.attrs["labels_holder"] = ArrayHolder(all_labels)
        clean_df.attrs["timestamps_holder"] = ArrayHolder(all_timestamps)
        clean_df.attrs["categories_holder"] = ArrayHolder(all_categories)
        clean_df.attrs["memmap_path"] = mmap_path
        clean_df.attrs["cleanup_mmap"] = _cleanup_mmap

        benign_count = int((all_labels == 0).sum())
        attack_count = int((all_labels == 1).sum())
        attack_ratio = float(attack_count) / float(N) if N > 0 else 0.0

        attack_cats = {str(k): int(v) for k, v in clean_df["attack_class"].value_counts().items()}

        ts_min = float(clean_df["flow_start_time"].min()) if N > 0 else None
        ts_max = float(clean_df["flow_start_time"].max()) if N > 0 else None

        aliased_sources = {CIC_COLUMN_ALIASES.get(c.strip(), c.strip()) for c in all_source_cols}
        mapped_feats = aliased_sources.intersection(set(CANONICAL_FEATURES))
        feature_coverage = float(len(mapped_feats)) / float(CANONICAL_FEATURE_DIM)

        quality_report = DataQualityReport(
            dataset_name=self.dataset_name,
            total_rows=total_raw_rows,
            usable_rows=N,
            discarded_rows=total_raw_rows - N,
            discard_reasons=discard_reasons,
            benign_count=benign_count,
            attack_count=attack_count,
            attack_ratio=attack_ratio,
            attack_categories=attack_cats,
            missing_value_count=total_missing_values,
            infinite_value_count=total_inf_values,
            duplicate_record_count=0,
            timestamp_start=ts_min,
            timestamp_end=ts_max,
            protocol_distribution=protocol_dist,
            feature_coverage_ratio=feature_coverage,
        )

        provenance = DatasetProvenance(
            dataset_name=self.dataset_name,
            source_files=[Path(p).name for p in sorted_paths],
            total_bytes=total_bytes,
            file_hashes_sha256=file_hashes,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            row_count=N,
            input_format="csv",
        )

        return clean_df, quality_report, provenance


class Ctu13Adapter(DatasetAdapter):
    """Adapter for CTU-13 Botnet flow and PCAP datasets."""

    def __init__(self) -> None:
        super().__init__(dataset_name="CTU-13")

    def _map_single_label(self, raw_str: str) -> tuple[int, str]:
        s = raw_str.strip().lower()
        if "botnet" in s or "c&c" in s:
            return (1, "Bot")
        elif "attack" in s or "ddos" in s:
            return (1, "DDoS")
        elif "portscan" in s:
            return (1, "PortScan")
        elif "normal" in s or "background" in s:
            return (0, "BENIGN")
        else:
            return (0, "BENIGN")

    def normalize_labels(self, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Normalize raw CTU-13 labels into binary_label and attack_category."""
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
    ) -> tuple[pd.DataFrame, DataQualityReport, DatasetProvenance]:
        """Load and normalize CTU-13 CSV / NetFlow files using memory-safe streaming."""
        if not data_paths:
            raise ValueError("No data paths provided to Ctu13Adapter")

        for p in data_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                raise FileNotFoundError(f"Dataset file not found: {path_obj}")

        sorted_paths = sorted(data_paths, key=_get_file_start_timestamp)

        file_hashes: dict[str, str] = {}
        total_bytes = 0
        for p in sorted_paths:
            path_obj = Path(p)
            total_bytes += path_obj.stat().st_size
            file_hashes[path_obj.name] = compute_file_sha256(path_obj)

        # Determine pre-allocation capacity to avoid duplicate array allocations
        if max_rows is not None:
            capacity = max(1, max_rows)
        else:
            capacity = max(1, _count_lines(sorted_paths) + 10_000)

        all_features = np.empty((capacity, len(CANONICAL_FEATURES)), dtype=np.float32)
        all_labels = np.empty(capacity, dtype=np.int8)
        all_categories = np.empty(capacity, dtype=object)
        all_raw_labels = np.empty(capacity, dtype=object)

        mmap_file = tempfile.NamedTemporaryFile(prefix="cyberheal_ts_ctu_", suffix=".mmap", delete=False)
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

        total_raw_rows = 0
        total_usable_rows = 0
        discard_reasons: dict[str, int] = {}
        total_missing_values = 0
        total_inf_values = 0
        protocol_dist: dict[int, int] = {}
        all_source_cols: set[str] = set()

        total_files = len(sorted_paths)
        start_wall_time = time.time()

        for f_idx, p in enumerate(sorted_paths):
            path_obj = Path(p)
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            print(
                f"Processing file {f_idx + 1}/{total_files}: {path_obj.name} "
                f"({size_mb:.1f} MB) | rows so far: {total_usable_rows:,} | elapsed: {time.time() - start_wall_time:.1f}s",
                flush=True,
            )

            file_start_offset = total_usable_rows

            try:
                reader = pd.read_csv(
                    path_obj,
                    chunksize=chunksize,
                    low_memory=False,
                )
            except Exception as e:
                logger.error("Failed to read CSV %s: %s", path_obj, e)
                raise

            for chunk in reader:
                raw_chunk_len = len(chunk)
                total_raw_rows += raw_chunk_len
                all_source_cols.update(chunk.columns)

                if max_rows is not None and total_usable_rows >= max_rows:
                    break

                label_cols = [c for c in chunk.columns if c.lower() in ("label", "tag", "class")]
                if not label_cols:
                    raise ValueError(f"Could not find a Label/Tag column in CTU-13 dataset CSV: {path_obj.name}")
                label_col = label_cols[0]

                if max_rows is not None and total_usable_rows + len(chunk) > max_rows:
                    chunk = chunk.iloc[: max_rows - total_usable_rows]

                raw_labels_s = chunk[label_col].astype(str).str.strip()
                binary_s, category_s = self.normalize_labels(raw_labels_s)

                ts_cols = [c for c in chunk.columns if "time" in c.lower()]
                ts_col_raw = ts_cols[0] if ts_cols else None

                renamed = {c: CIC_COLUMN_ALIASES.get(c.strip(), c.strip()) for c in chunk.columns}
                chunk.rename(columns=renamed, inplace=True)

                if "fwd_pkt_len_var" not in chunk.columns and "fwd_pkt_len_std" in chunk.columns:
                    chunk["fwd_pkt_len_var"] = pd.to_numeric(chunk["fwd_pkt_len_std"], errors="coerce").fillna(0.0) ** 2
                if "bwd_pkt_len_var" not in chunk.columns and "bwd_pkt_len_std" in chunk.columns:
                    chunk["bwd_pkt_len_var"] = pd.to_numeric(chunk["bwd_pkt_len_std"], errors="coerce").fillna(0.0) ** 2
                if "pkt_len_var" not in chunk.columns and "pkt_len_std" in chunk.columns:
                    chunk["pkt_len_var"] = pd.to_numeric(chunk["pkt_len_std"], errors="coerce").fillna(0.0) ** 2
                if "flow_iat_total" not in chunk.columns:
                    if "fwd_iat_total" in chunk.columns and "bwd_iat_total" in chunk.columns:
                        chunk["flow_iat_total"] = (
                            pd.to_numeric(chunk["fwd_iat_total"], errors="coerce").fillna(0.0)
                            + pd.to_numeric(chunk["bwd_iat_total"], errors="coerce").fillna(0.0)
                        )
                    elif "flow_duration_us" in chunk.columns:
                        chunk["flow_iat_total"] = pd.to_numeric(chunk["flow_duration_us"], errors="coerce").fillna(0.0)
                if "fwd_bytes_per_s" not in chunk.columns and "tot_fwd_bytes" in chunk.columns and "flow_duration_us" in chunk.columns:
                    dur_s = np.maximum(pd.to_numeric(chunk["flow_duration_us"], errors="coerce").fillna(0.0).to_numpy() / 1e6, 1e-6)
                    chunk["fwd_bytes_per_s"] = pd.to_numeric(chunk["tot_fwd_bytes"], errors="coerce").fillna(0.0).to_numpy() / dur_s
                if "bwd_bytes_per_s" not in chunk.columns and "tot_bwd_bytes" in chunk.columns and "flow_duration_us" in chunk.columns:
                    dur_s = np.maximum(pd.to_numeric(chunk["flow_duration_us"], errors="coerce").fillna(0.0).to_numpy() / 1e6, 1e-6)
                    chunk["bwd_bytes_per_s"] = pd.to_numeric(chunk["tot_bwd_bytes"], errors="coerce").fillna(0.0).to_numpy() / dur_s
                ts_col_name = "flow_start_time" if "flow_start_time" in chunk.columns else ts_col_raw
                if ts_col_name and ts_col_name in chunk.columns:
                    raw_ts = chunk[ts_col_name]
                    num_ts = pd.to_numeric(raw_ts, errors="coerce")
                    if num_ts.notna().sum() > len(raw_ts) * 0.5:
                        parsed_ts = num_ts.fillna(0.0).to_numpy(dtype=np.float64)
                    else:
                        dt_ts = pd.to_datetime(raw_ts, errors="coerce")
                        parsed_ts = dt_ts.astype("datetime64[s]").astype("int64").astype(np.float64).to_numpy()
                        if np.isnan(parsed_ts).any():
                            parsed_ts = np.nan_to_num(parsed_ts, nan=0.0)
                else:
                    parsed_ts = np.arange(total_usable_rows, total_usable_rows + len(chunk), dtype=np.float64)

                if "protocol" in chunk.columns:
                    p_num = pd.to_numeric(chunk["protocol"], errors="coerce").fillna(6).astype(int)
                    for k, v in p_num.value_counts().items():
                        protocol_dist[int(k)] = protocol_dist.get(int(k), 0) + int(v)
                else:
                    protocol_dist[6] = protocol_dist.get(6, 0) + len(chunk)

                feat_mat = np.zeros((len(chunk), len(CANONICAL_FEATURES)), dtype=np.float32)
                for j, f in enumerate(CANONICAL_FEATURES):
                    if f in chunk.columns:
                        col_num = pd.to_numeric(chunk[f], errors="coerce")
                        nan_cnt = int(col_num.isna().sum())
                        if nan_cnt > 0:
                            total_missing_values += nan_cnt
                        vals = col_num.to_numpy(dtype=np.float32, na_value=0.0)
                        inf_cnt = int(np.isinf(vals).sum())
                        if inf_cnt > 0:
                            total_inf_values += inf_cnt
                        np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        feat_mat[:, j] = vals
                    else:
                        feat_mat[:, j] = 0.0

                chunk_len = len(chunk)
                if total_usable_rows + chunk_len > len(all_features):
                    new_cap = max(total_usable_rows + chunk_len, int(len(all_features) * 1.25))
                    all_features = np.resize(all_features, (new_cap, len(CANONICAL_FEATURES)))
                    all_labels = np.resize(all_labels, new_cap)
                    all_categories = np.resize(all_categories, new_cap)
                    all_raw_labels = np.resize(all_raw_labels, new_cap)

                all_features[total_usable_rows : total_usable_rows + chunk_len] = feat_mat
                all_timestamps[total_usable_rows : total_usable_rows + chunk_len] = parsed_ts
                all_labels[total_usable_rows : total_usable_rows + chunk_len] = binary_s.to_numpy(dtype=np.int8)
                all_categories[total_usable_rows : total_usable_rows + chunk_len] = category_s.to_numpy(dtype=object)
                all_raw_labels[total_usable_rows : total_usable_rows + chunk_len] = raw_labels_s.to_numpy(dtype=object)

                total_usable_rows += chunk_len

            # Consolidate and sort this file's slice chronologically if out-of-order
            file_end_offset = total_usable_rows
            file_len = file_end_offset - file_start_offset
            if file_len > 1:
                file_ts = all_timestamps[file_start_offset:file_end_offset]
                if not np.all(file_ts[:-1] <= file_ts[1:]):
                    file_idx = np.argsort(file_ts, kind="stable")
                    all_timestamps[file_start_offset:file_end_offset] = file_ts[file_idx]
                    all_features[file_start_offset:file_end_offset] = all_features[file_start_offset:file_end_offset][file_idx]
                    all_labels[file_start_offset:file_end_offset] = all_labels[file_start_offset:file_end_offset][file_idx]
                    all_categories[file_start_offset:file_end_offset] = all_categories[file_start_offset:file_end_offset][file_idx]
                    all_raw_labels[file_start_offset:file_end_offset] = all_raw_labels[file_start_offset:file_end_offset][file_idx]

            if max_rows is not None and total_usable_rows >= max_rows:
                break

        N = total_usable_rows
        if N == 0:
            raise ValueError("No usable data rows extracted from provided dataset files.")

        all_features = all_features[:N]
        all_labels = all_labels[:N]
        all_categories = all_categories[:N]
        all_raw_labels = all_raw_labels[:N]
        all_timestamps = all_timestamps[:N]
        all_timestamps.flush()

        if N > 1 and not np.all(all_timestamps[:-1] <= all_timestamps[1:]):
            np.maximum.accumulate(all_timestamps, out=all_timestamps)
            all_timestamps.flush()

        col_dict: dict[str, Any] = {c: all_features[:, i] for i, c in enumerate(CANONICAL_FEATURES)}
        col_dict["flow_start_time"] = all_timestamps
        col_dict["label_int"] = all_labels
        col_dict["attack_class"] = pd.Categorical(all_categories)
        col_dict["raw_label"] = pd.Categorical(all_raw_labels)
        col_dict["scenario_id"] = "ctu_13"
        if N <= 100_000:
            col_dict["flow_id"] = [f"flow_{i}" for i in range(N)]
        else:
            col_dict["flow_id"] = np.arange(N, dtype=np.int64)

        clean_df = pd.DataFrame(col_dict, copy=False)

        clean_df.attrs["features_holder"] = ArrayHolder(all_features)
        clean_df.attrs["labels_holder"] = ArrayHolder(all_labels)
        clean_df.attrs["timestamps_holder"] = ArrayHolder(all_timestamps)
        clean_df.attrs["categories_holder"] = ArrayHolder(all_categories)
        clean_df.attrs["memmap_path"] = mmap_path
        clean_df.attrs["cleanup_mmap"] = _cleanup_mmap

        benign_count = int((all_labels == 0).sum())
        attack_count = int((all_labels == 1).sum())
        attack_ratio = float(attack_count) / float(N) if N > 0 else 0.0

        attack_cats = {str(k): int(v) for k, v in clean_df["attack_class"].value_counts().items()}
        ts_min = float(clean_df["flow_start_time"].min()) if N > 0 else None
        ts_max = float(clean_df["flow_start_time"].max()) if N > 0 else None

        aliased_sources = {CIC_COLUMN_ALIASES.get(c.strip(), c.strip()) for c in all_source_cols}
        mapped_feats = aliased_sources.intersection(set(CANONICAL_FEATURES))
        feature_coverage = float(len(mapped_feats)) / float(CANONICAL_FEATURE_DIM)

        quality_report = DataQualityReport(
            dataset_name=self.dataset_name,
            total_rows=total_raw_rows,
            usable_rows=N,
            discarded_rows=total_raw_rows - N,
            discard_reasons=discard_reasons,
            benign_count=benign_count,
            attack_count=attack_count,
            attack_ratio=attack_ratio,
            attack_categories=attack_cats,
            missing_value_count=total_missing_values,
            infinite_value_count=total_inf_values,
            duplicate_record_count=0,
            timestamp_start=ts_min,
            timestamp_end=ts_max,
            protocol_distribution=protocol_dist,
            feature_coverage_ratio=feature_coverage,
        )

        provenance = DatasetProvenance(
            dataset_name=self.dataset_name,
            source_files=[Path(p).name for p in sorted_paths],
            total_bytes=total_bytes,
            file_hashes_sha256=file_hashes,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            row_count=N,
            input_format="csv",
        )

        return clean_df, quality_report, provenance


def get_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    """Factory to instantiate the appropriate DatasetAdapter."""
    norm = dataset_name.lower().replace("-", "").replace("_", "")
    if "cicids2018" in norm or "csecicids2018" in norm or "cic2018" in norm:
        return CicIds2018Adapter()
    elif "ctu13" in norm or "ctu" in norm:
        return Ctu13Adapter()
    else:
        raise ValueError(
            f"Unsupported dataset: '{dataset_name}'. "
            f"Supported datasets: 'cic-ids2018', 'ctu13'"
        )
