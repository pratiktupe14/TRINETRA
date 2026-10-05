"""Tests for Memory-Safe Chunked Dataset Loader (Phase 9A Fix).

Validates:
1. Chunked CSV loading with small chunk sizes (multiple chunks).
2. Robust handling of mixed-type columns, non-numeric strings, and repeated header rows.
3. Strict preservation of 78-D canonical numerical features and ordering.
4. Dual label normalization (binary 0/1 and canonical attack categories).
5. Chronological file and flow ordering without lookahead bias.
6. Handling of missing, NaN, and Infinite values.
7. max_rows bounding during chunked ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from shield_rca.evaluation.dataset_adapters import (
    CicIds2018Adapter,
    Ctu13Adapter,
    DataQualityReport,
    DatasetProvenance,
)
from shield_rca.telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES


@pytest.fixture
def chunked_synthetic_csv(tmp_path: Path) -> Path:
    """Create a synthetic CSV with 50 rows designed for chunked testing."""
    file_path = tmp_path / "test_chunked_flows.csv"
    data: dict[str, Any] = {
        "Destination Port": [80, 443, 22, 53, 8080] * 10,
        "Flow Duration": [1000 + i * 100 for i in range(50)],
        "Tot Fwd Pkts": [5 + (i % 5) for i in range(50)],
        "Tot Bwd Pkts": [3 + (i % 3) for i in range(50)],
        "TotLen Fwd Pkts": [500 + i * 10 for i in range(50)],
        "TotLen Bwd Pkts": [300 + i * 10 for i in range(50)],
        "Protocol": [6, 17, 6, 17, 6] * 10,
        "Timestamp": [f"14/02/2018 08:{i:02d}:00" for i in range(50)],
        "Label": ["Benign", "DDoS attacks-LOIC-HTTP", "Bot", "DoS attacks-Hulk", "Benign"] * 10,
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def dirty_mixed_type_csv(tmp_path: Path) -> Path:
    """Create a synthetic CSV containing mixed-type columns, NaNs, Infs, and repeated headers."""
    file_path = tmp_path / "dirty_mixed_types.csv"
    lines = [
        "Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Flow Byts/s,Label",
        "80,6,14/02/2018 08:00:00,1000,5,250.5,Benign",
        "443,6,14/02/2018 08:01:00,2000,10,500.0,DDoS attacks-LOIC-HTTP",
        "Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Flow Byts/s,Label",  # Repeated header
        "22,6,14/02/2018 08:02:00,Infinity,invalid_str,NaN,SSH-Bruteforce",        # Non-numeric & Inf
        "53,17,14/02/2018 08:03:00,4000,2,100.0,Benign",
        "8080,6,14/02/2018 08:04:00,-500,4,Infinity,Bot",                           # Negative & Inf
        "Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Flow Byts/s,Label",  # Repeated header
        "80,6,14/02/2018 08:05:00,3000,8,400.0,Benign",
    ]
    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


# ── 1. Chunked Loading with Multiple Chunks ───────────────────────────────────
def test_chunked_csv_loading_multiple_chunks(chunked_synthetic_csv: Path) -> None:
    adapter = CicIds2018Adapter()
    chunk_size = 12  # 50 rows / 12 = 5 chunks

    clean_df, quality, prov = adapter.load_and_normalize(
        [chunked_synthetic_csv],
        chunksize=chunk_size,
    )

    assert len(clean_df) == 50
    assert quality.total_rows == 50
    assert quality.usable_rows == 50
    assert quality.discarded_rows == 0
    assert prov.row_count == 50

    # Ensure canonical columns exist
    for feat in CANONICAL_FEATURES:
        assert feat in clean_df.columns
        assert clean_df[feat].dtype == np.float32

    # Verify binary and category labels
    assert (clean_df["label_int"] == 0).sum() == 20
    assert (clean_df["label_int"] == 1).sum() == 30
    assert "BENIGN" in quality.attack_categories
    assert "DDoS" in quality.attack_categories
    assert "Bot" in quality.attack_categories
    assert "DoS Hulk" in quality.attack_categories


# ── 2. Mixed-Type and Repeated Header Handling ────────────────────────────────
def test_mixed_type_and_repeated_headers(dirty_mixed_type_csv: Path) -> None:
    adapter = CicIds2018Adapter()
    clean_df, quality, prov = adapter.load_and_normalize(
        [dirty_mixed_type_csv],
        chunksize=3,  # Tiny chunk size to split repeated headers across chunks
    )

    # 8 total rows read by chunks - 2 repeated header rows = 6 valid data rows
    assert quality.total_rows == 8
    assert quality.usable_rows == 6
    assert quality.discarded_rows == 2
    assert quality.discard_reasons.get("repeated_header_rows") == 2
    assert len(clean_df) == 6

    # Check finite float32 guarantees
    num_cols = list(CANONICAL_FEATURES)
    mat = clean_df[num_cols].to_numpy()
    assert np.isfinite(mat).all(), "Feature matrix contains non-finite values!"
    assert clean_df["flow_duration_us"].iloc[2] == 0.0  # Coerced from Infinity
    assert clean_df["tot_fwd_pkts"].iloc[2] == 0.0     # Coerced from 'invalid_str'


# ── 3. Canonical 78-D Feature Extraction ──────────────────────────────────────
def test_canonical_78d_feature_extraction(chunked_synthetic_csv: Path) -> None:
    adapter = CicIds2018Adapter()
    clean_df, _, _ = adapter.load_and_normalize([chunked_synthetic_csv], chunksize=20)

    # Schema must match CANONICAL_FEATURE_DIM (78)
    feat_cols = [c for c in clean_df.columns if c in CANONICAL_FEATURES]
    assert len(feat_cols) == CANONICAL_FEATURE_DIM

    # Features must match canonical alphabetical order
    assert tuple(clean_df.columns[:CANONICAL_FEATURE_DIM]) == CANONICAL_FEATURES


# ── 4. Label Normalization ────────────────────────────────────────────────────
def test_label_normalization_and_categories() -> None:
    adapter = CicIds2018Adapter()
    raw = pd.Series([
        "Benign",
        "DDoS attacks-LOIC-HTTP",
        "DDOS attack-HOIC",
        "DoS attacks-Hulk",
        "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris",
        "DoS attacks-SlowHTTPTest",
        "FTP-BruteForce",
        "SSH-Bruteforce",
        "Bot",
        "Infiltration",
        "Brute Force -Web",
        "Brute Force -XSS",
        "SQL Injection",
        "UnknownAttackPattern",
    ])
    bin_labels, cat_labels = adapter.normalize_labels(raw)

    assert bin_labels.iloc[0] == 0
    assert cat_labels.iloc[0] == "BENIGN"

    for i in range(1, len(raw)):
        assert bin_labels.iloc[i] == 1, f"Expected attack for {raw.iloc[i]}"

    assert cat_labels.iloc[1] == "DDoS"
    assert cat_labels.iloc[3] == "DoS Hulk"
    assert cat_labels.iloc[7] == "FTP-Patator"
    assert cat_labels.iloc[8] == "SSH-Patator"
    assert cat_labels.iloc[9] == "Bot"
    assert cat_labels.iloc[10] == "Infiltration"
    assert cat_labels.iloc[11] == "Web Attack - Brute Force"
    assert cat_labels.iloc[13] == "Web Attack - Sql Injection"


# ── 5. Chronological Ordering Across Files ────────────────────────────────────
def test_chronological_ordering_across_files(tmp_path: Path) -> None:
    # Create two files with out-of-order timestamps
    file_day2 = tmp_path / "day2.csv"
    file_day1 = tmp_path / "day1.csv"

    # Day 2: 15/02/2018
    df2 = pd.DataFrame({
        "Dst Port": [80, 443],
        "Flow Duration": [1000, 2000],
        "Timestamp": ["15/02/2018 10:00:00", "15/02/2018 11:00:00"],
        "Label": ["Benign", "Bot"],
    })
    df2.to_csv(file_day2, index=False)

    # Day 1: 14/02/2018
    df1 = pd.DataFrame({
        "Dst Port": [22, 53],
        "Flow Duration": [3000, 4000],
        "Timestamp": ["14/02/2018 10:00:00", "14/02/2018 11:00:00"],
        "Label": ["Benign", "DDoS"],
    })
    df1.to_csv(file_day1, index=False)

    adapter = CicIds2018Adapter()
    # Pass in reverse chronological order: [day2, day1]
    clean_df, _, prov = adapter.load_and_normalize([file_day2, file_day1], chunksize=10)

    # Ensure Day 1 rows precede Day 2 rows
    assert prov.source_files == ["day1.csv", "day2.csv"]
    assert clean_df["flow_start_time"].is_monotonic_increasing
    assert clean_df["flow_start_time"].iloc[0] < clean_df["flow_start_time"].iloc[-1]


# ── 6. max_rows Budget Bounding ───────────────────────────────────────────────
def test_max_rows_budget_with_chunking(chunked_synthetic_csv: Path) -> None:
    adapter = CicIds2018Adapter()
    clean_df, quality, prov = adapter.load_and_normalize(
        [chunked_synthetic_csv],
        max_rows=17,
        chunksize=5,
    )

    assert len(clean_df) == 17
    assert quality.usable_rows == 17
    assert prov.row_count == 17


# ── 7. Timestamp Memory Safety Regression Test ────────────────────────────────
def test_no_full_dataset_float64_in_memory_timestamp_allocation(
    chunked_synthetic_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test proving load_and_normalize() does not allocate a full float64 timestamp array in RAM.

    Verifies:
    1. No np.empty or buffer with dtype=float64 proportional to total rows/capacity is allocated in RAM.
    2. clean_df["flow_start_time"] is backed by a disk memmap, eliminating RAM commit charge.
    3. Timestamps maintain full float64 epoch precision.
    4. Chronological monotonicity is strictly preserved.
    5. Temporary memmap file is cleaned up after cleanup_mmap() is invoked.
    """
    import os
    original_empty = np.empty
    allocated_float64_1d_sizes: list[int] = []

    def tracking_empty(shape: Any, dtype: Any = float, *args: Any, **kwargs: Any) -> np.ndarray:
        arr = original_empty(shape, dtype, *args, **kwargs)
        # Record any 1D float64 in-memory allocations
        dt = np.dtype(dtype)
        if dt == np.float64:
            if isinstance(shape, int):
                allocated_float64_1d_sizes.append(shape)
            elif isinstance(shape, tuple) and len(shape) == 1:
                allocated_float64_1d_sizes.append(shape[0])
        return arr

    monkeypatch.setattr(np, "empty", tracking_empty)

    adapter = CicIds2018Adapter()
    clean_df, quality, prov = adapter.load_and_normalize(
        [chunked_synthetic_csv],
        chunksize=10,
    )

    # Total dataset rows in chunked_synthetic_csv fixture is 50
    assert len(clean_df) == 50

    # Requirement 1, 2, 3: flow_start_time must be float64 with full epoch precision
    assert clean_df["flow_start_time"].dtype == np.float64
    assert clean_df["flow_start_time"].is_monotonic_increasing
    assert clean_df["flow_start_time"].iloc[0] < clean_df["flow_start_time"].iloc[-1]

    # Must be backed by a disk memmap rather than in-RAM contiguous buffer
    raw_array = clean_df["flow_start_time"].values
    assert isinstance(raw_array, np.memmap) or isinstance(getattr(raw_array, "base", None), np.memmap), (
        "Expected flow_start_time to be backed by np.memmap"
    )

    # Requirement 8: Must NOT allocate an in-RAM float64 timestamp array proportional to dataset capacity
    assert not any(size >= 50 for size in allocated_float64_1d_sizes), (
        f"Violated memory safety: detected in-memory float64 1D array allocation of size {allocated_float64_1d_sizes} "
        f"proportional to dataset capacity!"
    )

    # Requirement 7: Test temporary memmap file cleanup
    memmap_path = clean_df.attrs.get("memmap_path")
    assert memmap_path is not None and os.path.exists(memmap_path)
    cleanup_fn = clean_df.attrs.get("cleanup_mmap")
    assert cleanup_fn is not None

    del clean_df
    del raw_array
    cleanup_fn()
    assert not os.path.exists(memmap_path), f"Temporary memmap file {memmap_path} was not cleaned up!"


