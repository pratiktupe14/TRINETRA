"""Tests for TRINETRA Real-Data Evaluation & Benchmarking Engine (Phase 9).

Validates all 20 required aspects of dataset adapters, provenance tracking,
leakage-free chronological splitting, supervised baselines, World Model evaluation,
unseen attack holdouts, explainability, dry-run remediation gating, and artifact isolation.

CRITICAL MANDATE:
All fixtures in this file are synthetic and explicitly labeled as:
'SYNTHETIC SOFTWARE VALIDATION'.
They must NEVER be mixed into real-world SIH benchmark reporting tables.
"""

from __future__ import annotations

import hashlib
import json
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
    compute_file_sha256,
    get_dataset_adapter,
)
from shield_rca.evaluation.real_data_benchmark import (
    RealDataBenchmarkRunner,
    discover_dataset,
)
from shield_rca.telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES


@pytest.fixture
def synthetic_cic_df() -> pd.DataFrame:
    """Generate deterministic synthetic CIC-IDS2018 style DataFrame (80 rows)."""
    n_rows = 80
    data: dict[str, Any] = {
        "Destination Port": [80, 443, 22, 53] * 20,
        "Flow Duration": [1000, 2000, 3000, 4000] * 20,
        "Total Fwd Packets": [10, 20, 5, 8] * 20,
        "Total Backward Packets": [8, 18, 4, 6] * 20,
        "Total Length of Fwd Packets": [1000, 2500, 400, 600] * 20,
        "Total Length of Bwd Packets": [1500, 3000, 500, 800] * 20,
        "Flow Bytes/s": [2500.0, 5500.0, 900.0, 1400.0] * 20,
        "Flow Packets/s": [18.0, 38.0, 9.0, 14.0] * 20,
        "Timestamp": [f"2018-02-14 10:{i:02d}:00" for i in range(n_rows)],
        "Label": ["Benign", "DDoS attacks-LOIC-HTTP", "Bot", "Benign"] * 20,
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_cic_dir(tmp_path: Path, synthetic_cic_df: pd.DataFrame) -> Path:
    """Create a temporary directory with synthetic CIC-IDS2018 CSV file."""
    data_dir = tmp_path / "cic_ids2018_synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / "02-14-2018.csv"
    synthetic_cic_df.to_csv(csv_file, index=False)
    return data_dir


# ── 1. Dataset Discovery ──────────────────────────────────────────────────────
def test_dataset_discovery(synthetic_cic_dir: Path) -> None:
    detected_format, files = discover_dataset(synthetic_cic_dir, input_format="auto")
    assert detected_format == "csv"
    assert len(files) == 1
    assert files[0].name == "02-14-2018.csv"


# ── 2. Missing Dataset Handling ───────────────────────────────────────────────
def test_missing_dataset_handling(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist"
    detected_format, files = discover_dataset(non_existent)
    assert detected_format == "not_found"
    assert len(files) == 0

    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=non_existent,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    with pytest.raises(FileNotFoundError, match="Real dataset not found"):
        runner.run_benchmark()


# ── 3. Dataset Adapter Selection ──────────────────────────────────────────────
def test_dataset_adapter_selection() -> None:
    cic_adapter = get_dataset_adapter("cic-ids2018")
    assert isinstance(cic_adapter, CicIds2018Adapter)

    ctu_adapter = get_dataset_adapter("ctu13")
    assert isinstance(ctu_adapter, Ctu13Adapter)

    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_dataset_adapter("unknown-dataset-xyz")


# ── 4. Label Normalization ────────────────────────────────────────────────────
def test_label_normalization() -> None:
    cic_adapter = CicIds2018Adapter()
    raw_labels = pd.Series(["Benign", "DDoS attacks-LOIC-HTTP", "Bot", "DoS attacks-Hulk"])
    binary, cats = cic_adapter.normalize_labels(raw_labels)
    assert list(binary) == [0, 1, 1, 1]
    assert list(cats) == ["BENIGN", "DDoS", "Bot", "DoS Hulk"]

    ctu_adapter = Ctu13Adapter()
    raw_ctu = pd.Series(["normal", "c&c-mirai", "attack", "portscan"])
    ctu_bin, ctu_cats = ctu_adapter.normalize_labels(raw_ctu)
    assert list(ctu_bin) == [0, 1, 1, 1]
    assert list(ctu_cats) == ["BENIGN", "Bot", "DDoS", "PortScan"]


# ── 5. Data Quality Report ────────────────────────────────────────────────────
def test_data_quality_report(synthetic_cic_dir: Path) -> None:
    adapter = CicIds2018Adapter()
    files = sorted(synthetic_cic_dir.glob("*.csv"))
    df, quality, provenance = adapter.load_and_normalize(files)

    assert quality.total_rows == 80
    assert quality.usable_rows == 80
    assert quality.discarded_rows == 0
    assert quality.benign_count == 40
    assert quality.attack_count == 40
    assert quality.attack_ratio == 0.50
    assert "BENIGN" in quality.attack_categories
    assert "DDoS" in quality.attack_categories
    assert "Bot" in quality.attack_categories
    assert quality.feature_coverage_ratio > 0.0


# ── 6. Chronological Split ────────────────────────────────────────────────────
def test_chronological_split(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    split = json.loads((tmp_path / "out" / "split_summary.json").read_text(encoding="utf-8"))
    assert split["train_samples"] == 56  # 70% of 80
    assert split["val_samples"] == 12    # 15% of 80
    assert split["test_samples"] == 12   # 15% of 80
    assert split["train_time_max"] <= split["val_time_min"]
    assert split["val_time_max"] <= split["test_time_min"]


# ── 7. No Temporal Overlap ────────────────────────────────────────────────────
def test_no_temporal_overlap(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    runner.run_benchmark()
    split = json.loads((tmp_path / "out" / "split_summary.json").read_text(encoding="utf-8"))
    assert split["train_time_max"] < split["val_time_max"]
    assert split["val_time_max"] < split["test_time_max"]


# ── 8. Feature Extraction Integration ─────────────────────────────────────────
def test_feature_extraction_integration(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    feat_rep = json.loads((tmp_path / "out" / "feature_report.json").read_text(encoding="utf-8"))
    assert feat_rep["canonical_feature_dim"] == CANONICAL_FEATURE_DIM
    assert feat_rep["feature_coverage_ratio"] > 0.0
    assert feat_rep["packet_features_available"] is False


# ── 9. Logistic Regression Integration ────────────────────────────────────────
def test_logistic_regression_integration(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    lr = res["logistic_baseline"]
    assert "precision" in lr
    assert "recall" in lr
    assert "f1_score" in lr
    assert "false_positive_rate" in lr
    assert "confusion_matrix" in lr
    assert 0.0 <= lr["f1_score"] <= 1.0


# ── 10. World Model Evaluator Integration ──────────────────────────────────────
def test_world_model_evaluator_integration(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    wm = res["world_model_forecasting"]
    assert "results_by_horizon" in wm
    for k_key in ["k_1", "k_3", "k_5"]:
        assert k_key in wm["results_by_horizon"]
        h_data = wm["results_by_horizon"][k_key]
        assert "world_model" in h_data
        assert "mae" in h_data["world_model"]
        assert "mse" in h_data["world_model"]


# ── 11. Baseline Integration (Persistence + Linear) ───────────────────────────
def test_baseline_integration(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    wm = res["world_model_forecasting"]
    for k_key in ["k_1", "k_3", "k_5"]:
        h_data = wm["results_by_horizon"][k_key]
        assert "persistence_baseline" in h_data
        assert "mae" in h_data["persistence_baseline"]
        assert "linear_baseline" in h_data
        assert "mae" in h_data["linear_baseline"]


# ── 12. Unseen Attack Integration ─────────────────────────────────────────────
def test_unseen_attack_integration(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    unseen = res["unseen_attack"]
    assert unseen.get("held_out_category") in ("Bot", "DDoS")
    assert "known_attack_metrics" in unseen
    assert "unseen_attack_metrics" in unseen
    assert "performance_degradation" in unseen
    assert "delta_f1" in unseen["performance_degradation"]


# ── 13. Artifact Isolation ────────────────────────────────────────────────────
def test_artifact_isolation(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    isolated_out = tmp_path / "artifacts" / "real_data_benchmark" / "cic_ids2018" / "exp_01"
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=isolated_out,
        is_synthetic=True,
    )
    runner.run_benchmark()

    expected_files = [
        "provenance.json",
        "data_quality_report.json",
        "split_summary.json",
        "feature_report.json",
        "logistic_baseline_metrics.json",
        "world_model_forecasting_metrics.json",
        "unseen_attack_metrics.json",
        "risk_and_stage_metrics.json",
        "cyberheal_simulation_summary.json",
        "benchmark_manifest.json",
        "BENCHMARK_SUMMARY.md",
    ]
    for fn in expected_files:
        assert (isolated_out / fn).exists(), f"Missing isolated artifact: {fn}"


# ── 14. Manifest Generation ───────────────────────────────────────────────────
def test_manifest_generation(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    manifest_data = json.loads((tmp_path / "out" / "benchmark_manifest.json").read_text(encoding="utf-8"))

    assert manifest_data["dataset_name"] == "CIC-IDS2018"
    assert manifest_data["production_model_weights_untouched"] is True
    assert manifest_data["remediation_mode"] == "DRY_RUN"
    assert manifest_data["execution_type"] == "SYNTHETIC SOFTWARE VALIDATION"


# ── 15. Dataset Provenance ────────────────────────────────────────────────────
def test_dataset_provenance(synthetic_cic_dir: Path) -> None:
    csv_file = synthetic_cic_dir / "02-14-2018.csv"
    h = compute_file_sha256(csv_file)
    assert len(h) == 64
    assert int(h, 16) > 0

    adapter = CicIds2018Adapter()
    _, _, prov = adapter.load_and_normalize([csv_file])
    assert prov.dataset_name == "CIC-IDS2018"
    assert prov.total_bytes == csv_file.stat().st_size
    assert prov.file_hashes_sha256["02-14-2018.csv"] == h


# ── 16. Synthetic Fixture Separation ──────────────────────────────────────────
def test_synthetic_fixture_separation(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    assert res["execution_type"] == "SYNTHETIC SOFTWARE VALIDATION"

    summary_text = (tmp_path / "out" / "BENCHMARK_SUMMARY.md").read_text(encoding="utf-8")
    assert "SYNTHETIC SOFTWARE VALIDATION" in summary_text


# ── 17. Dry-Run CyberHeal Integration ─────────────────────────────────────────
def test_dry_run_cyberheal_integration(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    res = runner.run_benchmark()
    ch = res["cyberheal_dry_run"]
    assert ch["execution_mode"] == "SIMULATION / DRY_RUN"
    assert ch["real_host_remediation_performed"] is False
    assert "handoff_status" in ch


# ── 18. Deterministic Results ─────────────────────────────────────────────────
def test_deterministic_results(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    runner1 = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "run1",
        random_state=42,
        is_synthetic=True,
    )
    res1 = runner1.run_benchmark()

    runner2 = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "run2",
        random_state=42,
        is_synthetic=True,
    )
    res2 = runner2.run_benchmark()

    assert res1["logistic_baseline"]["f1_score"] == res2["logistic_baseline"]["f1_score"]
    assert res1["quality_report"]["total_rows"] == res2["quality_report"]["total_rows"]


# ── 19. Invalid Dataset Rejection ─────────────────────────────────────────────
def test_invalid_dataset_rejection(tmp_path: Path) -> None:
    bad_dir = tmp_path / "bad_data"
    bad_dir.mkdir(parents=True, exist_ok=True)
    corrupt_csv = bad_dir / "bad.csv"
    # CSV missing any label column
    pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}).to_csv(corrupt_csv, index=False)

    adapter = CicIds2018Adapter()
    with pytest.raises(ValueError, match="Could not find a 'Label' column"):
        adapter.load_and_normalize([corrupt_csv])


# ── 20. No Production Model Overwrite ─────────────────────────────────────────
def test_no_production_model_overwrite(synthetic_cic_dir: Path, tmp_path: Path) -> None:
    prod_model_path = Path("artifacts/world_model/best_model.pt")
    assert prod_model_path.exists(), "Production World Model must exist prior to test"

    initial_mtime = prod_model_path.stat().st_mtime
    initial_hash = compute_file_sha256(prod_model_path)

    runner = RealDataBenchmarkRunner(
        dataset_name="CIC-IDS2018",
        data_dir=synthetic_cic_dir,
        output_dir=tmp_path / "out",
        is_synthetic=True,
    )
    runner.run_benchmark()

    final_mtime = prod_model_path.stat().st_mtime
    final_hash = compute_file_sha256(prod_model_path)

    assert initial_mtime == final_mtime, "Production model timestamp was modified!"
    assert initial_hash == final_hash, "Production model weights were overwritten!"
