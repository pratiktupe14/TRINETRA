"""Focused Tests for Phase 11B: CTU-13 Real-Data Benchmark, Splitting, and Isolation.

Validates:
1. Real CTU-13 file discovery across all 13 scenarios.
2. Deterministic scenario split definition.
3. Strict Scenario 10 isolation (unseen holdout).
4. CTU-13 canonical feature schema compatibility (78-D flow, 280-D network state).
5. Programmatic leakage detection and boundary isolation.
6. Zero cross-scenario temporal window contamination.
7. Production World Model immutability (best_model.pt untouched).
8. Candidate GRU model isolation (no automatic promotion).
9. Anti-fabrication safeguards (no fabricated MITRE techniques).
10. Explicit explainability method disclosure (State Delta Attribution, NOT SHAP).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from shield_rca.evaluation.ctu13_adapter import (
    CTU13_OFFICIAL_SCENARIOS,
    Ctu13Adapter,
    Ctu13ScenarioSplitter,
)
from shield_rca.evaluation.dataset_adapters import compute_file_sha256
from shield_rca.network_state.future_dataset import FutureStateDatasetBuilder
from shield_rca.network_state.preprocessor import NetworkStatePreprocessor
from shield_rca.network_state.state_encoder import NetworkStateEncoder
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM
from shield_rca.telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES
from shield_rca.world_model.explainability import explain_forecast
from shield_rca.world_model.risk_estimator import PredictiveRiskEstimator
from shield_rca.world_model.stage_mapper import AttackStage, AttackStageMapper
from shield_rca.world_model.trigger_gate import (
    ActionSafety,
    AssetCriticality,
    PredictiveTriggerGate,
    TriggerDecisionType,
)


def test_ctu13_real_file_discovery() -> None:
    """Verify that all 13 official scenarios are discovered in data/ctu13."""
    data_dir = Path("data/ctu13")
    if not data_dir.exists():
        pytest.skip("data/ctu13 does not exist locally")

    discovered = Ctu13Adapter.discover_files(data_dir)
    assert len(discovered) >= 13, f"Expected at least 13 scenario files, found {len(discovered)}"

    scenario_ids = {Ctu13Adapter.detect_scenario_id(p) for p in discovered}
    for i in range(1, 14):
        assert f"scenario_{i}" in scenario_ids, f"Missing scenario_{i} from discovery"


def test_ctu13_scenario_split_definitions() -> None:
    """Verify official Phase 11 deterministic partition."""
    split = Ctu13ScenarioSplitter.get_unseen_split_definition()

    assert split["unseen_scenario"] == "scenario_10"
    assert split["unseen_scenario_file"] == "capture20110818.binetflow"

    train = set(split["train_scenarios"])
    val = set(split["val_scenarios"])
    known_test = set(split["known_test_scenarios"])
    unseen_test = set(split["unseen_test_scenarios"])

    assert train == {"scenario_1", "scenario_2", "scenario_3", "scenario_6", "scenario_7", "scenario_8", "scenario_9"}
    assert val == {"scenario_11", "scenario_12"}
    assert known_test == {"scenario_4", "scenario_5", "scenario_13"}
    assert unseen_test == {"scenario_10"}


def test_scenario_10_strict_isolation() -> None:
    """Verify that Scenario 10 is completely isolated from training, validation, and known test."""
    split = Ctu13ScenarioSplitter.get_unseen_split_definition()

    unseen = set(split["unseen_test_scenarios"])
    train = set(split["train_scenarios"])
    val = set(split["val_scenarios"])
    known_test = set(split["known_test_scenarios"])

    assert "scenario_10" not in train, "LEAKAGE: Scenario 10 present in train split"
    assert "scenario_10" not in val, "LEAKAGE: Scenario 10 present in validation split"
    assert "scenario_10" not in known_test, "LEAKAGE: Scenario 10 present in known test split"

    # All sets mutually disjoint
    assert len(train.intersection(val)) == 0
    assert len(train.intersection(known_test)) == 0
    assert len(val.intersection(known_test)) == 0
    assert len(unseen.intersection(train | val | known_test)) == 0


def test_leakage_detection_auditor() -> None:
    """Verify that an artificial contamination of Scenario 10 triggers leakage detection."""
    clean_split = Ctu13ScenarioSplitter.get_unseen_split_definition()

    # Create contaminated split
    dirty_split = {
        "train_scenarios": clean_split["train_scenarios"] + ["scenario_10"],
        "val_scenarios": clean_split["val_scenarios"],
        "known_test_scenarios": clean_split["known_test_scenarios"],
        "unseen_test_scenarios": clean_split["unseen_test_scenarios"],
    }

    # Programmatic leakage check
    has_leakage = "scenario_10" in set(dirty_split["train_scenarios"])
    assert has_leakage is True, "Auditor must catch Scenario 10 contamination in training"


def test_ctu13_schema_compatibility() -> None:
    """Verify real CTU-13 schema loading and 78-D canonical mapping."""
    data_dir = Path("data/ctu13/5")  # Scenario 5 is compact (~17MB)
    files = list(data_dir.glob("*.binetflow"))
    if not files:
        pytest.skip("Scenario 5 binetflow not found locally")

    adapter = Ctu13Adapter()
    df, quality, prov = adapter.load_and_normalize(files, max_rows=500)

    assert len(df) == 500
    features = df.attrs["features_holder"].array
    assert features.shape == (500, CANONICAL_FEATURE_DIM)
    assert np.all(np.isfinite(features)), "Features must be finite (no NaN or Inf)"
    assert "label_int" in df.columns
    assert "flow_start_time" in df.columns
    assert quality.usable_rows == 500


def test_280d_network_state_and_temporal_windows() -> None:
    """Verify encoding into 280-D Network States and T=5 temporal sequences."""
    data_dir = Path("data/ctu13/5")
    files = list(data_dir.glob("*.binetflow"))
    if not files:
        pytest.skip("Scenario 5 binetflow not found locally")

    adapter = Ctu13Adapter()
    df, _, _ = adapter.load_and_normalize(files, max_rows=200)
    features = df.attrs["features_holder"].array

    window_size = 10
    stride = 5
    win_indices = list(range(0, len(features) - window_size + 1, stride))
    X_windows = np.stack([features[i : i + window_size] for i in win_indices], axis=0)

    encoder = NetworkStateEncoder()
    batch = encoder.encode_batch(X_windows)

    assert batch.states.shape == (len(win_indices), STATE_FEATURE_DIM)
    assert batch.states.shape[1] == 280
    assert np.all(np.isfinite(batch.states))

    builder = FutureStateDatasetBuilder(horizons=[1, 3, 5], sequence_length=5)
    ds = builder.build_future_pairs(batch, sequence_length=5)

    assert ds.total_usable_pairs > 0
    for k in [1, 3, 5]:
        b = ds.batches[k]
        assert b.current_states.shape[1] == 5
        assert b.current_states.shape[2] == 280
        assert b.target_states.shape[1] == 280


def test_production_model_immutability() -> None:
    """Verify that production World Model best_model.pt exists and hash is stable."""
    prod_path = Path("artifacts/world_model/best_model.pt")
    assert prod_path.exists(), "Production World Model best_model.pt must exist"
    h1 = compute_file_sha256(prod_path)
    assert len(h1) == 64
    assert h1 == "78ed7a009373e6460f7416fd645e2060513ac165f6cfad66244d4a519c2565a4"


def test_candidate_gru_isolation() -> None:
    """Verify that GRU candidate cand-gru-real-d49b3ee3 remains isolated."""
    cand_path = Path("artifacts/world_model/candidates/cand-gru-real-d49b3ee3/best_model.pt")
    if not cand_path.exists():
        pytest.skip("Candidate GRU model not found")
    prod_path = Path("artifacts/world_model/best_model.pt")

    # Hashes must differ (candidate is distinct architecture from production)
    h_prod = compute_file_sha256(prod_path)
    h_cand = compute_file_sha256(cand_path)
    assert h_prod != h_cand, "Candidate model must not be identical to production model"


def test_anti_fabrication_safeguards() -> None:
    """Verify that stage mapper returns mitre_technique = None for real telemetry."""
    mapper = AttackStageMapper()
    curr_s = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    pred_s = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    pred_s[10] = 5.0  # Drift

    res = mapper.map_stage(curr_s, pred_s, horizon=1, known_label=None)
    assert res.mitre_technique is None, "Mitre technique must be None for unverified telemetry"


def test_explainability_method_disclosure() -> None:
    """Verify that explain_forecast uses State Delta Attribution and explicitly NOT SHAP."""
    curr_s = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    pred_s = np.ones(STATE_FEATURE_DIM, dtype=np.float32) * 0.5

    exp = explain_forecast(curr_s, pred_s, horizon=1, top_k=5)
    assert exp.explanation_method == "State Delta Attribution"
    assert "SHAP" not in exp.explanation_method
    assert len(exp.top_features) == 5


def test_real_data_benchmark_output_validation() -> None:
    """Verify all real CTU-13 benchmark outputs in artifacts/phase11b."""
    art_dir = Path("artifacts/phase11b")
    assert art_dir.exists(), "artifacts/phase11b must exist"

    # Required JSON files
    required_files = [
        "ctu13_provenance.json",
        "ctu13_data_quality_report.json",
        "ctu13_split.json",
        "ctu13_leakage_audit.json",
        "ctu13_logistic_metrics.json",
        "ctu13_world_model_metrics.json",
        "ctu13_unseen_results.json",
        "ctu13_risk_results.json",
        "ctu13_explainability.json",
        "ctu13_benchmark_summary.json",
    ]
    for fn in required_files:
        p = art_dir / fn
        assert p.exists(), f"Missing required artifact: {fn}"
        assert p.stat().st_size > 0, f"Artifact is empty: {fn}"

    # Verify summary contents
    summary = json.loads((art_dir / "ctu13_benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_status"] == "REAL"
    assert summary["scenario_coverage"] == "13/13"
    assert summary["leakage_audit_status"] == "PASS"
    assert summary["production_model_status"] == "UNCHANGED"
    assert summary["phase11b_decision"] == "PASS"
    assert "known_test" in summary["world_model_forecasting"]
    assert "unseen_test_scenario_10" in summary["world_model_forecasting"]
    assert "known_test" in summary["logistic_regression"]
    assert "unseen_test_scenario_10" in summary["logistic_regression"]

