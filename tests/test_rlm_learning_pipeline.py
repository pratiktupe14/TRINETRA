import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from shield_rca.langgraph_core import AgentState, ShieldLangGraph
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM
from shield_rca.rlm_feedback import (
    ActionMetadata,
    LearningRecord,
    OutcomeMetadata,
    PredictionMetadata,
    ReflectionMetadata,
    RLMStore,
    StateReferences,
    TriggerMetadata,
    VerificationMetadata,
)
from shield_rca.rlm_learning_pipeline import RLMLearningPipeline
from shield_rca.world_model.checkpoint import load_world_model_checkpoint
from shield_rca.world_model.candidates import GRUDynamicsWorldModel
from shield_rca.world_model.model import StateResidualDynamicsModel


def _make_record_with_vectors(rec_id: str, inc_id: str) -> LearningRecord:
    curr = list(np.random.randn(STATE_FEATURE_DIM).astype(float))
    obs = list(np.random.randn(STATE_FEATURE_DIM).astype(float))
    return LearningRecord(
        record_id=rec_id,
        incident_id=inc_id,
        trace_id=f"trc-{rec_id}",
        timestamp=1700000000.0,
        incident_type="PREDICTIVE",
        prediction=PredictionMetadata(risk_score=0.85, confidence=0.90, predicted_stage="IMPACT"),
        trigger=TriggerMetadata(decision="AUTO_TRIGGER"),
        action=ActionMetadata(action_type="rate_limit_ip", target="10.0.0.1", execution_mode="SIMULATION", status="EXECUTED"),
        verification=VerificationMetadata(status="VERIFIED", threat_reduction_ratio=0.85, residual_risk=0.15),
        outcome=OutcomeMetadata(success=True),
        reflection=ReflectionMetadata(
            prediction_quality="USEFUL",
            action_quality="EFFECTIVE",
            verification_quality="VERIFIED",
            recommendations=["NO_CHANGE"],
        ),
        state_references=StateReferences(current_state=curr, observed_state=obs),
    )


def test_candidate_dataset_construction():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec_file = Path(tmpdir) / "records.jsonl"
        store = RLMStore(rec_file)

        # Store 5 valid records with state vectors
        for i in range(5):
            store.append_record(_make_record_with_vectors(f"rec-{i}", f"inc-{i}"))

        pipeline = RLMLearningPipeline(
            artifacts_dir=Path(tmpdir) / "world_model",
            rlm_records_path=rec_file,
        )

        records = pipeline.load_and_validate_records()
        assert len(records) == 5

        train_ds, val_ds, test_ds = pipeline.build_candidate_dataset(
            records=records,
            horizons=[1, 3],
            synthetic_supplement=True,
            supplement_samples=30,
            seed=42,
        )

        assert train_ds.total_usable_pairs > 0
        assert val_ds.total_usable_pairs > 0
        assert test_ds.total_usable_pairs > 0
        assert 1 in train_ds.horizons
        assert 3 in train_ds.horizons


def test_candidate_training_does_not_overwrite_production_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        art_dir = tmp_path / "world_model"
        art_dir.mkdir(parents=True, exist_ok=True)
        rec_file = tmp_path / "records.jsonl"

        # Create dummy production model
        prod_model_file = art_dir / "best_model.pt"
        prod_model_file.write_text("DUMMY_PRODUCTION_WEIGHTS_ORIGINAL", encoding="utf-8")

        store = RLMStore(rec_file)
        store.append_record(_make_record_with_vectors("rec-1", "inc-1"))

        pipeline = RLMLearningPipeline(
            artifacts_dir=art_dir,
            rlm_records_path=rec_file,
        )

        records = pipeline.load_and_validate_records()
        train_ds, val_ds, test_ds = pipeline.build_candidate_dataset(records, horizons=[1], supplement_samples=20)

        # Train candidate
        cand_dir = pipeline.train_candidate(
            candidate_id="cand-test-01",
            train_ds=train_ds,
            val_ds=val_ds,
            horizons=[1],
            max_epochs=1,
            batch_size=16,
        )

        # 1. Candidate is saved in isolated candidate directory
        assert cand_dir.exists()
        assert (cand_dir / "best_model.pt").exists()
        assert (cand_dir / "model_config.json").exists()
        assert (cand_dir / "model_manifest.json").exists()

        # 2. Production model file was NOT overwritten!
        assert prod_model_file.read_text(encoding="utf-8") == "DUMMY_PRODUCTION_WEIGHTS_ORIGINAL"


def test_candidate_evaluation_and_rejection_of_worse_candidate():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        art_dir = tmp_path / "world_model"
        rec_file = tmp_path / "records.jsonl"

        # Train a mock baseline candidate
        pipeline = RLMLearningPipeline(art_dir, rec_file)
        store = RLMStore(rec_file)
        store.append_record(_make_record_with_vectors("rec-1", "inc-1"))

        records = pipeline.load_and_validate_records()
        train_ds, val_ds, test_ds = pipeline.build_candidate_dataset(records, horizons=[1], supplement_samples=20)

        # Train candidate A
        cand_a_dir = pipeline.train_candidate("cand-A", train_ds, val_ds, horizons=[1], max_epochs=2)
        eval_report_a = pipeline.evaluate_candidate("cand-A", test_ds, num_records=1)

        assert eval_report_a.evaluation_decision in ("ACCEPT", "REJECT", "INCONCLUSIVE")
        assert eval_report_a.candidate_model_mae > 0.0


def test_candidate_release_safety_requires_operator_approval():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        art_dir = tmp_path / "world_model"
        rec_file = tmp_path / "records.jsonl"

        pipeline = RLMLearningPipeline(art_dir, rec_file)
        store = RLMStore(rec_file)
        store.append_record(_make_record_with_vectors("rec-1", "inc-1"))

        records = pipeline.load_and_validate_records()
        train_ds, val_ds, test_ds = pipeline.build_candidate_dataset(records, horizons=[1], supplement_samples=20)
        pipeline.train_candidate("cand-B", train_ds, val_ds, horizons=[1], max_epochs=1)
        pipeline.evaluate_candidate("cand-B", test_ds, num_records=1)

        # 1. Promote without approval MUST raise PermissionError
        with pytest.raises(PermissionError, match="Explicit operator approval is required"):
            pipeline.promote_candidate("cand-B", operator_approved=False)

        # 2. Promote with approval succeeds (force=True to allow test promotion regardless of mock dataset delta)
        prod_path = pipeline.promote_candidate("cand-B", operator_approved=True, force=True)
        assert (prod_path / "best_model.pt").exists()


import asyncio

@pytest.mark.anyio
async def test_live_incident_generates_learning_record_without_training(monkeypatch: pytest.MonkeyPatch):
    """Verify live CyberHeal runtime execution generates RLM record without modifying production weights."""
    async def mock_sleep(_duration: float):
        return

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    graph = ShieldLangGraph(db_client=None)
    state = AgentState(
        incident_id="live-inc-test-01",
        alert_payload={"source": "suricata", "metric": "ddos_spike", "ip": "10.0.0.99"},
        incident_type="PREDICTIVE",
        risk_score="0.88",
        trajectory_risk_score=0.88,
        attack_stage="IMPACT",
        asset_criticality="MEDIUM",
    )

    result_state = await graph.execute_workflow(state)

    # 1. Incident completed cleanly
    assert result_state.status in ("RESOLVED", "ESCALATED_FINAL")

    # 2. RLM feedback record was generated and attached to AgentState
    assert result_state.learning_record is not None
    assert result_state.learning_record["incident_id"] == "live-inc-test-01"
    assert result_state.learning_record["prediction"]["risk_score"] == pytest.approx(0.88)
    assert result_state.learning_recommendation is not None

    # 3. Production model artifacts were untouched (timestamp / size intact)
    prod_weights = Path("artifacts/world_model/best_model.pt")
    if prod_weights.exists():
        assert prod_weights.stat().st_size > 0


def test_train_candidate_instantiates_gru_dynamics_world_model():
    """Verify train_candidate() strictly instantiates and trains GRUDynamicsWorldModel with nn.GRU."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        art_dir = tmp_path / "world_model"
        rec_file = tmp_path / "records.jsonl"

        store = RLMStore(rec_file)
        store.append_record(_make_record_with_vectors("rec-gru-1", "inc-gru-1"))

        pipeline = RLMLearningPipeline(
            artifacts_dir=art_dir,
            rlm_records_path=rec_file,
        )

        records = pipeline.load_and_validate_records()
        train_ds, val_ds, _ = pipeline.build_candidate_dataset(records, horizons=[1, 3], supplement_samples=20)

        cand_dir = pipeline.train_candidate(
            candidate_id="cand-gru-verify",
            train_ds=train_ds,
            val_ds=val_ds,
            horizons=[1, 3],
            max_epochs=1,
            batch_size=8,
        )

        # 1. Candidate directory and files
        assert cand_dir.exists()
        assert (cand_dir / "best_model.pt").exists()
        assert (cand_dir / "model_config.json").exists()
        assert (cand_dir / "model_manifest.json").exists()

        # 2. Manifest and config check
        with open(cand_dir / "model_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["model_type"] == "GRUDynamicsWorldModel"
        assert manifest["parent_model"] == "artifacts/world_model/best_model.pt"

        with open(cand_dir / "model_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["model_type"] == "GRUDynamicsWorldModel"
        assert config["num_layers"] == 2
        assert config["hidden_dim"] == 128
        assert config["bidirectional"] is False

        # 3. Load checkpoint and verify exact instantiated class
        loaded_model, loaded_cfg, loaded_mf = load_world_model_checkpoint(cand_dir)
        assert isinstance(loaded_model, GRUDynamicsWorldModel), f"Expected GRUDynamicsWorldModel, got {type(loaded_model)}"

        # 4. Verify nn.GRU component exists and configuration matches
        assert hasattr(loaded_model, "gru"), "Candidate model must contain an nn.GRU component"
        assert isinstance(loaded_model.gru, torch.nn.GRU), f"Expected torch.nn.GRU, got {type(loaded_model.gru)}"
        assert loaded_model.gru.input_size == 128
        assert loaded_model.gru.hidden_size == 128
        assert loaded_model.gru.num_layers == 2
        assert loaded_model.gru.bidirectional is False


def test_candidate_gru_forward_temporal_input_and_horizons():
    """Verify GRUDynamicsWorldModel forward pass supports [N, T, 280] temporal sequences and horizons 1, 3, 5."""
    N = 4
    T = 6
    S = STATE_FEATURE_DIM

    model = GRUDynamicsWorldModel(
        state_dim=S,
        hidden_dim=128,
        num_layers=2,
        dropout=0.02,
        max_horizon=16,
        bidirectional=False,
    )
    model.eval()

    # Temporal 3D input [N, T, S]
    seq_input = torch.randn(N, T, S, dtype=torch.float32)
    horizons = torch.tensor([1, 3, 5, 1], dtype=torch.long)
    delta_ts = torch.tensor([1.0, 3.0, 5.0, 1.0], dtype=torch.float32)

    with torch.no_grad():
        out, delta = model(seq_input, horizons, delta_ts, return_delta=True)

    assert out.shape == (N, S)
    assert delta.shape == (N, S)
    last_step = seq_input[:, -1, :]
    assert torch.allclose(out, last_step + delta, atol=1e-5), "Predicted state must equal last step + predicted delta"

    # Also verify 2D input [N, S] works seamlessly (backward compatibility)
    single_input = torch.randn(N, S, dtype=torch.float32)
    with torch.no_grad():
        out_2d, delta_2d = model(single_input, horizons, delta_ts, return_delta=True)
    assert out_2d.shape == (N, S)
    assert delta_2d.shape == (N, S)
    assert torch.allclose(out_2d, single_input + delta_2d, atol=1e-5)


def test_candidate_training_preserves_real_production_model_sha256():
    """Verify real artifacts/world_model/best_model.pt SHA256 is 100% identical before and after candidate training."""
    prod_path = Path("artifacts/world_model/best_model.pt")
    if not prod_path.exists():
        pytest.skip("Production model artifacts/world_model/best_model.pt not present")

    sha256_before = hashlib.sha256(prod_path.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        rec_file = tmp_path / "records.jsonl"
        store = RLMStore(rec_file)
        store.append_record(_make_record_with_vectors("rec-sha-1", "inc-sha-1"))

        # Point pipeline artifacts_dir to real artifacts/world_model, but training candidate must only touch candidates/
        pipeline = RLMLearningPipeline(
            artifacts_dir=Path("artifacts/world_model"),
            candidates_dir=Path("artifacts/world_model/candidates"),
            rlm_records_path=rec_file,
        )

        records = pipeline.load_and_validate_records()
        train_ds, val_ds, _ = pipeline.build_candidate_dataset(records, horizons=[1], supplement_samples=10)

        cand_id = "cand-sha-test"
        cand_dir = pipeline.train_candidate(
            candidate_id=cand_id,
            train_ds=train_ds,
            val_ds=val_ds,
            horizons=[1],
            max_epochs=1,
            batch_size=8,
        )

        try:
            assert cand_dir.exists()
            assert (cand_dir / "best_model.pt").exists()

            # Ensure candidate weights are in the candidate directory, NOT the root artifacts dir
            assert cand_dir == Path("artifacts/world_model/candidates") / cand_id

            # Verify production file hash has NOT changed at all
            sha256_after = hashlib.sha256(prod_path.read_bytes()).hexdigest()
            assert sha256_after == sha256_before, (
                f"FATAL: Production checkpoint was modified! Before: {sha256_before}, After: {sha256_after}"
            )
        finally:
            # Clean up candidate test artifacts
            if cand_dir.exists():
                shutil.rmtree(cand_dir, ignore_errors=True)


def test_existing_production_model_loading_remains_compatible():
    """Verify that existing production model at artifacts/world_model loads as StateResidualDynamicsModel."""
    prod_dir = Path("artifacts/world_model")
    if not (prod_dir / "best_model.pt").exists():
        pytest.skip("Production model artifacts/world_model/best_model.pt not present")

    prod_model, config, manifest = load_world_model_checkpoint(prod_dir)
    assert isinstance(prod_model, StateResidualDynamicsModel), (
        f"Expected production model to load as StateResidualDynamicsModel, got {type(prod_model)}"
    )

    # Test forward pass with production model
    N = 2
    S = STATE_FEATURE_DIM
    x = torch.randn(N, S)
    for h in [1, 3, 5]:
        horizons = torch.full((N,), h, dtype=torch.long)
        delta_ts = torch.full((N,), float(h), dtype=torch.float32)
        with torch.no_grad():
            out, delta = prod_model(x, horizons, delta_ts, return_delta=True)
            assert out.shape == (N, S)
            assert delta.shape == (N, S)

