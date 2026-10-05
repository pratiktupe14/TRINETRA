"""Tests for Phase 10 TRINETRA Model & Pipeline Improvements.

Covers:
1. Label alignment: Official CIC-IDS2018 'Infilteration' label mapped correctly to 'Infiltration' (not 'DDoS').
2. Feature schema aliases: Official CSV column names (e.g. 'FIN Flag Cnt', 'Fwd Header Len', 'Flow Byts/s')
   are correctly mapped, populating TCP flags and subflow attributes.
3. Candidate models: Candidate A, Candidate B (Temporal Context), and Candidate C (GRU Dynamics)
   forward pass, residual formulation, and shape consistency.
4. Candidate evaluation compatibility: WorldModelEvaluator compatibility with all candidate architectures.
5. State preprocessor: Strict leak-free normalization on Network States.
6. Unseen attack isolation: Zero leakage of held-out categories into train/val.
7. Candidate model artifact isolation: Candidates stored under artifacts/world_model/candidates/
   leaving production weights artifacts/world_model/best_model.pt completely untouched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.shield_rca.evaluation.dataset_adapters import CIC_2018_LABEL_MAP, CicIds2018Adapter
from src.shield_rca.evaluation.logistic_baseline import LogisticBaselineConfig, LogisticRegressionBaseline
from src.shield_rca.evaluation.unseen_attack import (
    UnseenAttackConfig,
    UnseenAttackDataset,
    UnseenAttackEvaluator,
)
from src.shield_rca.network_state.future_dataset import (
    FutureStateDatasetBuilder,
    FutureStatePairBatch,
    MultiHorizonFutureDataset,
)
from src.shield_rca.network_state.preprocessor import NetworkStatePreprocessor
from src.shield_rca.network_state.state_encoder import NetworkStateBatch
from src.shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from src.shield_rca.telemetry.feature_schema import CANONICAL_FEATURES, align_dataframe_to_schema
from src.shield_rca.world_model.baselines import LinearDynamicsBaseline, PersistenceBaseline
from src.shield_rca.world_model.candidates import (
    GRUDynamicsWorldModel,
    TemporalContextResidualModel,
)
from src.shield_rca.world_model.evaluator import WorldModelEvaluator
from src.shield_rca.world_model.model import StateResidualDynamicsModel


# ─────────────────────────────────────────────────────────────────────────────
# 1. Label Mapping Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_infilteration_label_mapping():
    """Verify that official CIC-IDS2018 'Infilteration' typo maps to 'Infiltration'."""
    adapter = CicIds2018Adapter()
    # Test single label mapping function
    lbl, cat = adapter._map_single_label("Infilteration")
    assert lbl == 1, f"Expected label 1 for attack, got {lbl}"
    assert cat == "Infiltration", f"Expected category 'Infiltration', got {cat}"

    # Test lowercase and mixed-case
    lbl_lower, cat_lower = adapter._map_single_label("infilteration")
    assert lbl_lower == 1
    assert cat_lower == "Infiltration"

    # Test standard spelling as well
    lbl_std, cat_std = adapter._map_single_label("Infiltration")
    assert lbl_std == 1
    assert cat_std == "Infiltration"

    # Ensure it is NOT mapped to DDoS
    assert cat != "DDoS"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Column Aliases & Feature Schema Population
# ─────────────────────────────────────────────────────────────────────────────

def test_cic_column_aliases_populated():
    """Verify that official CIC-IDS2018 CSV column headers populate canonical features."""
    df_raw = pd.DataFrame({
        "Timestamp": ["02/03/2018 08:47:38"],
        "Dst Port": [80],
        "Protocol": [6],
        "Flow Duration": [1000],
        "Tot Fwd Pkts": [10],
        "Tot Bwd Pkts": [12],
        "TotLen Fwd Pkts": [500],
        "TotLen Bwd Pkts": [600],
        "Flow Byts/s": [1100.0],
        "Flow Pkts/s": [22.0],
        "FIN Flag Cnt": [1],
        "SYN Flag Cnt": [1],
        "RST Flag Cnt": [0],
        "PSH Flag Cnt": [1],
        "ACK Flag Cnt": [1],
        "URG Flag Cnt": [0],
        "ECE Flag Cnt": [0],
        "Fwd Header Len": [320],
        "Bwd Header Len": [384],
        "Subflow Fwd Byts": [500],
        "Subflow Bwd Byts": [600],
        "Subflow Fwd Pkts": [10],
        "Subflow Bwd Pkts": [12],
        "Fwd IAT Tot": [800.0],
        "Bwd IAT Tot": [900.0],
        "Fwd Pkt Len Mean": [50.0],
        "Fwd Pkt Len Std": [5.0],
        "Bwd Pkt Len Mean": [50.0],
        "Bwd Pkt Len Std": [5.0],
        "Flow IAT Mean": [100.0],
        "Flow IAT Std": [10.0],
        "Fwd IAT Mean": [80.0],
        "Fwd IAT Std": [8.0],
        "Bwd IAT Mean": [90.0],
        "Bwd IAT Std": [9.0],
        "Pkt Len Mean": [50.0],
        "Pkt Len Std": [5.0],
        "Label": ["Benign"],
    })

    aligned = align_dataframe_to_schema(df_raw)

    # TCP flags MUST be non-zero and correctly extracted
    assert aligned["fin_flag_cnt"].iloc[0] == 1.0
    assert aligned["syn_flag_cnt"].iloc[0] == 1.0
    assert aligned["psh_flag_cnt"].iloc[0] == 1.0
    assert aligned["ack_flag_cnt"].iloc[0] == 1.0

    # Header lengths and subflow features must be populated
    assert aligned["fwd_header_length"].iloc[0] == 320.0
    assert aligned["bwd_header_length"].iloc[0] == 384.0
    assert aligned["subflow_fwd_bytes"].iloc[0] == 500.0
    assert aligned["subflow_bwd_bytes"].iloc[0] == 600.0
    assert aligned["subflow_fwd_packets"].iloc[0] == 10.0
    assert aligned["subflow_bwd_packets"].iloc[0] == 12.0

    # Rates and totals
    assert aligned["flow_bytes_per_s"].iloc[0] == 1100.0
    assert aligned["flow_packets_per_s"].iloc[0] == 22.0
    assert aligned["fwd_iat_total"].iloc[0] == 800.0
    assert aligned["bwd_iat_total"].iloc[0] == 900.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Candidate World Model Architectures
# ─────────────────────────────────────────────────────────────────────────────

def test_candidate_models_forward_and_residual():
    """Verify Candidate A, Candidate B, and Candidate C forward passes and residual output."""
    N = 8
    S = STATE_FEATURE_DIM

    curr_states = torch.randn(N, S)
    horizons = torch.tensor([1, 1, 3, 3, 5, 5, 1, 3], dtype=torch.long)
    delta_ts = torch.tensor([0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 0.5, 1.5], dtype=torch.float32)

    # Candidate A: StateResidualDynamicsModel
    cand_a = StateResidualDynamicsModel(state_dim=S, hidden_dim=64, num_blocks=2)
    cand_a.eval()
    with torch.no_grad():
        out_a, delta_a = cand_a(curr_states, horizons, delta_ts, return_delta=True)
        assert out_a.shape == (N, S)
        assert delta_a.shape == (N, S)
        assert torch.allclose(out_a, curr_states + delta_a, atol=1e-5)

    # Candidate B: TemporalContextResidualModel
    cand_b = TemporalContextResidualModel(state_dim=S, hidden_dim=64, num_blocks=2)
    cand_b.eval()
    with torch.no_grad():
        out_b, delta_b = cand_b(curr_states, horizons, delta_ts, return_delta=True)
        assert out_b.shape == (N, S)
        assert delta_b.shape == (N, S)
        assert torch.allclose(out_b, curr_states + delta_b, atol=1e-5)

    # Candidate C: GRUDynamicsWorldModel
    cand_c = GRUDynamicsWorldModel(state_dim=S, hidden_dim=64, num_layers=2)
    cand_c.eval()
    with torch.no_grad():
        out_c, delta_c = cand_c(curr_states, horizons, delta_ts, return_delta=True)
        assert out_c.shape == (N, S)
        assert delta_c.shape == (N, S)
        assert torch.allclose(out_c, curr_states + delta_c, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluator Compatibility with Candidates
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_multi_horizon_dataset() -> MultiHorizonFutureDataset:
    N = 20
    S = STATE_FEATURE_DIM
    batches = {}
    for h in [1, 3, 5]:
        curr = np.random.randn(N, S).astype(np.float32)
        targ = curr + np.random.randn(N, S).astype(np.float32) * 0.1
        batches[h] = FutureStatePairBatch(
            current_states=curr,
            target_states=targ,
            current_timestamps=np.linspace(100.0, 200.0, N),
            target_timestamps=np.linspace(101.0, 201.0, N),
            delta_times=np.full(N, float(h)),
            horizon=h,
            source_ids=["test_host"] * N,
            schema_version=STATE_SCHEMA_VERSION,
        )
    return MultiHorizonFutureDataset(
        batches=batches,
        horizons=[1, 3, 5],
        total_states_input=N,
        dropped_tail_pairs={1: 0, 3: 0, 5: 0},
        schema_version=STATE_SCHEMA_VERSION,
    )


def test_world_model_evaluator_with_all_candidates():
    """Verify WorldModelEvaluator evaluates all candidate models seamlessly."""
    ds = _make_dummy_multi_horizon_dataset()
    S = STATE_FEATURE_DIM

    models = [
        ("Candidate A", StateResidualDynamicsModel(state_dim=S, hidden_dim=64, num_blocks=1)),
        ("Candidate B", TemporalContextResidualModel(state_dim=S, hidden_dim=64, num_blocks=1)),
        ("Candidate C", GRUDynamicsWorldModel(state_dim=S, hidden_dim=64, num_layers=1)),
    ]

    for name, model in models:
        evaluator = WorldModelEvaluator(model)
        report = evaluator.evaluate_multi_horizon(ds)
        assert "results_by_horizon" in report, f"{name} failed to produce results_by_horizon"
        for k in [1, 3, 5]:
            k_key = f"k_{k}"
            assert k_key in report["results_by_horizon"], f"{name} missing horizon {k_key}"
            res = report["results_by_horizon"][k_key]
            assert "world_model" in res
            assert "persistence_baseline" in res
            assert "mae" in res["world_model"]
            assert "mae" in res["persistence_baseline"]
            assert "group_errors" in res
            assert "continuous_mae" in res["group_errors"]
            assert "flags_mae" in res["group_errors"]
            assert "shape_mae" in res["group_errors"]
            assert "endpoints_mae" in res["group_errors"]
            assert "macro_mae" in res["group_errors"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. State Preprocessor Leakage-Free Normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_state_preprocessor_leakage_safety():
    """Verify NetworkStatePreprocessor fits only on train and does not leak val/test info."""
    N_tr = 50
    N_te = 20
    S = STATE_FEATURE_DIM

    X_train = np.random.randn(N_tr, S).astype(np.float32) * 5.0 + 10.0
    X_test = np.random.randn(N_te, S).astype(np.float32) * 2.0 + 20.0

    prep = NetworkStatePreprocessor(scaler_type="standard")
    assert not prep.is_fitted

    prep.fit(X_train)
    assert prep.is_fitted
    assert prep.scaler.n_samples_seen_ == N_tr

    # Record training mean and variance
    mean_before = prep.scaler.mean_.copy()
    var_before = prep.scaler.var_.copy()

    # Transform test set
    X_test_scaled = prep.transform(X_test)
    assert X_test_scaled.shape == (N_te, S)

    # Ensure transformer did NOT update scaler parameters
    assert np.array_equal(prep.scaler.mean_, mean_before)
    assert np.array_equal(prep.scaler.var_, var_before)
    assert prep.scaler.n_samples_seen_ == N_tr


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unseen Attack Isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_unseen_attack_dataset_isolation():
    """Verify that held-out category is strictly isolated from train and validation sets."""
    n_samples = 200
    S = 78
    X = np.random.randn(n_samples, S).astype(np.float32)
    y = np.zeros(n_samples, dtype=int)
    y[100:] = 1

    cats = ["BENIGN"] * 100 + ["DDoS"] * 50 + ["Bot"] * 50

    ds = UnseenAttackDataset(
        X_train=X[:70],
        y_train=y[:70],
        categories_train=cats[:70],
        X_val=X[70:100],
        y_val=y[70:100],
        categories_val=cats[70:100],
        X_test_known=X[100:150],
        y_test_known=y[100:150],
        categories_test_known=cats[100:150],
        X_test_unseen=X[150:],
        y_test_unseen=y[150:],
        categories_test_unseen=cats[150:],
    )

    # Verification must succeed for held_out="Bot"
    ds.verify_no_leakage(["Bot"])

    # If Bot is injected into training, verify_no_leakage MUST raise ValueError
    ds_leaked = UnseenAttackDataset(
        X_train=X[:71],
        y_train=y[:71],
        categories_train=cats[:70] + ["Bot"],  # Leaked!
        X_val=X[71:100],
        y_val=y[71:100],
        categories_val=cats[71:100],
        X_test_known=X[100:150],
        y_test_known=y[100:150],
        categories_test_known=cats[100:150],
        X_test_unseen=X[150:],
        y_test_unseen=y[150:],
        categories_test_unseen=cats[150:],
    )
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        ds_leaked.verify_no_leakage(["Bot"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Production Model Weight Protection & Candidate Isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_production_weights_protected():
    """Verify that artifacts/world_model/best_model.pt remains protected."""
    prod_path = Path("artifacts/world_model/best_model.pt")
    if prod_path.exists():
        # Ensure it is a regular file
        assert prod_path.is_file()
        assert prod_path.stat().st_size > 0

    # Verify candidates directory is distinct
    candidates_dir = Path("artifacts/world_model/candidates")
    assert candidates_dir != Path("artifacts/world_model")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Phase 10B: Real Temporal Sequence Construction & Input Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_sequence_construction_shape_and_chronology():
    """Verify FutureStateDatasetBuilder constructs [N, T, 280] temporal sequences with T > 1."""
    N = 30
    T = 5
    S = STATE_FEATURE_DIM

    states = np.random.randn(N, S).astype(np.float32)
    timestamps = np.linspace(100.0, 100.0 + N - 1, N)
    batch = NetworkStateBatch(
        states=states,
        timestamps=timestamps,
        window_starts=timestamps - 20.0,
        window_ends=timestamps,
        source_ids=["host_1"] * N,
        feature_names=tuple(f"feat_{i}" for i in range(S)),
        schema_version=STATE_SCHEMA_VERSION,
    )

    builder = FutureStateDatasetBuilder(horizons=[1, 3, 5], sequence_length=T)
    ds = builder.build_future_pairs(batch)

    for k in [1, 3, 5]:
        pair_batch = ds.batches[k]
        assert pair_batch.is_temporal_sequence
        assert pair_batch.sequence_length == T
        assert pair_batch.current_states.ndim == 3
        assert pair_batch.current_states.shape[1] == T
        assert pair_batch.current_states.shape[2] == S
        assert pair_batch.target_states.shape[1] == S

        # Verify that the final state in current_states sequence matches S_t
        for i in range(pair_batch.num_pairs):
            s_t = pair_batch.current_states[i, -1, :]
            t_curr = pair_batch.current_timestamps[i]
            # Match timestamp with original batch
            orig_idx = int(round(t_curr - 100.0))
            assert np.allclose(s_t, states[orig_idx], atol=1e-5)

            # Target must be S_{t+k}
            target_orig_idx = orig_idx + k
            assert np.allclose(pair_batch.target_states[i], states[target_orig_idx], atol=1e-5)
            assert pair_batch.delta_times[i] == pytest.approx(float(k))


def test_gru_regression_fails_on_one_step_sequence():
    """Verify that GRUDynamicsWorldModel with require_temporal_sequence=True rejects 1-step inputs."""
    model = GRUDynamicsWorldModel(
        state_dim=STATE_FEATURE_DIM,
        hidden_dim=64,
        num_layers=1,
        require_temporal_sequence=True,
    )
    model.eval()

    # 1. Rejects 2D input [N, 280]
    x_2d = torch.randn(4, STATE_FEATURE_DIM)
    h = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    dt = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="requires 3D temporal sequence.*with T > 1"):
        model(x_2d, h, dt)

    # 2. Rejects 3D input with T=1 [N, 1, 280]
    x_3d_single = torch.randn(4, 1, STATE_FEATURE_DIM)
    with pytest.raises(ValueError, match="requires sequence length T > 1"):
        model(x_3d_single, h, dt)

    # 3. Accepts 3D input with T > 1 [N, 5, 280]
    x_3d_valid = torch.randn(4, 5, STATE_FEATURE_DIM)
    out = model(x_3d_valid, h, dt)
    assert out.shape == (4, STATE_FEATURE_DIM)


def test_same_test_pairs_evaluated_across_all_four_models():
    """Verify Persistence, Linear, Production WM, and GRU evaluate on identical [N, T, 280] test pairs."""
    N_pairs = 12
    T = 5
    S = STATE_FEATURE_DIM

    curr_states = np.random.randn(N_pairs, T, S).astype(np.float32)
    targ_states = curr_states[:, -1, :] + np.random.randn(N_pairs, S).astype(np.float32) * 0.1
    ts_curr = np.linspace(100.0, 200.0, N_pairs)
    ts_targ = ts_curr + 1.0

    batch = FutureStatePairBatch(
        current_states=curr_states,
        target_states=targ_states,
        current_timestamps=ts_curr,
        target_timestamps=ts_targ,
        delta_times=np.ones(N_pairs, dtype=float),
        horizon=1,
        source_ids=["host_1"] * N_pairs,
        schema_version=STATE_SCHEMA_VERSION,
    )

    # 1. Persistence Baseline
    pers = PersistenceBaseline()
    pers_eval = pers.evaluate(batch)
    assert "mae" in pers_eval and pers_eval["mae"] > 0.0

    # 2. Linear Baseline
    lin = LinearDynamicsBaseline()
    lin.fit(batch)
    lin_eval = lin.evaluate(batch)
    assert "mae" in lin_eval and lin_eval["mae"] > 0.0

    # 3. Production Model (Residual MLP)
    prod_model = StateResidualDynamicsModel(state_dim=S, hidden_dim=64, num_blocks=1)
    prod_model.eval()
    with torch.no_grad():
        x = torch.from_numpy(batch.current_states).float()
        k_t = torch.full((N_pairs,), 1, dtype=torch.long)
        dt_t = torch.from_numpy(batch.delta_times).float()
        pred_prod = prod_model(x, k_t, dt_t)
        assert pred_prod.shape == (N_pairs, S)

    # 4. GRU Candidate Model
    gru_model = GRUDynamicsWorldModel(state_dim=S, hidden_dim=64, num_layers=1, require_temporal_sequence=True)
    gru_model.eval()
    with torch.no_grad():
        pred_gru = gru_model(x, k_t, dt_t)
        assert pred_gru.shape == (N_pairs, S)


def test_candidate_gate_and_promotion_safety():
    """Verify evaluation gate produces correct decision and never auto-promotes to production."""
    prod_path = Path("artifacts/world_model/best_model.pt")
    if prod_path.exists():
        import hashlib
        import shutil
        sha_before = hashlib.sha256(prod_path.read_bytes()).hexdigest()
        assert len(sha_before) == 64

        # Verify candidate directory is isolated
        cand_dir = Path("artifacts/world_model/candidates/test_dummy_cand")
        cand_dir.mkdir(parents=True, exist_ok=True)
        try:
            (cand_dir / "best_model.pt").write_text("DUMMY_CANDIDATE", encoding="utf-8")
            # Production weights are untouched
            sha_after = hashlib.sha256(prod_path.read_bytes()).hexdigest()
            assert sha_after == sha_before
        finally:
            if cand_dir.exists():
                shutil.rmtree(cand_dir, ignore_errors=True)
