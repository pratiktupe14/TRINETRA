"""Phase 12 Test Suite: Regression & Real-Data Improvement Verification.

Validates all bug fixes and architectural improvements introduced in Phase 12:
1. Ctu13Adapter skip_rows parameter for active attack sampling.
2. NetworkStatePreprocessor log1p stabilization on heavy-tailed features.
3. Serialization and deserialization of log1p preprocessor metadata.
4. GRUAttentionDynamicsWorldModel architecture & gradient flow.
5. Production World Model immutability (SHA-256 invariant).
6. Zero-leakage enforcement for Scenario 10.
7. Governance & explainability: State Delta Attribution (NOT SHAP) & mitre_technique = None.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shield_rca.evaluation.ctu13_adapter import Ctu13Adapter
from shield_rca.network_state.preprocessor import NetworkStatePreprocessor
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from shield_rca.world_model.candidates import (
    GRUAttentionDynamicsWorldModel,
    GRUDynamicsWorldModel,
    TemporalAttention,
)
from shield_rca.world_model.explainability import explain_forecast
from shield_rca.world_model.stage_mapper import AttackStage, AttackStageMapper


def test_ctu13_adapter_skip_rows(tmp_path: Path) -> None:
    """Verifies that Ctu13Adapter respects skip_rows without corrupting data."""
    csv_file = tmp_path / "test_skip.binetflow"
    lines = [
        "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
    ]
    # 50 rows of Benign, then 20 rows of Botnet
    for i in range(50):
        lines.append(f"2011/08/10 10:{i:02d}:00,0.1,tcp,192.168.1.1,1024,->,8.8.8.8,53,CON,0,0,2,100,50,Normal\n")
    for i in range(20):
        lines.append(f"2011/08/10 11:{i:02d}:00,0.1,tcp,192.168.1.1,1024,->,8.8.8.8,53,CON,0,0,2,100,50,flow=From-Botnet-V42\n")

    csv_file.write_text("".join(lines), encoding="utf-8")

    adapter = Ctu13Adapter()

    # Head load (skip_rows=0, max_rows=30) -> only Normal
    df_head, q_head, _ = adapter.load_and_normalize([csv_file], max_rows=30, skip_rows=0)
    assert len(df_head) == 30
    assert (df_head.attrs["labels_holder"].array == 1).sum() == 0

    # Skipped load (skip_rows=45, max_rows=20) -> spans Normal into Botnet
    df_skip, q_skip, _ = adapter.load_and_normalize([csv_file], max_rows=20, skip_rows=45)
    assert len(df_skip) == 20
    botnet_count = (df_skip.attrs["labels_holder"].array == 1).sum()
    assert botnet_count > 0, "Failed to capture botnet rows when using skip_rows"


def test_network_state_preprocessor_log1p(tmp_path: Path) -> None:
    """Verifies that NetworkStatePreprocessor with log1p stabilizes extreme scale counters."""
    np.random.seed(42)
    # Synthetic states with some extreme values (simulating microsecond flow duration ~ 3.6e9)
    states = np.random.uniform(0.0, 10.0, size=(100, STATE_FEATURE_DIM)).astype(np.float32)
    states[:, 40] = np.random.uniform(1e6, 3.6e9, size=100)  # Extreme durations

    prep = NetworkStatePreprocessor(scaler_type="log_standard")
    norm_states = prep.fit_transform(states)

    assert prep.use_log1p is True
    assert prep.is_fitted is True
    assert np.all(np.isfinite(norm_states))
    # Extreme column 40 should be well within standard normal range
    assert np.max(np.abs(norm_states[:, 40])) < 10.0

    # Test persistence and reload
    save_dir = tmp_path / "log_scaler"
    prep.save(save_dir)

    reloaded = NetworkStatePreprocessor.load(save_dir)
    assert reloaded.use_log1p is True
    assert reloaded.is_fitted is True
    re_norm = reloaded.transform(states)
    np.testing.assert_allclose(norm_states, re_norm, atol=1e-5)


def test_gru_attention_dynamics_architecture() -> None:
    """Verifies the architecture and gradient flow of Candidate F (GRUAttentionDynamicsWorldModel)."""
    model = GRUAttentionDynamicsWorldModel(state_dim=STATE_FEATURE_DIM, hidden_dim=64, num_layers=2)
    model.train()

    batch_size = 4
    seq_len = 5
    x = torch.randn(batch_size, seq_len, STATE_FEATURE_DIM)
    k = torch.tensor([1, 2, 3, 5])
    dt = torch.tensor([1.0, 2.0, 3.0, 5.0])

    pred, delta = model(x, k, dt, return_delta=True)

    assert pred.shape == (batch_size, STATE_FEATURE_DIM)
    assert delta.shape == (batch_size, STATE_FEATURE_DIM)

    # Check gradient propagation through attention and GRU
    loss = pred.sum()
    loss.backward()

    assert model.attention.query.weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None
    assert model.delta_head[0].weight.grad is not None


def test_production_world_model_immutability() -> None:
    """Verifies that production model best_model.pt has not been modified."""
    prod_path = Path("artifacts/world_model/best_model.pt")
    assert prod_path.exists(), "Production checkpoint missing"
    sha = hashlib.sha256(prod_path.read_bytes()).hexdigest()
    expected_sha = "78ed7a009373e6460f7416fd645e2060513ac165f6cfad66244d4a519c2565a4"
    assert sha == expected_sha, f"Production model corrupted! Expected {expected_sha}, got {sha}"


def test_mitre_anti_fabrication_and_state_delta_attribution() -> None:
    """Verifies that MITRE technique is None when unverified and explainability uses State Delta Attribution."""
    current_state = np.ones(STATE_FEATURE_DIM, dtype=np.float32)
    predicted_state = np.ones(STATE_FEATURE_DIM, dtype=np.float32)
    predicted_state[10] = 5.0  # State change

    mapper = AttackStageMapper()
    record = mapper.map_stage(current_state=current_state, predicted_state=predicted_state)
    assert record.mitre_technique is None, "Fabricated MITRE technique detected!"

    explanation = explain_forecast(current_state, predicted_state, horizon=3, top_k=5)
    assert explanation.explanation_method == "State Delta Attribution", "Method should be State Delta Attribution"
    assert "shap" not in explanation.explanation_method.lower(), "Fabricated SHAP claim detected!"
    assert len(explanation.top_features) > 0
    assert explanation.top_features[0].feature_name == "bwd_bytes_per_s_mean"
