"""Demonstration and Training Pipeline for the CyberHeal World Model.

Trains StateResidualDynamicsModel on deterministic multi-horizon synthetic dynamics:
    S_{t+k} = W_theta(S_t, k, Delta tau)
Compares against Persistence and Linear baselines, logs per-group errors,
and saves model artifacts to artifacts/world_model/.

NOTE: Labeled explicitly as SYNTHETIC SANITY CHECK until real CIC-IDS captures are integrated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import torch

from shield_rca.network_state.future_dataset import (
    FutureStatePairBatch,
    MultiHorizonFutureDataset,
)
from shield_rca.network_state.preprocessor import NetworkStatePreprocessor
from shield_rca.network_state.quality_report import generate_future_dataset_quality_report
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from shield_rca.world_model.baselines import LinearDynamicsBaseline, PersistenceBaseline
from shield_rca.world_model.evaluator import WorldModelEvaluator
from shield_rca.world_model.model import StateResidualDynamicsModel
from shield_rca.world_model.trainer import WorldModelTrainer


def generate_synthetic_trajectory_data(
    n_samples: int,
    horizons: list[int] = [1, 3, 5],
    seed: int = 42,
    base_time: float = 0.0,
) -> MultiHorizonFutureDataset:
    """Generate deterministic synthetic trajectory with known temporal dynamics."""
    np.random.seed(seed)
    current_states = np.random.randn(n_samples, STATE_FEATURE_DIM).astype(np.float32)

    batches: dict[int, FutureStatePairBatch] = {}
    dropped_counts: dict[int, int] = {}

    for k in horizons:
        # Simulate realistic multi-step dynamics: state drift proportional to k + elapsed time
        dts = (k * 2.5 + np.random.uniform(0.1, 0.5, size=n_samples)).astype(np.float32)
        cur_ts = base_time + np.arange(n_samples, dtype=float) * 10.0
        targ_ts = cur_ts + dts

        # Known transition: decay of magnitude + non-linear delta surge
        decay = 0.95 ** k
        drift = 0.05 * k * np.sin(current_states)
        targ_states = (decay * current_states + drift).astype(np.float32)

        batches[k] = FutureStatePairBatch(
            current_states=current_states,
            target_states=targ_states,
            current_timestamps=cur_ts,
            target_timestamps=targ_ts,
            delta_times=dts,
            horizon=k,
            source_ids=["synthetic-session"] * n_samples,
            schema_version=STATE_SCHEMA_VERSION,
        )
        dropped_counts[k] = k

    return MultiHorizonFutureDataset(
        batches=batches,
        horizons=horizons,
        total_states_input=n_samples + max(horizons),
        dropped_tail_pairs=dropped_counts,
        schema_version=STATE_SCHEMA_VERSION,
    )


def run_training_experiment():
    print("==========================================================")
    print("   CYBERHEAL AI: WORLD MODEL TRAINING (SYNTHETIC SANITY)   ")
    print("==========================================================")

    horizons = [1, 3, 5]
    print(f"Generating deterministic synthetic data (horizons: {horizons})...")
    train_ds = generate_synthetic_trajectory_data(120, horizons=horizons, seed=42, base_time=0.0)
    val_ds = generate_synthetic_trajectory_data(30, horizons=horizons, seed=142, base_time=2000.0)
    test_ds = generate_synthetic_trajectory_data(30, horizons=horizons, seed=242, base_time=4000.0)

    # 1. State Preprocessing (Fit strictly on TRAIN)
    print("Fitting NetworkStatePreprocessor strictly on training states...")
    preprocessor = NetworkStatePreprocessor(scaler_type="standard")
    # Collect all train current states
    all_train_curr = np.vstack([train_ds.batches[k].current_states for k in horizons])
    preprocessor.fit(all_train_curr)

    # 2. Fit Linear Dynamics Baselines on Train
    linear_baselines: dict[int, LinearDynamicsBaseline] = {}
    for k in horizons:
        lin = LinearDynamicsBaseline(alpha=1.0)
        lin.fit(train_ds.batches[k])
        linear_baselines[k] = lin

    # 3. Model Architecture
    model = StateResidualDynamicsModel(
        state_dim=STATE_FEATURE_DIM,
        hidden_dim=128,
        num_blocks=2,
        dropout=0.02,
        max_horizon=16,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"World Model architecture initialized (Parameters: {param_count:,})")

    # 4. Train Model
    trainer = WorldModelTrainer(
        model=model,
        learning_rate=2e-3,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=20,
        patience=5,
        seed=42,
    )
    print("Training World Model...")
    history = trainer.fit(train_ds, val_ds)
    print(f"Training completed in {len(history)} epochs. Best val loss: {trainer.best_val_loss:.6f}")

    # 5. Evaluate Test Set
    evaluator = WorldModelEvaluator(model)
    eval_report = evaluator.evaluate_multi_horizon(test_ds, linear_baselines=linear_baselines)

    print("\n--- TEST EVALUATION RESULTS ---")
    for k in horizons:
        res = eval_report["results_by_horizon"][f"k_{k}"]
        wm_mae = res["world_model"]["mae"]
        pers_mae = res["persistence_baseline"]["mae"]
        lin_mae = res["linear_baseline"]["mae"]
        imp_pers = res["improvement_over_persistence_pct"]
        imp_lin = res["improvement_over_linear_pct"]

        print(f"\n[Horizon k={k}] ({res['num_pairs']} test pairs):")
        print(f"  Persistence Baseline MAE: {pers_mae:.6f}")
        print(f"  Linear Baseline MAE:      {lin_mae:.6f}")
        print(f"  World Model MAE:          {wm_mae:.6f}")
        print(f"  Improvement over Persist: {imp_pers:+.2f}%")
        print(f"  Improvement over Linear:  {imp_lin:+.2f}%")
        print("  Group MAE breakdown (World Model):")
        for g_name, g_val in res["group_errors"].items():
            print(f"    - {g_name}: {g_val:.6f}")

    # 6. Save Artifacts
    artifact_dir = Path("artifacts/world_model")
    trainer.save(artifact_dir, horizons=horizons)
    preprocessor.save(artifact_dir)

    # Save evaluation summary
    eval_path = artifact_dir / "test_evaluation_summary.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nArtifacts successfully saved to {artifact_dir}")
    print("==========================================================")


if __name__ == "__main__":
    run_training_experiment()
