"""Phase 10B CLI: Real CIC-IDS2018 Temporal World-Model Candidate Training & Evaluation.

Trains GRUDynamicsWorldModel on real CIC-IDS2018 temporal state sequences [N, T, 280]
and compares it directly against Persistence, Linear Dynamics, and the Production
World Model on identical test pairs.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield_rca.world_model.real_data_pipeline import (
    load_cic_ids2018_temporal_datasets,
    train_and_evaluate_gru_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10B Real CIC-IDS2018 Temporal World-Model Candidate CLI"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/cic_ids2018/processed",
        help="Path to CIC-IDS2018 CSV directory",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Maximum rows to load (default: 100,000 benchmark cap)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Temporal state sequence length T (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum epochs for training (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--candidate-id",
        type=str,
        default=None,
        help="Unique candidate ID (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )

    args = parser.parse_args()

    candidate_id = args.candidate_id or f"cand-gru-real-{uuid.uuid4().hex[:8]}"

    print("==================================================================")
    print("   PHASE 10B: REAL CIC-IDS2018 TEMPORAL WORLD MODEL TRAINING     ")
    print("==================================================================")
    print(f"Dataset Directory:   {args.data_dir}")
    print(f"Max Ingestion Rows:  {args.max_rows:,}")
    print(f"Sequence Length (T): {args.sequence_length}")
    print(f"Candidate ID:        {candidate_id}")
    print(f"Training Epochs:     {args.epochs}")
    print(f"Random Seed:         {args.seed}")
    print("------------------------------------------------------------------")

    # 1. Load real data and construct leakage-free temporal datasets
    print("Step 1: Loading real CIC-IDS2018 data and building temporal sequences...")
    datasets = load_cic_ids2018_temporal_datasets(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        sequence_length=args.sequence_length,
        horizons=[1, 3, 5],
        chunksize=100_000,
    )

    print("  * Real records ingested:  ", f"{datasets.provenance['row_count']:,}")
    print("  * Usable flow records:    ", f"{datasets.quality_report['usable_rows']:,}")
    print("  * Chronological split:     70% train / 15% val / 15% test")
    print("  * Train samples:          ", f"{datasets.split_summary['train_samples']:,}")
    print("  * Val samples:            ", f"{datasets.split_summary['val_samples']:,}")
    print("  * Test samples:           ", f"{datasets.split_summary['test_samples']:,}")
    print("  * Train temporal pairs:   ", f"{datasets.train_ds.total_usable_pairs:,}")
    print("  * Val temporal pairs:     ", f"{datasets.val_ds.total_usable_pairs:,}")
    print("  * Test temporal pairs:    ", f"{datasets.test_ds.total_usable_pairs:,}")
    print("  * Input sequence shape:   ", f"[N, {datasets.sequence_length}, 280]")

    # 2. Train and evaluate GRU candidate
    print("\nStep 2: Training GRUDynamicsWorldModel & evaluating all models...")
    report = train_and_evaluate_gru_candidate(
        datasets=datasets,
        candidate_id=candidate_id,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # 3. Print Results Summary
    print("\n==================================================================")
    print("   MULTI-MODEL FORECASTING BENCHMARK RESULTS (TEST SPLIT)         ")
    print("==================================================================")
    print(f"{'Horizon':<8} | {'Persistence MAE':<16} | {'Linear MAE':<16} | {'Production MAE':<16} | {'GRU MAE':<16} | {'vs Prod':<10}")
    print("-" * 92)

    for k in [1, 3, 5]:
        k_key = f"k_{k}"
        res = report["results_by_horizon"][k_key]
        pers_m = res["persistence"]["mae"]
        lin_m = res["linear"]["mae"] if res["linear"] else float("nan")
        prod_m = res["production"]["mae"] if res["production"] else float("nan")
        gru_m = res["gru_candidate"]["mae"]
        vs_prod = res["improvement_over_production_pct"]
        print(f"{k_key:<8} | {pers_m:<16.4f} | {lin_m:<16.4f} | {prod_m:<16.4f} | {gru_m:<16.4f} | {vs_prod:+.2f}%")

    print("-" * 92)
    print(f"Overall Mean Production MAE: {report['mean_production_mae']:.4f}")
    print(f"Overall Mean GRU MAE:        {report['mean_gru_mae']:.4f}")
    print(f"Overall Improvement vs Prod: {report['improvement_over_production_pct']:+.2f}%")
    print(f"Overall Improvement vs Pers: {report['improvement_over_persistence_pct']:+.2f}%")
    print("------------------------------------------------------------------")
    print(f"CANDIDATE GATE DECISION:     {report['evaluation_decision']}")
    for r in report["decision_reasons"]:
        print(f"  * {r}")
    print(f"PRODUCTION MODEL MODIFIED:   {'YES' if report['production_weights_modified'] else 'NO'}")
    print(f"CANDIDATE ARTIFACT DIR:      artifacts/world_model/candidates/{candidate_id}")
    print("==================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
