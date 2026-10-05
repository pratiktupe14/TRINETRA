"""TRINETRA Offline RLM Learning & Candidate Model Evaluation CLI.

Executes the controlled offline feedback loop:
    1. Ingests and validates RLM LearningRecords from artifacts/rlm/
    2. Builds candidate training, validation, and test datasets
    3. Trains candidate model in artifacts/world_model/candidates/<candidate_id>/
    4. Evaluates candidate against current production model (Evaluation Gate)
    5. Emits CandidateEvaluationReport
    6. Optionally promotes candidate if and only if explicit operator approval is given.

NOTE: This script is OFFLINE ONLY. Live incidents never invoke it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield_rca.rlm_learning_pipeline import RLMLearningPipeline


def main():
    parser = argparse.ArgumentParser(description="TRINETRA RLM Offline Learning Pipeline")
    parser.add_argument("--candidate-id", type=str, default=None, help="Unique identifier for candidate model")
    parser.add_argument("--records-file", type=str, default="artifacts/rlm/learning_records.jsonl", help="Path to RLM JSONL file")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/world_model", help="Path to World Model artifacts directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for candidate")
    parser.add_argument("--approve-release", action="store_true", help="Explicitly approve promoting candidate if evaluation passes")
    parser.add_argument("--force-release", action="store_true", help="Force promotion even if evaluation gate reports warnings")

    args = parser.parse_args()

    pipeline = RLMLearningPipeline(
        artifacts_dir=args.artifacts_dir,
        rlm_records_path=args.records_file,
    )

    print("==========================================================")
    print("   TRINETRA AI: RLM OFFLINE CANDIDATE LEARNING PIPELINE   ")
    print("==========================================================")

    # 1. Load and validate records
    print(f"Loading RLM learning records from {args.records_file}...")
    records = pipeline.load_and_validate_records()
    print(f"Successfully loaded and validated {len(records)} learning records.")

    # 2. Build candidate dataset
    print("Constructing candidate multi-horizon future datasets...")
    train_ds, val_ds, test_ds = pipeline.build_candidate_dataset(records)
    print(f"Dataset ready: {train_ds.total_usable_pairs} train pairs, {val_ds.total_usable_pairs} val pairs, {test_ds.total_usable_pairs} test pairs.")

    # 3. Train candidate model
    import uuid
    candidate_id = args.candidate_id or f"cand-{uuid.uuid4().hex[:8]}"
    print(f"\nTraining candidate model '{candidate_id}' for {args.epochs} epochs...")
    candidate_dir = pipeline.train_candidate(
        candidate_id=candidate_id,
        train_ds=train_ds,
        val_ds=val_ds,
        max_epochs=args.epochs,
    )
    print(f"Candidate model saved to: {candidate_dir}")

    # 4. Evaluation Gate
    print("\nRunning Evaluation Gate (Candidate vs. Production Baseline)...")
    eval_report = pipeline.evaluate_candidate(
        candidate_id=candidate_id,
        test_ds=test_ds,
        num_records=len(records),
    )

    print("\n--- EVALUATION GATE REPORT ---")
    print(f"Candidate ID:        {eval_report.candidate_id}")
    print(f"Current Model MAE:   {eval_report.current_model_mae:.6f}")
    print(f"Candidate Model MAE: {eval_report.candidate_model_mae:.6f}")
    print(f"Improvement:         {eval_report.mae_improvement_pct:+.2f}%")
    print(f"DECISION:            {eval_report.evaluation_decision}")
    for reason in eval_report.decision_reasons:
        print(f"  * {reason}")

    # 5. Promotion Gate
    if args.approve_release:
        print("\nOperator release approval flag detected. Attempting promotion...")
        try:
            prod_path = pipeline.promote_candidate(
                candidate_id=candidate_id,
                operator_approved=True,
                force=args.force_release,
            )
            print(f"SUCCESS: Candidate {candidate_id} released to production at {prod_path}")
        except Exception as e:
            print(f"PROMOTION REJECTED: {e}")
            sys.exit(1)
    else:
        print(f"\nNOTE: Candidate model remains isolated in {candidate_dir}.")
        print("Production weights (artifacts/world_model/best_model.pt) were NOT modified.")
        print("To promote this candidate after verification, rerun with --approve-release.")

    print("==========================================================")


if __name__ == "__main__":
    main()
