"""TRINETRA Unseen Attack Evaluation & Zero-Day Generalization CLI.

Executes the unseen attack holdout protocol:
    1. Withholds selected attack category/categories entirely from training data
    2. Trains models (Track A: Logistic Regression, Track B: World Model) on Known Attacks + Benign
    3. Evaluates models on Unseen Attack Category vs Known Attack Category
    4. Computes performance degradation deltas (Delta F1, Delta Recall, Delta FPR)
    5. Saves experiment artifacts and manifest into artifacts/unseen_attack/<exp_id>/

IMPORTANT:
This experiment evaluates zero-day-style generalization by withholding attack categories.
It does NOT prove universal zero-day protection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield_rca.evaluation.unseen_attack import (
    UnseenAttackConfig,
    UnseenAttackEvaluator,
    generate_synthetic_unseen_attack_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="TRINETRA Unseen Attack Evaluation CLI")
    parser.add_argument(
        "--holdout",
        nargs="+",
        default=["Bot"],
        help="One or more canonical attack categories to withhold (e.g. Bot, Infiltration, DDoS)",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/unseen_attack", help="Artifacts root directory")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment identifier")
    parser.add_argument("--samples-per-category", type=int, default=60, help="Synthetic samples per attack category")
    parser.add_argument("--benign-samples", type=int, default=200, help="Synthetic benign samples")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")

    args = parser.parse_args()

    exp_id = args.experiment_id or f"unseen_{'_'.join(args.holdout).lower()}"

    print("==========================================================")
    print("   TRINETRA AI: UNSEEN ATTACK GENERALIZATION EVALUATION   ")
    print("==========================================================")
    print("EVALUATION PROTOCOL: Zero-Day-Style Category Holdout")
    print("STATUS: Synthetic Software Validation (Pipeline Verification)")
    print(f"HELD-OUT CATEGORIES: {args.holdout}")
    print("----------------------------------------------------------")

    # 1. Config & Validation
    config = UnseenAttackConfig(
        held_out_categories=args.holdout,
        seed=args.seed,
        experiment_id=exp_id,
    )
    try:
        config.validate()
        print(f"Configuration validated. Canonical held-out: {config.held_out_categories}")
    except Exception as e:
        print(f"CONFIGURATION ERROR: {e}")
        sys.exit(1)

    # 2. Dataset Generation / Partitioning
    print("\nGenerating synthetic partitioned dataset (70% Train, 15% Val, 15% Test)...")
    dataset = generate_synthetic_unseen_attack_dataset(
        held_out_category=config.held_out_categories[0],
        samples_per_category=args.samples_per_category,
        benign_samples=args.benign_samples,
        seed=args.seed,
    )
    print(f"Training samples:   {len(dataset.X_train)} (Known Attacks + Benign)")
    print(f"Validation samples: {len(dataset.X_val)} (Known Attacks + Benign)")
    print(f"Known Test samples: {len(dataset.X_test_known)} (Known Attacks + Benign)")
    print(f"Unseen Test samples:{len(dataset.X_test_unseen)} (Unseen [{config.held_out_categories[0]}] + Benign)")

    # 3. Leakage Verification
    try:
        dataset.verify_no_leakage(config.held_out_categories)
        print("LEAKAGE CHECK: PASSED. Zero held-out samples in training/validation splits.")
    except Exception as e:
        print(f"LEAKAGE CHECK FAILED: {e}")
        sys.exit(1)

    # 4. Evaluator Execution
    evaluator = UnseenAttackEvaluator(config=config, output_dir=args.output_dir)

    print("\nExecuting Track A (Logistic Regression on Unseen Generalization)...")
    result = evaluator.run_experiment(dataset=dataset)

    # 5. Output Results
    track_a = result.track_a_logistic
    known_m = track_a["known_attack_metrics"]
    unseen_m = track_a["unseen_attack_metrics"]
    deg = track_a["performance_degradation"]

    print("\n--- GENERALIZATION BENCHMARK RESULTS ---")
    print(f"{'Metric':<25} | {'Known Attack':<15} | {'Unseen Attack':<15} | {'Degradation (Delta)':<15}")
    print("-" * 75)
    print(f"{'Precision':<25} | {known_m['precision']:<15.4f} | {unseen_m['precision']:<15.4f} | {deg['delta_recall']:+.4f}")
    print(f"{'Recall':<25} | {known_m['recall']:<15.4f} | {unseen_m['recall']:<15.4f} | {deg['delta_recall']:+.4f}")
    print(f"{'F1 Score':<25} | {known_m['f1_score']:<15.4f} | {unseen_m['f1_score']:<15.4f} | {deg['delta_f1']:+.4f}")
    print(f"{'False Positive Rate':<25} | {known_m['false_positive_rate']:<15.4f} | {unseen_m['false_positive_rate']:<15.4f} | {deg['delta_fpr']:+.4f}")

    print(f"\nKnown Attack Confusion Matrix:  {known_m['confusion_matrix']}")
    print(f"Unseen Attack Confusion Matrix: {unseen_m['confusion_matrix']}")

    print(f"\nArtifacts saved successfully to: {evaluator.output_dir}")
    print("  * experiment_config.json")
    print("  * experiment_manifest.json")
    print("  * metrics.json")
    print("  * results.json")
    print("  * confusion_matrix.json")
    print("==========================================================")


if __name__ == "__main__":
    main()
