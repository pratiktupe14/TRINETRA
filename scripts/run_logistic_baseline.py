"""TRINETRA Logistic Regression Baseline & SIH Benchmarking CLI.

Executes the classical ML benchmark required by SIH:
    1. Prepares leakage-safe chronological partitions (70% Train, 15% Val, 15% Test)
    2. Fits StandardScaler strictly on training samples
    3. Fits Logistic Regression with class-weight balancing
    4. Evaluates test set on SIH metrics: Precision, Recall, F1, and False Positive Rate (FPR)
    5. Saves model artifacts and reports to artifacts/logistic_baseline/

IMPORTANT:
This evaluates SUPERVISED ATTACK CLASSIFICATION.
The TRINETRA World Model evaluates CONTINUOUS FUTURE-STATE DYNAMICS.
These are complementary evaluation tracks, not interchangeable metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield_rca.evaluation.logistic_baseline import (
    LogisticBaselineConfig,
    LogisticRegressionBaseline,
    generate_synthetic_classification_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="TRINETRA Logistic Regression Baseline Benchmark")
    parser.add_argument("--output-dir", type=str, default="artifacts/logistic_baseline", help="Directory to save model artifacts")
    parser.add_argument("--c", type=float, default=1.0, help="Regularization parameter C")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations")
    parser.add_argument("--solver", type=str, default="liblinear", help="Scikit-learn solver")
    parser.add_argument("--samples", type=int, default=300, help="Number of synthetic samples (if synthetic mode)")
    parser.add_argument("--window-size", type=int, default=20, help="Window length T")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary classification")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")

    args = parser.parse_args()

    print("==========================================================")
    print("   TRINETRA AI: LOGISTIC REGRESSION BENCHMARKING (SIH)    ")
    print("==========================================================")
    print("TASK: Supervised Network Attack Classification")
    print("STATUS: Synthetic Sanity Validation (Software Verification)")
    print("NOTE: Real-world SIH benchmark datasets (CIC-IDS/CTU-13) are evaluated separately.")
    print("----------------------------------------------------------")

    # 1. Dataset Generation
    print(f"Generating deterministic synthetic data ({args.samples} samples, window={args.window_size}, seed={args.seed})...")
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_classification_dataset(
        n_samples=args.samples,
        window_size=args.window_size,
        seed=args.seed,
    )
    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} (Chronological 70/15/15)")

    # 2. Model Initialization
    config = LogisticBaselineConfig(
        C=args.c,
        max_iter=args.max_iter,
        solver=args.solver,
        random_state=args.seed,
        decision_threshold=args.threshold,
    )
    baseline = LogisticRegressionBaseline(config=config)

    # 3. Training
    print("\nFitting Logistic Regression baseline (Scaler fitted ONLY on Train)...")
    baseline.fit(X_train, y_train)

    # 4. Evaluation
    print("Evaluating test set...")
    metrics = baseline.evaluate(X_test, y_test)

    # 5. Output Report
    print("\n--- SIH CLASSIFICATION BENCHMARK REPORT ---")
    print(f"Precision:            {metrics.precision:.4f} ({metrics.precision * 100:.2f}%)")
    print(f"Recall:               {metrics.recall:.4f} ({metrics.recall * 100:.2f}%)")
    print(f"F1 Score:             {metrics.f1_score:.4f} ({metrics.f1_score * 100:.2f}%)")
    print(f"False Positive Rate:  {metrics.false_positive_rate:.4f} ({metrics.false_positive_rate * 100:.2f}%)")
    print(f"ROC-AUC:              {metrics.roc_auc:.4f}" if metrics.roc_auc else "ROC-AUC:              N/A")
    print(f"PR-AUC:               {metrics.pr_auc:.4f}" if metrics.pr_auc else "PR-AUC:               N/A")
    print(f"\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  {metrics.confusion_matrix[0]}")
    print(f"  {metrics.confusion_matrix[1]}")
    print(f"Counts: TP={metrics.tp}, TN={metrics.tn}, FP={metrics.fp}, FN={metrics.fn}")
    print(f"Support: BENIGN={metrics.support_benign}, ATTACK={metrics.support_attack} (Total={metrics.total_samples})")

    # 6. Save Artifacts
    output_path = baseline.save(args.output_dir, metrics_report=metrics)
    print(f"\nArtifacts successfully saved to: {output_path}")
    print("  * model.joblib")
    print("  * scaler.joblib")
    print("  * model_config.json")
    print("  * model_manifest.json")
    print("  * metrics.json")
    print("  * confusion_matrix.json")
    print("==========================================================")


if __name__ == "__main__":
    main()
