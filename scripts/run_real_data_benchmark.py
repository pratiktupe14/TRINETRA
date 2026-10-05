"""TRINETRA Real-Data Evaluation & Benchmarking CLI.

Executes real-data benchmarking on open-source labelled traffic datasets
(CIC-IDS2018 / CTU-13) using TRINETRA's modular evaluation framework:
    1. Dataset discovery & cryptographic provenance (SHA-256)
    2. Data Quality Audit
    3. Chronological leakage-safe split (70/15/15)
    4. Feature coverage audit (78 flow-level, 18 packet-level)
    5. Supervised Logistic Regression baseline
    6. World Model multi-horizon forecasting (k=1, 3, 5 vs Persistence & Linear baselines)
    7. Unseen attack generalization evaluation
    8. Risk estimation, MITRE stage mapping, and explainability attribution
    9. CyberHeal dry-run / simulation integration
    10. Isolated artifact persistence

CRITICAL ANTI-FABRICATION RULE:
If the dataset directory is missing or empty, this CLI exits with an error:
"Real dataset not found. No benchmark metrics were generated."
It does NOT silently generate or report synthetic data as real benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield_rca.evaluation.real_data_benchmark import (
    RealDataBenchmarkRunner,
    discover_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRINETRA Real-Data Evaluation & Benchmarking CLI"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cic-ids2018",
        choices=["cic-ids2018", "ctu13"],
        help="Target cybersecurity dataset name (cic-ids2018 or ctu13)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Local directory containing the real dataset files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for benchmark reports and manifests",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "csv", "pcap", "pcapng"],
        help="Dataset format: csv, pcap, pcapng, or auto (default)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to load (useful for smoke tests or bounded execution)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for memory-safe CSV reading (default: 100,000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Enforce simulation-only CyberHeal execution (default: True)",
    )

    args = parser.parse_args()

    data_dir_path = Path(args.data_dir)
    print("==========================================================")
    print("   TRINETRA AI: REAL-DATA EVALUATION & BENCHMARKING       ")
    print("==========================================================")
    print(f"TARGET DATASET: {args.dataset.upper()}")
    print(f"SEARCH DIRECTORY: {data_dir_path}")
    print(f"INPUT FORMAT: {args.input_format}")
    print(f"CHUNK SIZE: {args.chunk_size:,} rows")
    print(f"REMEDIATION SAFETY: {'SIMULATION / DRY_RUN' if args.dry_run else 'LIVE'}")
    print("----------------------------------------------------------")

    # Step 1: Pre-flight check for dataset presence
    detected_format, files = discover_dataset(data_dir_path, args.input_format)
    if detected_format in ("not_found", "none") or not files:
        print(
            f"ERROR: Real dataset not found in '{args.data_dir}'. "
            "No benchmark metrics were generated.",
            file=sys.stderr,
        )
        print("\nOperator Action Required:")
        print(f"1. Download or place the official {args.dataset.upper()} files into: {args.data_dir}")
        print("2. Re-run this command once the data is present.")
        print("3. See docs/TRINETRA_PHASE9_DATASET_STATUS.md for detailed setup instructions.")
        return 1

    print(f"FOUND: {len(files)} files matching format '{detected_format}'.")

    # Step 2: Execute benchmark runner
    runner = RealDataBenchmarkRunner(
        dataset_name=args.dataset,
        data_dir=data_dir_path,
        output_dir=args.output_dir,
        input_format=args.input_format,
        dry_run=args.dry_run,
        chunksize=args.chunk_size,
    )

    try:
        results = runner.run_benchmark(max_rows=args.max_rows)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"BENCHMARK EXECUTION FAILED: {e}", file=sys.stderr)
        return 1

    # Step 3: Print summary tables
    prov = results.get("provenance", {})
    qual = results.get("quality_report", {})
    lr = results.get("logistic_baseline", {})
    wm = results.get("world_model_forecasting", {})
    unseen = results.get("unseen_attack", {})

    print("\n==========================================================")
    print("   BENCHMARK EXECUTION SUMMARY                            ")
    print("==========================================================")
    print(f"Records Ingested: {prov.get('row_count', 0):,}")
    print(f"Benign Samples:   {qual.get('benign_count', 0):,}")
    print(f"Attack Samples:   {qual.get('attack_count', 0):,} ({qual.get('attack_ratio', 0.0)*100:.2f}%)")
    print("----------------------------------------------------------")
    print("TABLE A — LOGISTIC REGRESSION CLASSIFICATION BASELINE")
    print(f"Precision: {lr.get('precision', 0.0):.4f}")
    print(f"Recall:    {lr.get('recall', 0.0):.4f}")
    print(f"F1 Score:  {lr.get('f1_score', 0.0):.4f}")
    print(f"FPR:       {lr.get('false_positive_rate', 0.0):.4f}")
    print("----------------------------------------------------------")
    print("TABLE B — WORLD MODEL MULTI-HORIZON FORECASTING")
    if "results_by_horizon" in wm:
        print(f"{'Horizon':<8} | {'Persistence MAE':<16} | {'World Model MAE':<16} | {'Improvement':<12}")
        print("-" * 60)
        for h_key, d in wm["results_by_horizon"].items():
            p_mae = d.get("persistence_baseline", {}).get("mae", 0.0)
            w_mae = d.get("world_model", {}).get("mae", 0.0)
            imp = d.get("improvement_over_persistence_pct", 0.0)
            print(f"{h_key:<8} | {p_mae:<16.4f} | {w_mae:<16.4f} | {imp:+.2f}%")
    else:
        print(f"Forecasting status: {wm.get('status', 'EVALUATED')}")
    print("----------------------------------------------------------")
    print("TABLE C — UNSEEN ATTACK GENERALIZATION")
    if "known_attack_metrics" in unseen:
        print(f"Held-out Category: {unseen.get('held_out_category')}")
        print(f"Known Attack F1:   {unseen['known_attack_metrics'].get('f1_score', 0.0):.4f}")
        print(f"Unseen Attack F1:  {unseen['unseen_attack_metrics'].get('f1_score', 0.0):.4f}")
        print(f"Delta F1:          {unseen.get('performance_degradation', {}).get('delta_f1', 0.0):+.4f}")
    else:
        reason = f" - {unseen.get('reason', '')}" if 'reason' in unseen else ""
        print(f"Unseen attack status: {unseen.get('status', 'NOT_APPLICABLE')}{reason}")
    print("==========================================================")
    print(f"Artifacts saved to: {runner.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
