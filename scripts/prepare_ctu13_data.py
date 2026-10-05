"""Copies and verifies official CTU-13 .binetflow files into CyberHeal data/ctu13/."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shield_rca.evaluation.ctu13_adapter import (
    CTU13_OFFICIAL_SCENARIOS,
    Ctu13Adapter,
)


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    source_all_dir = Path(r"D:\software download\CTU13_ALL")
    source_ctu13_dir = Path(r"D:\software download\CTU13\CTU-13-Dataset")
    dest_base = Path(r"D:\sih 2026\CyberHeal\data\ctu13")
    artifacts_dir = Path(r"D:\sih 2026\CyberHeal\artifacts\phase11b")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STEP 1: AUDIT CTU-13 SOURCE DATA")
    print("==================================================")

    # 1. Audit D:\software download\CTU13_ALL
    all_binetflows = sorted(list(source_all_dir.rglob("*.binetflow")))
    print(f"Discovered in CTU13_ALL: {len(all_binetflows)} .binetflow files:")
    for f in all_binetflows:
        scen = Ctu13Adapter.detect_scenario_id(f)
        print(f"  * {f.name} ({f.stat().st_size:,} bytes) -> {scen}")

    # 2. Audit primary complete source
    complete_binetflows = sorted(list(source_ctu13_dir.rglob("*.binetflow")))
    print(f"\nDiscovered in primary CTU-13 source: {len(complete_binetflows)} .binetflow files:")
    scenario_files: dict[str, Path] = {}
    for f in complete_binetflows:
        scen = Ctu13Adapter.detect_scenario_id(f)
        scenario_files[scen] = f
        print(f"  * [{scen}] {f.name} ({f.stat().st_size:,} bytes) at {f}")

    # Check for scenario 10 specifically
    if "scenario_10" not in scenario_files:
        print("\nERROR: Scenario 10 is missing! Stop condition met.")
        return 1
    print(f"\nCONFIRMED: Scenario 10 exists: {scenario_files['scenario_10']}")

    # Check for all 13 scenarios
    missing = [f"scenario_{i}" for i in range(1, 14) if f"scenario_{i}" not in scenario_files]
    if missing:
        print(f"\nERROR: Missing scenarios: {missing}")
        return 1
    print("CONFIRMED: All 13 scenarios (1 to 13) are present.")

    print("\n==================================================")
    print("STEP 2: PLACE DATA INTO CYBERHEAL (data/ctu13)")
    print("==================================================")
    dest_base.mkdir(parents=True, exist_ok=True)

    provenance_records = []
    total_bytes_copied = 0

    for i in range(1, 14):
        scen_key = f"scenario_{i}"
        src_path = scenario_files[scen_key]
        dest_scen_dir = dest_base / str(i)
        dest_scen_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_scen_dir / src_path.name

        print(f"Copying {scen_key}: {src_path.name} ({src_path.stat().st_size:,} bytes)...")
        if not dest_path.exists() or dest_path.stat().st_size != src_path.stat().st_size:
            shutil.copy2(src_path, dest_path)
            print(f"  -> Copied to {dest_path}")
        else:
            print(f"  -> Already exists and size matches at {dest_path}")

        file_size = dest_path.stat().st_size
        total_bytes_copied += file_size
        print(f"  -> Computing SHA-256 for {dest_path.name}...")
        sha256_hash = compute_sha256(dest_path)
        print(f"  -> SHA-256: {sha256_hash}")

        provenance_records.append({
            "scenario_id": scen_key,
            "scenario_number": i,
            "botnet_family": CTU13_OFFICIAL_SCENARIOS[scen_key]["botnet"],
            "filename": src_path.name,
            "source_path": str(src_path),
            "destination_path": str(dest_path),
            "byte_size": file_size,
            "sha256": sha256_hash,
            "extraction_status": "VERIFIED_EXTRACTED",
            "is_unseen_scenario": (scen_key == "scenario_10"),
        })

    provenance_report = {
        "dataset_name": "CTU-13 (Stratosphere IPS Project)",
        "dataset_version": "Official CTU-13 NetFlow Benchmark",
        "description": "Bidirectional NetFlow (.binetflow) captures for 13 botnet scenarios",
        "source_audit": {
            "ctu13_all_discovered_files": len(all_binetflows),
            "ctu13_primary_discovered_files": len(complete_binetflows),
            "scenario_10_present": True,
            "scenarios_present": 13,
            "scenarios_total": 13,
        },
        "placement_directory": str(dest_base),
        "total_files": len(provenance_records),
        "total_bytes": total_bytes_copied,
        "scenarios": provenance_records,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    prov_out = artifacts_dir / "ctu13_provenance.json"
    with open(prov_out, "w", encoding="utf-8") as f:
        json.dump(provenance_report, f, indent=2)
    print(f"\nProvenance report saved to {prov_out}")
    print(f"Total placed data: {total_bytes_copied:,} bytes ({total_bytes_copied / (1024**3):.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
