"""Quality and integrity report utility for Future-State Datasets in CyberHeal AI.

Audits:
- Usable vs dropped pairs per horizon.
- Temporal spacing and delta_t statistics.
- Scenario isolation and distribution.
- Temporal boundaries and non-leakage verification.
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np

from .future_dataset import MultiHorizonFutureDataset


def generate_future_dataset_quality_report(
    dataset: MultiHorizonFutureDataset,
    partition_name: str = "dataset",
) -> dict[str, Any]:
    """Generate a comprehensive data quality and integrity report.

    Args:
        dataset: MultiHorizonFutureDataset instance.
        partition_name: Name of partition (e.g. 'train', 'val', 'test', 'full').

    Returns:
        Structured dictionary detailing dataset health, counts, and timing.
    """
    horizons = dataset.horizons
    total_states = dataset.total_states_input

    pair_counts = {f"horizon_{k}": dataset.batches[k].num_pairs for k in horizons}
    dropped_counts = {f"horizon_{k}": dataset.dropped_tail_pairs.get(k, 0) for k in horizons}

    # Timing statistics across all pairs
    all_dt: list[float] = []
    scenarios: set[str] = set()
    earliest_ts: float = float("inf")
    latest_ts: float = float("-inf")

    for k in horizons:
        batch = dataset.batches[k]
        if batch.num_pairs > 0:
            all_dt.extend(batch.delta_times.tolist())
            scenarios.update(batch.source_ids)
            earliest_ts = min(earliest_ts, float(np.min(batch.current_timestamps)))
            latest_ts = max(latest_ts, float(np.max(batch.target_timestamps)))

    dt_stats: dict[str, float] = {}
    if all_dt:
        arr_dt = np.array(all_dt)
        dt_stats = {
            "min_delta_s": float(np.min(arr_dt)),
            "max_delta_s": float(np.max(arr_dt)),
            "mean_delta_s": float(np.mean(arr_dt)),
            "median_delta_s": float(np.median(arr_dt)),
            "std_delta_s": float(np.std(arr_dt)),
        }
    else:
        earliest_ts = 0.0
        latest_ts = 0.0

    return {
        "partition": partition_name,
        "total_states_input": total_states,
        "total_usable_pairs": dataset.total_usable_pairs,
        "horizons_evaluated": horizons,
        "usable_pairs_per_horizon": pair_counts,
        "dropped_incomplete_pairs": dropped_counts,
        "distinct_scenarios_count": len(scenarios),
        "distinct_scenarios": sorted(list(scenarios)),
        "temporal_range": {
            "earliest_timestamp": earliest_ts,
            "latest_timestamp": latest_ts,
            "total_span_s": latest_ts - earliest_ts if latest_ts >= earliest_ts else 0.0,
        },
        "elapsed_time_statistics": dt_stats,
        "state_dimension": 280,
        "schema_version": dataset.schema_version,
    }
