"""K-Step Future-State Dataset Construction for CyberHeal AI.

Constructs mathematically valid, leakage-free prediction datasets:
    (S_t, S_{t+k}) for k in {1, ..., K}
where S_t in R^280 is the current Network State and S_{t+k} in R^280 is the
true future Network State k temporal observation windows ahead.

Enforces:
1. Exact temporal indexing and non-negative elapsed time (delta_t = t_{t+k} - t_t >= 0).
2. Strict scenario/session isolation (zero cross-scenario pairing).
3. Strict chronological partition isolation (zero cross-split leakage between train, val, and test).
4. Explicit reporting of dropped incomplete boundary pairs.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .state_encoder import NetworkStateBatch
from .state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FutureStatePairBatch:
    """Production container for single-horizon future state pairs (S_t -> S_{t+k}).

    Attributes:
        current_states: Array of shape [N_pairs, 280] representing S_t.
        target_states: Array of shape [N_pairs, 280] representing S_{t+k}.
        current_timestamps: Array of POSIX timestamps for S_t (length N_pairs).
        target_timestamps: Array of POSIX timestamps for S_{t+k} (length N_pairs).
        delta_times: Elapsed wall-clock time in seconds (t_{t+k} - t_t) (length N_pairs).
        horizon: Discrete temporal window step k.
        source_ids: Scenario or host identifier strings (length N_pairs).
        schema_version: State schema version string.
    """

    current_states: np.ndarray
    target_states: np.ndarray
    current_timestamps: np.ndarray
    target_timestamps: np.ndarray
    delta_times: np.ndarray
    horizon: int
    source_ids: list[str]
    schema_version: str
    current_labels: np.ndarray | None = None
    target_labels: np.ndarray | None = None
    current_categories: list[str] | None = None
    target_categories: list[str] | None = None

    @property
    def num_pairs(self) -> int:
        return int(self.current_states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.current_states.shape[-1])

    @property
    def sequence_length(self) -> int:
        return int(self.current_states.shape[1]) if self.current_states.ndim == 3 else 1

    @property
    def is_temporal_sequence(self) -> bool:
        return self.current_states.ndim == 3 and self.current_states.shape[1] > 1


@dataclass(slots=True)
class MultiHorizonFutureDataset:
    """Multi-horizon container holding pair batches across k in {1, 3, 5, ...}."""

    batches: dict[int, FutureStatePairBatch]
    horizons: list[int]
    total_states_input: int
    dropped_tail_pairs: dict[int, int]
    schema_version: str

    @property
    def total_usable_pairs(self) -> int:
        return sum(b.num_pairs for b in self.batches.values())

    def get_horizon(self, k: int) -> FutureStatePairBatch:
        """Retrieve the pair batch for a specific horizon k."""
        if k not in self.batches:
            raise KeyError(f"Horizon k={k} not found. Available horizons: {self.horizons}")
        return self.batches[k]

    def to_multi_horizon_tensor(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Extract multi-horizon tensor for samples having targets across ALL configured horizons.

        Returns:
            X: [N_common, 280] current states
            Y: [N_common, K, 280] future states across all horizons
            timestamps: [N_common] current timestamps
            delta_times: [N_common, K] elapsed times
            source_ids: list of length N_common
        """
        if not self.batches:
            raise ValueError("Dataset contains no horizon batches")

        # The maximum horizon determines the minimum tail index
        max_k = max(self.horizons)
        ref_batch = self.batches[self.horizons[0]]
        N_ref = ref_batch.num_pairs

        # Find common indices available across all horizons
        min_pairs = min(b.num_pairs for b in self.batches.values())
        if min_pairs == 0:
            empty_X = np.empty((0, STATE_FEATURE_DIM), dtype=np.float32)
            empty_Y = np.empty((0, len(self.horizons), STATE_FEATURE_DIM), dtype=np.float32)
            return empty_X, empty_Y, np.empty(0), np.empty((0, len(self.horizons))), []

        # Common current states (taken from the most constrained batch)
        X = self.batches[max_k].current_states
        N_common = len(X)
        timestamps = self.batches[max_k].current_timestamps
        source_ids = self.batches[max_k].source_ids

        Y_list: list[np.ndarray] = []
        dt_list: list[np.ndarray] = []

        for k in self.horizons:
            b = self.batches[k]
            Y_list.append(b.target_states[:N_common])
            dt_list.append(b.delta_times[:N_common])

        Y = np.stack(Y_list, axis=1)  # [N_common, K, 280]
        delta_times = np.stack(dt_list, axis=1)  # [N_common, K]
        return X, Y, timestamps, delta_times, source_ids


class FutureStateDatasetBuilder:
    """Constructs leakage-safe future state pairs across configurable horizons."""

    def __init__(
        self,
        horizons: list[int] | None = None,
        sequence_length: int = 1,
        schema_version: str = STATE_SCHEMA_VERSION,
    ) -> None:
        self.horizons: list[int] = sorted(horizons) if horizons is not None else [1, 3, 5]
        self.sequence_length: int = max(1, int(sequence_length))
        self.schema_version = schema_version
        if any(k < 1 for k in self.horizons):
            raise ValueError(f"All horizons k must be >= 1, got {self.horizons}")

    def build_future_pairs(
        self,
        batch: NetworkStateBatch,
        sequence_length: int | None = None,
    ) -> MultiHorizonFutureDataset:
        """Construct future pairs from an ordered NetworkStateBatch.

        Pairs are formed strictly within each unique source_id / scenario sequence.
        Incomplete tail states where t+k exceeds the sequence are dropped and counted.
        When sequence_length T > 1, inputs are [N, T, 280] temporal sequences ending at S_t.

        Args:
            batch: NetworkStateBatch containing states in R^280 and timestamps.
            sequence_length: History length T (defaults to self.sequence_length).

        Returns:
            MultiHorizonFutureDataset containing pair batches for each horizon k.
        """
        N = batch.num_states
        T = max(1, int(sequence_length if sequence_length is not None else self.sequence_length))

        if N == 0:
            empty_curr_shape = (0, T, STATE_FEATURE_DIM) if T > 1 else (0, STATE_FEATURE_DIM)
            empty_batches = {
                k: FutureStatePairBatch(
                    current_states=np.empty(empty_curr_shape, dtype=np.float32),
                    target_states=np.empty((0, STATE_FEATURE_DIM), dtype=np.float32),
                    current_timestamps=np.empty(0, dtype=float),
                    target_timestamps=np.empty(0, dtype=float),
                    delta_times=np.empty(0, dtype=float),
                    horizon=k,
                    source_ids=[],
                    schema_version=self.schema_version,
                )
                for k in self.horizons
            }
            return MultiHorizonFutureDataset(
                batches=empty_batches,
                horizons=self.horizons,
                total_states_input=0,
                dropped_tail_pairs={k: 0 for k in self.horizons},
                schema_version=self.schema_version,
            )

        states = batch.states
        timestamps = batch.timestamps
        sources = batch.source_ids

        # Group indices by scenario/source to prevent cross-scenario pairing
        scenario_groups: dict[str, list[int]] = collections.defaultdict(list)
        for idx, src in enumerate(sources):
            scenario_groups[src].append(idx)

        # Sort indices within each scenario chronologically by timestamp
        for src in scenario_groups:
            scenario_groups[src].sort(key=lambda i: timestamps[i])

        horizon_batches: dict[int, FutureStatePairBatch] = {}
        dropped_counts: dict[int, int] = {}

        for k in self.horizons:
            curr_list: list[np.ndarray] = []
            target_list: list[np.ndarray] = []
            curr_ts_list: list[float] = []
            target_ts_list: list[float] = []
            dt_list: list[float] = []
            pair_sources: list[str] = []
            curr_labels: list[int] = []
            target_labels: list[int] = []
            curr_cats: list[str] = []
            target_cats: list[str] = []
            dropped = 0

            for src, indices in scenario_groups.items():
                seq_len = len(indices)
                start_pos = T - 1 if T > 1 else 0
                # Incomplete head states that cannot form history length T
                dropped += start_pos

                for pos in range(start_pos, seq_len):
                    if pos + k < seq_len:
                        curr_idx = indices[pos]
                        targ_idx = indices[pos + k]
                        t_curr = float(timestamps[curr_idx])
                        t_targ = float(timestamps[targ_idx])
                        dt = t_targ - t_curr

                        if dt < 0:
                            logger.warning(
                                "Negative delta_t observed in scenario %s: t_curr=%s, t_targ=%s",
                                src,
                                t_curr,
                                t_targ,
                            )

                        if T > 1:
                            seq_indices = [indices[pos - T + 1 + step] for step in range(T)]
                            curr_list.append(states[seq_indices])
                        else:
                            curr_list.append(states[curr_idx])

                        target_list.append(states[targ_idx])
                        curr_ts_list.append(t_curr)
                        target_ts_list.append(t_targ)
                        dt_list.append(dt)
                        pair_sources.append(src)

                        if batch.labels is not None:
                            curr_labels.append(int(batch.labels[curr_idx]))
                            target_labels.append(int(batch.labels[targ_idx]))
                        if batch.categories is not None:
                            curr_cats.append(str(batch.categories[curr_idx]))
                            target_cats.append(str(batch.categories[targ_idx]))
                    else:
                        dropped += 1

            if curr_list:
                curr_arr = np.stack(curr_list, axis=0)
            else:
                curr_arr = (
                    np.empty((0, T, STATE_FEATURE_DIM), dtype=np.float32)
                    if T > 1
                    else np.empty((0, STATE_FEATURE_DIM), dtype=np.float32)
                )

            targ_arr = (
                np.stack(target_list, axis=0)
                if target_list
                else np.empty((0, STATE_FEATURE_DIM), dtype=np.float32)
            )

            horizon_batches[k] = FutureStatePairBatch(
                current_states=curr_arr,
                target_states=targ_arr,
                current_timestamps=np.array(curr_ts_list, dtype=float),
                target_timestamps=np.array(target_ts_list, dtype=float),
                delta_times=np.array(dt_list, dtype=float),
                horizon=k,
                source_ids=pair_sources,
                schema_version=self.schema_version,
                current_labels=np.array(curr_labels, dtype=np.int64) if len(curr_labels) > 0 else None,
                target_labels=np.array(target_labels, dtype=np.int64) if len(target_labels) > 0 else None,
                current_categories=curr_cats if len(curr_cats) > 0 else None,
                target_categories=target_cats if len(target_cats) > 0 else None,
            )
            dropped_counts[k] = dropped

        return MultiHorizonFutureDataset(
            batches=horizon_batches,
            horizons=self.horizons,
            total_states_input=N,
            dropped_tail_pairs=dropped_counts,
            schema_version=self.schema_version,
        )

    def build_partitioned_future_datasets(
        self,
        train_states: NetworkStateBatch,
        val_states: NetworkStateBatch,
        test_states: NetworkStateBatch,
        sequence_length: int | None = None,
    ) -> dict[str, MultiHorizonFutureDataset]:
        """Build future datasets for train, val, and test with strict boundary isolation.

        Guarantees:
            1. Zero training pairs contain validation target states.
            2. Zero validation pairs contain test target states.
            3. Incomplete boundary pairs at the tail of train/val/test are dropped cleanly.
        """
        train_ds = self.build_future_pairs(train_states, sequence_length=sequence_length)
        val_ds = self.build_future_pairs(val_states, sequence_length=sequence_length)
        test_ds = self.build_future_pairs(test_states, sequence_length=sequence_length)

        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        }
