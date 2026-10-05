"""Real-Data Temporal World-Model Training and Evaluation Pipeline for CyberHeal AI.

Orchestrates the ingestion, leakage-free chronological splitting, state encoding,
temporal sequence preparation (shape [N, T, 280]), training, and rigorous evaluation
of the GRU Dynamics World Model on real CSE-CIC-IDS2018 data.

Enforces:
1. Exact reuse of real CIC-IDS2018 data loading via CicIds2018Adapter.
2. Chronological split (70% train, 15% val, 15% test) without temporal leakage.
3. State preprocessor fitted EXCLUSIVELY on training data.
4. Real temporal sequence generation [N, T, 280] with T > 1 ending at S_t.
5. Target corresponds to true future state S_{t+k} across horizons k in {1, 3, 5}.
6. Simultaneous, fair evaluation on identical test pairs across:
   - Persistence Baseline
   - Linear Dynamics Baseline
   - Current Production World Model (artifacts/world_model/best_model.pt)
   - GRU Candidate World Model
7. Complete isolation of candidate artifacts under artifacts/world_model/candidates/<candidate_id>/
8. Strict protection of production model checkpoint (zero automatic promotion).
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..evaluation.dataset_adapters import (
    CicIds2018Adapter,
    DataQualityReport,
    DatasetProvenance,
    get_dataset_adapter,
)
from ..evaluation.real_data_benchmark import discover_dataset
from ..network_state.future_dataset import (
    FutureStateDatasetBuilder,
    FutureStatePairBatch,
    MultiHorizonFutureDataset,
)
from ..network_state.preprocessor import NetworkStatePreprocessor
from ..network_state.state_encoder import NetworkStateBatch, NetworkStateEncoder
from ..network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from ..telemetry.feature_schema import CANONICAL_FEATURES
from ..telemetry.split import chronological_split
from ..telemetry.windows import WindowDataset, build_timeseries_windows
from .baselines import LinearDynamicsBaseline, PersistenceBaseline
from .candidates import GRUDynamicsWorldModel
from .checkpoint import load_world_model_checkpoint, save_world_model_checkpoint
from .evaluator import WorldModelEvaluator
from .loss import WorldModelLoss
from .model import StateResidualDynamicsModel
from .trainer import WorldModelTrainer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RealDataTemporalDatasets:
    """Container for processed real-data temporal datasets and provenance."""

    train_ds: MultiHorizonFutureDataset
    val_ds: MultiHorizonFutureDataset
    test_ds: MultiHorizonFutureDataset
    preprocessor: NetworkStatePreprocessor
    split_summary: dict[str, Any]
    quality_report: dict[str, Any]
    provenance: dict[str, Any]
    sequence_length: int
    horizons: list[int]
    flow_window_size: int


def load_cic_ids2018_temporal_datasets(
    data_dir: Path | str = "data/cic_ids2018/processed",
    max_rows: int = 100_000,
    sequence_length: int = 5,
    horizons: list[int] | None = None,
    scaler_type: str = "standard",
    chunksize: int = 100_000,
) -> RealDataTemporalDatasets:
    """Load real CIC-IDS2018 data and build leakage-safe temporal datasets.

    Args:
        data_dir: Directory containing CIC-IDS2018 CSV files.
        max_rows: Maximum records to ingest (default: 100,000 benchmark limit).
        sequence_length: Temporal history length T for GRU (must be > 1).
        horizons: List of forecast step horizons (default: [1, 3, 5]).
        scaler_type: 'standard' or 'robust'.
        chunksize: Streaming chunk size for CSV ingestion.

    Returns:
        RealDataTemporalDatasets container with train, val, test datasets.
    """
    if sequence_length <= 1:
        raise ValueError(
            f"Temporal sequence length T must be > 1 for real GRU training, got {sequence_length}"
        )

    horizons_list = sorted(horizons) if horizons is not None else [1, 3, 5]
    data_path = Path(data_dir)

    # 1. Discover dataset files
    detected_format, files = discover_dataset(data_path, "csv")
    if not files:
        # Fallback to parent if processed was given
        parent_dir = data_path.parent
        detected_format, files = discover_dataset(parent_dir, "csv")

    if not files:
        raise FileNotFoundError(
            f"No CIC-IDS2018 CSV files found in {data_path}. Ensure data is present."
        )

    logger.info("Ingesting real CIC-IDS2018 dataset from %d files (max_rows=%d)", len(files), max_rows)
    adapter = CicIds2018Adapter()
    clean_df, quality_report, provenance = adapter.load_and_normalize(
        data_paths=files,
        max_rows=max_rows,
        chunksize=chunksize,
    )

    if len(clean_df) < 50:
        raise ValueError(f"Insufficient clean records ({len(clean_df)}) for benchmark.")

    # 2. Chronological Split (70/15/15)
    splits = chronological_split(
        clean_df,
        time_col="flow_start_time",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    train_df, val_df, test_df = splits.train, splits.val, splits.test

    # Strict leakage assertion
    train_max_t = float(train_df["flow_start_time"].max())
    val_min_t = float(val_df["flow_start_time"].min())
    val_max_t = float(val_df["flow_start_time"].max())
    test_min_t = float(test_df["flow_start_time"].min())

    assert train_max_t <= val_min_t, (
        f"DATA LEAKAGE: train max time ({train_max_t}) > val min time ({val_min_t})"
    )
    assert val_max_t <= test_min_t, (
        f"DATA LEAKAGE: val max time ({val_max_t}) > test min time ({test_min_t})"
    )

    split_summary = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_time_range": [float(train_df["flow_start_time"].min()), train_max_t],
        "val_time_range": [val_min_t, val_max_t],
        "test_time_range": [test_min_t, float(test_df["flow_start_time"].max())],
    }

    # 3. Feature extraction and observation windowing
    feature_cols = [c for c in CANONICAL_FEATURES if c in clean_df.columns]
    T_flow = min(20, max(2, len(test_df) // 4)) if len(test_df) >= 8 else 2

    # Strides matching RealDataBenchmarkRunner
    train_stride = max(1, T_flow // 2)
    if len(train_df) > 50_000:
        train_stride = max(train_stride, len(train_df) // 2000)

    val_stride = max(1, T_flow // 2)
    test_stride = max(1, T_flow // 2)
    if len(test_df) > 50_000:
        test_stride = max(test_stride, len(test_df) // 2000)

    # Build observation windows
    if "features_holder" in clean_df.attrs:
        all_feats = clean_df.attrs["features_holder"].array
        all_lbls = clean_df.attrs["labels_holder"].array
        all_cats = clean_df.attrs["categories_holder"].array if "categories_holder" in clean_df.attrs else None
        train_n = len(train_df)
        val_n = len(val_df)
        test_n = len(test_df)

        def _make_holder_windows(feats_sub, lbls_sub, cats_sub, stride_val):
            sub_len = len(feats_sub)
            win_X = []
            win_y = []
            win_cats = []
            for s_idx in range(0, sub_len - T_flow + 1, stride_val):
                win_X.append(feats_sub[s_idx : s_idx + T_flow])
                sub_y = lbls_sub[s_idx : s_idx + T_flow]
                dom_y = int(np.argmax(np.bincount(sub_y.astype(int), minlength=1)))
                win_y.append(dom_y)
                if cats_sub is not None:
                    sub_c = cats_sub[s_idx : s_idx + T_flow]
                    c_counts = collections.Counter(sub_c)
                    win_cats.append(str(c_counts.most_common(1)[0][0]))
                else:
                    win_cats.append("Attack" if dom_y == 1 else "Benign")
            if win_X:
                dataset = WindowDataset(
                    X=np.stack(win_X, axis=0).astype(np.float32),
                    y=np.array(win_y, dtype=np.int64),
                    feature_names=feature_cols,
                    group_ids=["cic_ids2018"] * len(win_X),
                )
                return dataset, win_cats
            return WindowDataset(
                X=np.zeros((0, T_flow, len(feature_cols)), dtype=np.float32),
                y=np.zeros(0, dtype=np.int64),
                feature_names=feature_cols,
                group_ids=[],
            ), []

        win_train, tr_cats = _make_holder_windows(all_feats[:train_n], all_lbls[:train_n], all_cats[:train_n] if all_cats is not None else None, train_stride)
        win_val, val_cats = _make_holder_windows(
            all_feats[train_n : train_n + val_n], all_lbls[train_n : train_n + val_n], all_cats[train_n : train_n + val_n] if all_cats is not None else None, val_stride
        )
        win_test, te_cats = _make_holder_windows(
            all_feats[train_n + val_n : train_n + val_n + test_n],
            all_lbls[train_n + val_n : train_n + val_n + test_n],
            all_cats[train_n + val_n : train_n + val_n + test_n] if all_cats is not None else None,
            test_stride,
        )
    else:
        win_train = build_timeseries_windows(train_df, feature_cols, window_size=T_flow, stride=train_stride)
        win_val = build_timeseries_windows(val_df, feature_cols, window_size=T_flow, stride=val_stride)
        win_test = build_timeseries_windows(test_df, feature_cols, window_size=T_flow, stride=test_stride)
        tr_cats = ["Attack" if y == 1 else "Benign" for y in win_train.y]
        val_cats = ["Attack" if y == 1 else "Benign" for y in win_val.y]
        te_cats = ["Attack" if y == 1 else "Benign" for y in win_test.y]

    # 4. Deterministic Network State Encoding to R^280
    encoder = NetworkStateEncoder()
    train_states = encoder.encode_batch(win_train.X, labels=win_train.y, categories=tr_cats)
    val_states = encoder.encode_batch(win_val.X, labels=win_val.y, categories=val_cats)
    test_states = encoder.encode_batch(win_test.X, labels=win_test.y, categories=te_cats)

    # 5. Leakage-free state preprocessor (FITTED EXCLUSIVELY ON TRAIN)
    preprocessor = NetworkStatePreprocessor(scaler_type=scaler_type)
    preprocessor.fit(train_states.states)

    norm_train_states = NetworkStateBatch(
        states=preprocessor.transform(train_states.states),
        timestamps=train_states.timestamps,
        window_starts=train_states.window_starts,
        window_ends=train_states.window_ends,
        source_ids=train_states.source_ids,
        feature_names=train_states.feature_names,
        schema_version=train_states.schema_version,
        labels=train_states.labels,
        categories=train_states.categories,
    )
    norm_val_states = NetworkStateBatch(
        states=preprocessor.transform(val_states.states),
        timestamps=val_states.timestamps,
        window_starts=val_states.window_starts,
        window_ends=val_states.window_ends,
        source_ids=val_states.source_ids,
        feature_names=val_states.feature_names,
        schema_version=val_states.schema_version,
        labels=val_states.labels,
        categories=val_states.categories,
    )
    norm_test_states = NetworkStateBatch(
        states=preprocessor.transform(test_states.states),
        timestamps=test_states.timestamps,
        window_starts=test_states.window_starts,
        window_ends=test_states.window_ends,
        source_ids=test_states.source_ids,
        feature_names=test_states.feature_names,
        schema_version=test_states.schema_version,
        labels=test_states.labels,
        categories=test_states.categories,
    )

    # 6. Future temporal state sequence pair generation [N, T, 280]
    builder = FutureStateDatasetBuilder(horizons=horizons_list, sequence_length=sequence_length)
    train_ds = builder.build_future_pairs(norm_train_states, sequence_length=sequence_length)
    val_ds = builder.build_future_pairs(norm_val_states, sequence_length=sequence_length)
    test_ds = builder.build_future_pairs(norm_test_states, sequence_length=sequence_length)

    # Verify temporal sequence properties
    for k in horizons_list:
        b_tr = train_ds.batches[k]
        if b_tr.num_pairs > 0:
            assert b_tr.current_states.ndim == 3, f"Expected 3D current_states, got {b_tr.current_states.ndim}"
            assert b_tr.current_states.shape[1] == sequence_length, (
                f"Expected T={sequence_length}, got {b_tr.current_states.shape[1]}"
            )
            assert b_tr.current_states.shape[2] == STATE_FEATURE_DIM
            assert b_tr.target_states.shape[1] == STATE_FEATURE_DIM
            assert np.all(b_tr.delta_times >= 0), "Negative delta_times detected in future pairs"

    return RealDataTemporalDatasets(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        preprocessor=preprocessor,
        split_summary=split_summary,
        quality_report=quality_report.to_dict(),
        provenance=provenance.to_dict(),
        sequence_length=sequence_length,
        horizons=horizons_list,
        flow_window_size=T_flow,
    )


def train_and_evaluate_gru_candidate(
    datasets: RealDataTemporalDatasets,
    candidate_id: str,
    candidates_dir: Path | str = "artifacts/world_model/candidates",
    production_dir: Path | str = "artifacts/world_model",
    max_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Train GRUDynamicsWorldModel on real temporal sequences and evaluate against production.

    Guarantees:
    - artifacts/world_model/best_model.pt is NEVER modified.
    - Candidate artifacts are isolated in artifacts/world_model/candidates/<candidate_id>/.
    - Exact same test pairs evaluated across Persistence, Linear, Production WM, and GRU Candidate.
    - Evaluation gate emits CANDIDATE_PASS or REJECT.
    """
    prod_dir = Path(production_dir)
    prod_weights = prod_dir / "best_model.pt"

    # Compute production weights hash before training to ensure immutability
    sha256_prod_before = hashlib.sha256(prod_weights.read_bytes()).hexdigest() if prod_weights.exists() else None

    cand_dir = Path(candidates_dir) / candidate_id
    cand_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate GRU Candidate with temporal sequence validation
    logger.info("Instantiating GRUDynamicsWorldModel (hidden_dim=128, num_layers=2, T=%d)", datasets.sequence_length)
    candidate_model = GRUDynamicsWorldModel(
        state_dim=STATE_FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        dropout=0.02,
        max_horizon=16,
        bidirectional=False,
        require_temporal_sequence=True,
    )

    # 2. Train candidate with deterministic seeding and early stopping
    trainer = WorldModelTrainer(
        model=candidate_model,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        delta_huber=1.0,
        seed=seed,
    )

    logger.info("Training GRU candidate '%s' on %d train pairs...", candidate_id, datasets.train_ds.total_usable_pairs)
    history = trainer.fit(datasets.train_ds, datasets.val_ds)

    # 3. Save candidate checkpoint & metadata in isolated directory
    cand_config = {
        "model_type": "GRUDynamicsWorldModel",
        "state_dim": STATE_FEATURE_DIM,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_blocks": 2,
        "dropout": 0.02,
        "max_horizon": 16,
        "horizons": datasets.horizons,
        "bidirectional": False,
        "sequence_length": datasets.sequence_length,
    }
    cand_manifest = {
        "model_type": "GRUDynamicsWorldModel",
        "candidate_id": candidate_id,
        "parent_model": str(prod_weights),
        "best_val_loss": trainer.best_val_loss,
        "epochs_trained": len(trainer.history),
        "seed": seed,
        "batch_size": batch_size,
        "sequence_length": datasets.sequence_length,
        "dataset_source": "CSE-CIC-IDS2018",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_world_model_checkpoint(
        model=candidate_model,
        dest_dir=cand_dir,
        config=cand_config,
        manifest_extra=cand_manifest,
    )

    with open(cand_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(cand_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(datasets.split_summary, f, indent=2)

    datasets.preprocessor.save(cand_dir)

    # 4. Verify production weights were completely untouched
    if prod_weights.exists():
        sha256_prod_after = hashlib.sha256(prod_weights.read_bytes()).hexdigest()
        assert sha256_prod_after == sha256_prod_before, (
            f"FATAL SECURITY VIOLATION: Production weights were modified! Before: {sha256_prod_before}, After: {sha256_prod_after}"
        )

    # 5. Multi-Model Evaluation on the EXACT SAME test dataset
    logger.info("Evaluating all models on identical test pairs (total=%d)...", datasets.test_ds.total_usable_pairs)
    pers_baseline = PersistenceBaseline()

    # Fit linear dynamics baselines on train pairs
    linear_baselines: dict[int, LinearDynamicsBaseline] = {}
    for k in datasets.horizons:
        b_tr = datasets.train_ds.batches[k]
        if b_tr.num_pairs >= 2:
            lin = LinearDynamicsBaseline()
            lin.fit(b_tr)
            linear_baselines[k] = lin

    # Load production model
    prod_model: StateResidualDynamicsModel | None = None
    if prod_weights.exists():
        prod_model, _, _ = load_world_model_checkpoint(prod_dir)

    # Load trained candidate model
    cand_model, _, _ = load_world_model_checkpoint(cand_dir)

    comparison_results: dict[str, Any] = {}
    loss_criterion = WorldModelLoss(delta=1.0)

    for k in datasets.horizons:
        k_key = f"k_{k}"
        batch = datasets.test_ds.batches[k]
        N_pairs = batch.num_pairs

        # 1. Persistence Baseline
        pers_eval = pers_baseline.evaluate(batch)

        # 2. Linear Baseline
        lin_eval = linear_baselines[k].evaluate(batch) if k in linear_baselines else None

        # 3. Production Model
        prod_eval: dict[str, float] | None = None
        if prod_model is not None:
            prod_model.eval()
            with torch.no_grad():
                X_te = torch.from_numpy(batch.current_states).float()
                k_te = torch.full((N_pairs,), k, dtype=torch.long)
                dt_te = torch.from_numpy(batch.delta_times).float()
                Y_te = torch.from_numpy(batch.target_states).float()
                pred_prod = prod_model(X_te, k_te, dt_te)
                prod_eval = loss_criterion.compute_metrics(pred_prod, Y_te)

        # 4. GRU Candidate Model
        cand_model.eval()
        with torch.no_grad():
            X_te = torch.from_numpy(batch.current_states).float()
            k_te = torch.full((N_pairs,), k, dtype=torch.long)
            dt_te = torch.from_numpy(batch.delta_times).float()
            Y_te = torch.from_numpy(batch.target_states).float()
            pred_cand = cand_model(X_te, k_te, dt_te)
            gru_eval = loss_criterion.compute_metrics(pred_cand, Y_te)

        # Deltas
        eps = 1e-7
        pers_mae = pers_eval["mae"]
        lin_mae = lin_eval["mae"] if lin_eval else None
        prod_mae = prod_eval["mae"] if prod_eval else pers_mae
        gru_mae = gru_eval["mae"]

        imp_over_pers = (1.0 - (gru_mae / (pers_mae + eps))) * 100.0
        imp_over_prod = (1.0 - (gru_mae / (prod_mae + eps))) * 100.0 if prod_eval else 0.0

        comparison_results[k_key] = {
            "horizon": k,
            "num_test_pairs": N_pairs,
            "persistence": pers_eval,
            "linear": lin_eval,
            "production": prod_eval,
            "gru_candidate": gru_eval,
            "improvement_over_persistence_pct": imp_over_pers,
            "improvement_over_production_pct": imp_over_prod,
        }

    # 6. Evaluation Gate Decision
    gru_maes = [comparison_results[f"k_{k}"]["gru_candidate"]["mae"] for k in datasets.horizons]
    prod_maes = [
        comparison_results[f"k_{k}"]["production"]["mae"]
        if comparison_results[f"k_{k}"]["production"]
        else comparison_results[f"k_{k}"]["persistence"]["mae"]
        for k in datasets.horizons
    ]
    pers_maes = [comparison_results[f"k_{k}"]["persistence"]["mae"] for k in datasets.horizons]

    mean_gru_mae = float(np.mean(gru_maes))
    mean_prod_mae = float(np.mean(prod_maes))
    mean_pers_mae = float(np.mean(pers_maes))

    overall_imp_prod = (1.0 - (mean_gru_mae / (mean_prod_mae + 1e-7))) * 100.0
    overall_imp_pers = (1.0 - (mean_gru_mae / (mean_pers_mae + 1e-7))) * 100.0

    decision_reasons = []
    if mean_gru_mae < mean_prod_mae:
        decision = "CANDIDATE_PASS"
        decision_reasons.append(
            f"GRU Candidate mean MAE ({mean_gru_mae:.4f}) outperforms Production mean MAE ({mean_prod_mae:.4f}) "
            f"by {overall_imp_prod:+.2f}%."
        )
    else:
        decision = "REJECT"
        decision_reasons.append(
            f"GRU Candidate mean MAE ({mean_gru_mae:.4f}) does not outperform Production mean MAE ({mean_prod_mae:.4f}). "
            f"Delta: {overall_imp_prod:+.2f}%."
        )

    decision_reasons.append(
        "Candidate weights remain isolated under artifacts/world_model/candidates/. "
        "Explicit operator approval is strictly required before any promotion."
    )

    evaluation_report = {
        "candidate_id": candidate_id,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_test_pairs": datasets.test_ds.total_usable_pairs,
        "sequence_length": datasets.sequence_length,
        "horizons": datasets.horizons,
        "mean_persistence_mae": mean_pers_mae,
        "mean_production_mae": mean_prod_mae,
        "mean_gru_mae": mean_gru_mae,
        "improvement_over_production_pct": overall_imp_prod,
        "improvement_over_persistence_pct": overall_imp_pers,
        "evaluation_decision": decision,
        "decision_reasons": decision_reasons,
        "results_by_horizon": comparison_results,
        "production_weights_sha256": sha256_prod_before,
        "production_weights_modified": False,
    }

    with open(cand_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    return evaluation_report
