"""Real-Data Evaluation and Benchmarking Engine for TRINETRA / CyberHeal AI.

Executes reproducible, leakage-safe evaluation on open-source network traffic datasets
(primary: CIC-IDS2018, secondary: CTU-13) covering:
1. Dataset ingestion & cryptographic provenance tracking (SHA-256).
2. Data Quality Audit (missing values, infinite values, duplicates, class distribution).
3. Leakage-safe chronological splitting (70% train, 15% val, 15% test).
4. Feature extraction & coverage analysis (78 flow-level, 18 packet-level).
5. Supervised Logistic Regression baseline (BENIGN vs ATTACK).
6. Temporal Network State construction (S_t in R^280).
7. World Model forecasting evaluation (horizons k=1, 3, 5 vs Persistence & Linear baselines).
8. Real unseen attack generalization evaluation (held-out attack categories).
9. Risk estimation, MITRE attack-stage progression mapping, and state-delta attribution.
10. CyberHeal dry-run / simulation integration test.
11. Isolated artifact storage (never overwriting production model weights).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..network_state.future_dataset import FutureStateDatasetBuilder
from ..network_state.state_encoder import NetworkStateEncoder
from ..network_state.state_schema import STATE_FEATURE_DIM
from ..predictive_integration import PredictiveIncidentAdapter
from ..telemetry.feature_schema import CANONICAL_FEATURES
from ..telemetry.packet_feature_schema import PACKET_FEATURE_NAMES
from ..telemetry.split import chronological_split
from ..telemetry.windows import WindowDataset, build_timeseries_windows
from ..world_model.baselines import LinearDynamicsBaseline
from ..world_model.checkpoint import load_world_model_checkpoint
from ..world_model.evaluator import WorldModelEvaluator
from ..world_model.explainability import ForecastExplainabilityEngine, explain_forecast
from ..world_model.risk_estimator import PredictiveRiskEstimator
from ..world_model.stage_mapper import AttackStageMapper
from ..world_model.trigger_gate import (
    ActionSafety,
    AssetCriticality,
    PredictiveTriggerGate,
    TriggerPolicy,
)
from .dataset_adapters import (
    DataQualityReport,
    DatasetAdapter,
    DatasetProvenance,
    get_dataset_adapter,
)
from .logistic_baseline import LogisticBaselineConfig, LogisticRegressionBaseline
from .unseen_attack import (
    UnseenAttackConfig,
    UnseenAttackDataset,
    UnseenAttackEvaluator,
    UnseenExperimentResult,
)

logger = logging.getLogger(__name__)


def discover_dataset(
    data_dir: Path | str, input_format: str = "auto"
) -> tuple[str, list[Path]]:
    """Discover dataset files in the specified directory.

    Args:
        data_dir: Path to search for data files.
        input_format: 'csv', 'pcap', 'pcapng', or 'auto'.

    Returns:
        tuple of (detected_format, list_of_file_paths)
    """
    path_obj = Path(data_dir)
    if not path_obj.exists() or not path_obj.is_dir():
        return "not_found", []

    format_norm = input_format.lower().strip()
    if format_norm == "csv":
        files = sorted(path_obj.glob("*.csv")) + sorted(path_obj.glob("**/*.csv"))
        return ("csv", list(dict.fromkeys(files)))
    elif format_norm in ("pcap", "pcapng"):
        files = (
            sorted(path_obj.glob("*.pcap"))
            + sorted(path_obj.glob("*.pcapng"))
            + sorted(path_obj.glob("**/*.pcap"))
            + sorted(path_obj.glob("**/*.pcapng"))
        )
        return ("pcap", list(dict.fromkeys(files)))
    elif format_norm == "auto":
        csv_files = sorted(path_obj.glob("*.csv")) + sorted(path_obj.glob("**/*.csv"))
        if csv_files:
            return ("csv", list(dict.fromkeys(csv_files)))
        pcap_files = (
            sorted(path_obj.glob("*.pcap"))
            + sorted(path_obj.glob("*.pcapng"))
            + sorted(path_obj.glob("**/*.pcap"))
            + sorted(path_obj.glob("**/*.pcapng"))
        )
        if pcap_files:
            return ("pcap", list(dict.fromkeys(pcap_files)))
        return ("none", [])
    else:
        raise ValueError(
            f"Invalid input_format '{input_format}'. Expected 'csv', 'pcap', 'pcapng', or 'auto'."
        )


def _partition_unseen_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    held_out_category: str,
    known_categories: list[str],
) -> UnseenAttackDataset:
    """Partition DataFrame into leakage-free UnseenAttackDataset."""
    if (
        "features_holder" in df.attrs
        and "labels_holder" in df.attrs
        and "categories_holder" in df.attrs
    ):
        feats = df.attrs["features_holder"].array
        lbls = df.attrs["labels_holder"].array
        cats = df.attrs["categories_holder"].array

        # 1. Benign indices
        b_idx = np.where(lbls == 0)[0]
        max_benign = 100_000
        if len(b_idx) > max_benign:
            step = max(1, len(b_idx) // max_benign)
            b_idx = b_idx[::step]

        # 2. Known attacks indices (preserving every known category)
        known_parts = []
        for cat in known_categories:
            c_idx = np.where(cats == cat)[0]
            if len(c_idx) > 10_000:
                c_idx = c_idx[:: len(c_idx) // 10_000]
            if len(c_idx) > 0:
                known_parts.append(c_idx)
        k_idx = np.concatenate(known_parts) if known_parts else np.empty(0, dtype=int)
        k_idx.sort()

        # 3. Unseen attack indices
        u_idx = np.where(cats == held_out_category)[0]
        if len(u_idx) > 20_000:
            u_idx = u_idx[:: len(u_idx) // 20_000]

        # 4. Split benign chronologically: 70% train, 15% val, 7.5% known test, 7.5% unseen test
        n_b = len(b_idx)
        n_b_train = max(1, int(n_b * 0.70))
        n_b_val = max(1, int(n_b * 0.15))
        n_b_ktest = max(1, int(n_b * 0.075))
        b_train_idx = b_idx[:n_b_train]
        b_val_idx = b_idx[n_b_train : n_b_train + n_b_val]
        b_ktest_idx = b_idx[n_b_train + n_b_val : n_b_train + n_b_val + n_b_ktest]
        b_utest_idx = b_idx[n_b_train + n_b_val + n_b_ktest :]
        if len(b_utest_idx) == 0:
            b_utest_idx = b_ktest_idx

        # 5. Split known attack chronologically: 70% train, 15% val, 15% test
        n_k = len(k_idx)
        n_k_train = max(1, int(n_k * 0.70))
        n_k_val = max(1, int(n_k * 0.15))
        k_train_idx = k_idx[:n_k_train]
        k_val_idx = k_idx[n_k_train : n_k_train + n_k_val]
        k_test_idx = k_idx[n_k_train + n_k_val :]
        if len(k_test_idx) == 0:
            k_test_idx = k_val_idx

        tr_idx = np.concatenate([b_train_idx, k_train_idx])
        val_idx = np.concatenate([b_val_idx, k_val_idx])
        ktest_idx = np.concatenate([b_ktest_idx, k_test_idx])
        utest_idx = np.concatenate([b_utest_idx, u_idx])

        return UnseenAttackDataset(
            X_train=feats[tr_idx],
            y_train=lbls[tr_idx].astype(int),
            categories_train=[str(c) for c in cats[tr_idx]],
            X_val=feats[val_idx],
            y_val=lbls[val_idx].astype(int),
            categories_val=[str(c) for c in cats[val_idx]],
            X_test_known=feats[ktest_idx],
            y_test_known=lbls[ktest_idx].astype(int),
            categories_test_known=[str(c) for c in cats[ktest_idx]],
            X_test_unseen=feats[utest_idx],
            y_test_unseen=lbls[utest_idx].astype(int),
            categories_test_unseen=[str(c) for c in cats[utest_idx]],
        )

    benign_mask = df["label_int"] == 0
    benign_df = df[benign_mask]
    # Sample benign traffic evenly across time if dataset is massive to prevent OOM
    max_benign = 100_000
    if len(benign_df) > max_benign:
        step = max(1, len(benign_df) // max_benign)
        benign_df = benign_df.iloc[::step]

    known_df = df[df["attack_class"].isin(known_categories)]
    unseen_df = df[df["attack_class"] == held_out_category]

    # Benign split: 70% train, 15% val, 7.5% known test, 7.5% unseen test
    n_b = len(benign_df)
    n_b_train = max(1, int(n_b * 0.70))
    n_b_val = max(1, int(n_b * 0.15))
    n_b_ktest = max(1, int(n_b * 0.075))

    b_train = benign_df.iloc[:n_b_train]
    b_val = benign_df.iloc[n_b_train : n_b_train + n_b_val]
    b_ktest = benign_df.iloc[n_b_train + n_b_val : n_b_train + n_b_val + n_b_ktest]
    b_utest = benign_df.iloc[n_b_train + n_b_val + n_b_ktest :]
    if len(b_utest) == 0:
        b_utest = b_ktest

    # Known attack split: 70% train, 15% val, 15% test
    n_k = len(known_df)
    n_k_train = max(1, int(n_k * 0.70))
    n_k_val = max(1, int(n_k * 0.15))

    k_train = known_df.iloc[:n_k_train]
    k_val = known_df.iloc[n_k_train : n_k_train + n_k_val]
    k_test = known_df.iloc[n_k_train + n_k_val :]
    if len(k_test) == 0:
        k_test = k_val

    train_df = pd.concat([b_train, k_train])
    val_df = pd.concat([b_val, k_val])
    known_test_df = pd.concat([b_ktest, k_test])
    unseen_test_df = pd.concat([b_utest, unseen_df])

    return UnseenAttackDataset(
        X_train=train_df[feature_cols].to_numpy(dtype=np.float32),
        y_train=train_df["label_int"].to_numpy(dtype=int),
        categories_train=train_df["attack_class"].tolist(),
        X_val=val_df[feature_cols].to_numpy(dtype=np.float32),
        y_val=val_df["label_int"].to_numpy(dtype=int),
        categories_val=val_df["attack_class"].tolist(),
        X_test_known=known_test_df[feature_cols].to_numpy(dtype=np.float32),
        y_test_known=known_test_df["label_int"].to_numpy(dtype=int),
        categories_test_known=known_test_df["attack_class"].tolist(),
        X_test_unseen=unseen_test_df[feature_cols].to_numpy(dtype=np.float32),
        y_test_unseen=unseen_test_df["label_int"].to_numpy(dtype=int),
        categories_test_unseen=unseen_test_df["attack_class"].tolist(),
    )


class RealDataBenchmarkRunner:
    """Orchestrates end-to-end evaluation of TRINETRA on real labelled network data."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: Path | str,
        output_dir: Path | str | None = None,
        input_format: str = "auto",
        dry_run: bool = True,
        random_state: int = 42,
        is_synthetic: bool = False,
        chunksize: int = 100_000,
    ) -> None:
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.input_format = input_format
        self.dry_run = dry_run
        self.random_state = random_state
        self.is_synthetic = is_synthetic
        self.chunksize = chunksize

        if output_dir is None:
            exp_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("artifacts") / "real_data_benchmark" / dataset_name.lower().replace("-", "_") / exp_id
        else:
            self.output_dir = Path(output_dir)

    def run_benchmark(self, max_rows: int | None = None) -> dict[str, Any]:
        """Execute full benchmark pipeline and return consolidated report."""
        # ── 1. Dataset Discovery ──────────────────────────────────────────────
        detected_format, files = discover_dataset(self.data_dir, self.input_format)
        if detected_format in ("not_found", "none") or not files:
            raise FileNotFoundError(
                f"Real dataset not found in '{self.data_dir}'. No benchmark metrics were generated."
            )

        logger.info(
            "Found %d files for dataset '%s' with format '%s'",
            len(files),
            self.dataset_name,
            detected_format,
        )

        # ── 2. Ingestion & Quality Audit via Adapter ──────────────────────────
        clean_df = None
        try:
            adapter: DatasetAdapter = get_dataset_adapter(self.dataset_name)
            clean_df, quality_report, provenance = adapter.load_and_normalize(
                data_paths=files,
                max_rows=max_rows,
                chunksize=self.chunksize,
            )

            if len(clean_df) < 10:
                raise ValueError(
                    f"Dataset in '{self.data_dir}' contains insufficient usable records ({len(clean_df)}) for benchmarking."
                )

            self.output_dir.mkdir(parents=True, exist_ok=True)

            # ── 3. Chronological Splitting ────────────────────────────────────────
            splits = chronological_split(
                clean_df,
                time_col="flow_start_time",
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )
            train_df, val_df, test_df = splits.train, splits.val, splits.test

            split_summary = {
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "train_time_min": float(train_df["flow_start_time"].min()),
                "train_time_max": float(train_df["flow_start_time"].max()),
                "val_time_min": float(val_df["flow_start_time"].min()),
                "val_time_max": float(val_df["flow_start_time"].max()),
                "test_time_min": float(test_df["flow_start_time"].min()),
                "test_time_max": float(test_df["flow_start_time"].max()),
                "train_time_range": [float(train_df["flow_start_time"].min()), float(train_df["flow_start_time"].max())],
                "val_time_range": [float(val_df["flow_start_time"].min()), float(val_df["flow_start_time"].max())],
                "test_time_range": [float(test_df["flow_start_time"].min()), float(test_df["flow_start_time"].max())],
            }

            # ── 4. Feature Coverage ───────────────────────────────────────────────
            feature_cols = [c for c in CANONICAL_FEATURES if c in clean_df.columns]
            packet_avail = detected_format in ("pcap", "pcapng")
            packet_coverage_msg: str
            if packet_avail:
                packet_coverage_msg = "Packet-level features extracted from PCAP source."
            else:
                packet_coverage_msg = "Packet-level features could not be evaluated from the flow-only input."

            feature_report = {
                "canonical_features_available": len(feature_cols),
                "canonical_features_total": 78,
                "canonical_feature_dim": 78,
                "feature_coverage_ratio": len(feature_cols) / 78.0,
                "packet_features_available": packet_avail,
                "packet_level_status": packet_coverage_msg,
                "packet_features_total": len(PACKET_FEATURE_NAMES),
            }

            # ── 5. Logistic Regression Classification Baseline ────────────────────
            if "features_holder" in clean_df.attrs and "labels_holder" in clean_df.attrs:
                all_feats = clean_df.attrs["features_holder"].array
                all_lbls = clean_df.attrs["labels_holder"].array
                train_n = len(train_df)
                val_n = len(val_df)
                test_n = len(test_df)
                X_train = all_feats[:train_n]
                y_train = all_lbls[:train_n].astype(int)
                X_val = all_feats[train_n : train_n + val_n]
                y_val = all_lbls[train_n : train_n + val_n].astype(int)
                X_test = all_feats[train_n + val_n : train_n + val_n + test_n]
                y_test = all_lbls[train_n + val_n : train_n + val_n + test_n].astype(int)
            else:
                X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
                y_train = train_df["label_int"].to_numpy(dtype=int)
                X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
                y_val = val_df["label_int"].to_numpy(dtype=int)
                X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
                y_test = test_df["label_int"].to_numpy(dtype=int)

            log_cfg = LogisticBaselineConfig(
                feature_representation="flow_features",
                random_state=self.random_state,
            )
            lr_baseline = LogisticRegressionBaseline(config=log_cfg)
            lr_baseline.fit(X_train, y_train)

            # Tune threshold on validation split strictly without test leakage
            if (y_val == 1).sum() > 0 and len(np.unique(y_val)) > 1:
                try:
                    val_probas = lr_baseline.predict_proba(X_val)
                    pos_idx = 1 if 1 in lr_baseline.classes_ else len(lr_baseline.classes_) - 1
                    val_pos = val_probas[:, pos_idx]
                    best_thresh = 0.50
                    best_f1 = -1.0
                    for th in np.linspace(0.1, 0.9, 17):
                        preds = (val_pos >= th).astype(int)
                        tp = np.sum((y_val == 1) & (preds == 1))
                        fp = np.sum((y_val == 0) & (preds == 1))
                        fn = np.sum((y_val == 1) & (preds == 0))
                        denom = 2 * tp + fp + fn
                        f1 = (2 * tp) / denom if denom > 0 else 0.0
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thresh = float(th)
                    lr_baseline.config.decision_threshold = best_thresh
                except Exception:
                    pass

            lr_metrics = lr_baseline.evaluate(X_test, y_test)
            logistic_report = lr_metrics.to_dict()

            # ── 6. Temporal Network State & World Model Forecasting ───────────────
            world_model_report: dict[str, Any] = {}
            # Dynamic window size T ensuring at least 6 windows are available for k=5 horizon
            T = min(20, max(2, len(test_df) // 4)) if len(test_df) >= 8 else 2
            stride = 1 if len(test_df) < 50 else max(1, T // 2)
            if len(test_df) > 50_000:
                stride = max(stride, len(test_df) // 2000)

            if len(test_df) >= T + 5:
                if "features_holder" in clean_df.attrs:
                    test_len = len(test_df)
                    all_win_X = []
                    all_win_y = []
                    for s_idx in range(0, test_len - T + 1, stride):
                        all_win_X.append(X_test[s_idx : s_idx + T])
                        sub_y = y_test[s_idx : s_idx + T]
                        all_win_y.append(int(np.argmax(np.bincount(sub_y, minlength=1))))
                    if all_win_X:
                        win_ds = WindowDataset(
                            X=np.stack(all_win_X, axis=0).astype(np.float32),
                            y=np.array(all_win_y, dtype=np.int64),
                            feature_names=feature_cols,
                            group_ids=["cic_ids_2018"] * len(all_win_X),
                        )
                    else:
                        win_ds = WindowDataset(
                            X=np.zeros((0, T, len(feature_cols)), dtype=np.float32),
                            y=np.zeros(0, dtype=np.int64),
                            feature_names=feature_cols,
                            group_ids=[],
                        )
                else:
                    win_ds = build_timeseries_windows(
                        test_df,
                        feature_cols=feature_cols,
                        window_size=T,
                        stride=stride,
                    )
                if len(win_ds.X) >= 6:
                    encoder = NetworkStateEncoder()
                    state_batch = encoder.encode_batch(win_ds.X)

                    pair_builder = FutureStateDatasetBuilder(horizons=[1, 3, 5])
                    multi_hor_ds = pair_builder.build_future_pairs(state_batch)

                    # Fit Linear Dynamics Baselines on train/val states if available
                    linear_baselines: dict[int, LinearDynamicsBaseline] = {}
                    if len(train_df) >= T + 5:
                        train_stride = max(1, T // 2)
                        if len(train_df) > 50_000:
                            train_stride = max(train_stride, len(train_df) // 2000)
                        if "features_holder" in clean_df.attrs:
                            train_len = len(train_df)
                            tr_win_X = []
                            tr_win_y = []
                            for s_idx in range(0, train_len - T + 1, train_stride):
                                tr_win_X.append(X_train[s_idx : s_idx + T])
                                sub_y = y_train[s_idx : s_idx + T]
                                tr_win_y.append(int(np.argmax(np.bincount(sub_y, minlength=1))))
                            if tr_win_X:
                                win_tr = WindowDataset(
                                    X=np.stack(tr_win_X, axis=0).astype(np.float32),
                                    y=np.array(tr_win_y, dtype=np.int64),
                                    feature_names=feature_cols,
                                    group_ids=["cic_ids_2018"] * len(tr_win_X),
                                )
                            else:
                                win_tr = WindowDataset(
                                    X=np.zeros((0, T, len(feature_cols)), dtype=np.float32),
                                    y=np.zeros(0, dtype=np.int64),
                                    feature_names=feature_cols,
                                    group_ids=[],
                                )
                        else:
                            win_tr = build_timeseries_windows(
                                train_df,
                                feature_cols=feature_cols,
                                window_size=T,
                                stride=train_stride,
                            )
                        if len(win_tr.X) >= 6:
                            state_tr = encoder.encode_batch(win_tr.X)
                            ds_tr = pair_builder.build_future_pairs(state_tr)
                            for h in [1, 3, 5]:
                                if h in ds_tr.batches and ds_tr.batches[h].num_pairs >= 2:
                                    lin_m = LinearDynamicsBaseline()
                                    lin_m.fit(ds_tr.batches[h])
                                    linear_baselines[h] = lin_m

                    # Load current production World Model strictly without retraining
                    wm_model, wm_cfg, _ = load_world_model_checkpoint("artifacts/world_model")
                    evaluator = WorldModelEvaluator(wm_model)
                    world_model_report = evaluator.evaluate_multi_horizon(
                        multi_hor_ds,
                        linear_baselines=linear_baselines if linear_baselines else None,
                    )
                else:
                    world_model_report = {"status": "INSUFFICIENT_TEST_WINDOWS", "num_windows": len(win_ds.X)}
            else:
                world_model_report = {"status": "INSUFFICIENT_TEST_ROWS", "test_rows": len(test_df)}

            # ── 7. Real Unseen Attack Generalization Evaluation ───────────────────
            unseen_report: dict[str, Any] = {}
            attack_categories = [c for c in clean_df["attack_class"].unique() if c.upper() != "BENIGN"]
            if len(attack_categories) >= 2:
                held_out = attack_categories[-1]
                known_cats = [c for c in attack_categories if c != held_out]
                try:
                    unseen_cfg = UnseenAttackConfig(
                        held_out_categories=[held_out],
                        known_categories=known_cats,
                        experiment_id="real_data_unseen_01",
                    )
                    unseen_evaluator = UnseenAttackEvaluator(
                        config=unseen_cfg,
                        output_dir=self.output_dir / "unseen_attack",
                    )
                    unseen_ds = _partition_unseen_dataset(
                        df=clean_df,
                        feature_cols=feature_cols,
                        held_out_category=held_out,
                        known_categories=known_cats,
                    )
                    unseen_res = unseen_evaluator.evaluate_track_a_logistic(unseen_ds)
                    unseen_report = {
                        "held_out_category": held_out,
                        "known_categories": known_cats,
                        **unseen_res,
                    }
                except Exception as e:
                    unseen_report = {"status": "EVALUATION_ERROR", "error": str(e)}
            else:
                unseen_report = {
                    "status": "UNAVAILABLE",
                    "reason": "Real unseen attack evaluation requires at least 2 distinct attack categories.",
                    "available_attack_categories": attack_categories,
                }

            # ── 8. Risk, Attack Stage & Explainability Evaluation ─────────────────
            risk_and_stage_report: dict[str, Any] = {}
            sample_state = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
            drift_state = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
            # Apply synthetic shift on flags and macro features to test pipeline response
            drift_state[150:160] = 5.0
            drift_state[270:280] = 3.0

            risk_engine = PredictiveRiskEstimator()
            risk_est = risk_engine.estimate_horizon_risk(sample_state, drift_state, horizon=3, delta_t=1.5)
            stage_engine = AttackStageMapper()
            stage_res = stage_engine.map_stage(sample_state, drift_state, horizon=3)
            explanation = explain_forecast(sample_state, drift_state, horizon=3, top_k=5)
            exp_dict = explanation.to_dict()

            risk_and_stage_report = {
                "sample_risk_score": float(risk_est.risk_score),
                "sample_infiltration_score": float(risk_est.infiltration_score),
                "predicted_attack_stage": stage_res.stage.value,
                "stage_confidence": stage_res.confidence,
                "top_features": exp_dict.get("top_features", []),
                "group_attributions": exp_dict.get("group_attributions", []),
                "summary_text": explanation.summary_text,
            }

            # ── 9. CyberHeal Dry-Run Integration Test ─────────────────────────────
            cyberheal_summary: dict[str, Any] = {}
            trigger_gate = PredictiveTriggerGate(policy=TriggerPolicy(risk_auto_threshold=0.60))
            decision = trigger_gate.evaluate(
                risk_score=risk_est,
                predicted_stage=stage_res,
                explanation=explanation,
                asset_criticality=AssetCriticality.LOW,
                action_safety=ActionSafety.SAFE,
            )

            adapter_bridge = PredictiveIncidentAdapter()
            incident_res = adapter_bridge.process_trigger_decision(decision, target_ip="192.168.1.100", explanation=explanation)

            cyberheal_summary = {
                "trigger_decision": decision.decision.value,
                "handoff_status": incident_res.status.value,
                "incident_id": incident_res.incident_id,
                "execution_mode": "SIMULATION / DRY_RUN",
                "real_host_remediation_performed": False,
                "reasons": decision.reasons,
            }

            # ── 10. Persist Isolated Artifacts ────────────────────────────────────
            self._save_json("provenance.json", provenance.to_dict())
            self._save_json("data_quality_report.json", quality_report.to_dict())
            self._save_json("split_summary.json", split_summary)
            self._save_json("feature_report.json", feature_report)
            self._save_json("logistic_baseline_metrics.json", logistic_report)
            self._save_json("world_model_forecasting_metrics.json", world_model_report)
            self._save_json("unseen_attack_metrics.json", unseen_report)
            self._save_json("risk_and_stage_metrics.json", risk_and_stage_report)
            self._save_json("cyberheal_simulation_summary.json", cyberheal_summary)

            execution_type = "SYNTHETIC SOFTWARE VALIDATION" if self.is_synthetic else "REAL_DATA_EVALUATION"
            manifest = {
                "dataset_name": self.dataset_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(self.output_dir),
                "production_model_weights_untouched": True,
                "remediation_mode": "DRY_RUN",
                "execution_type": execution_type,
            }
            self._save_json("benchmark_manifest.json", manifest)

            # Generate Human-Readable Markdown Summary
            self._generate_markdown_summary(
                provenance=provenance,
                quality_report=quality_report,
                logistic_report=logistic_report,
                world_model_report=world_model_report,
                unseen_report=unseen_report,
            )

            return {
                "dataset_name": self.dataset_name,
                "execution_type": execution_type,
                "provenance": provenance.to_dict(),
                "quality_report": quality_report.to_dict(),
                "logistic_baseline": logistic_report,
                "world_model_forecasting": world_model_report,
                "unseen_attack": unseen_report,
                "cyberheal_dry_run": cyberheal_summary,
                "manifest": manifest,
            }
        finally:
            if clean_df is not None:
                cleanup_fn = clean_df.attrs.get("cleanup_mmap")
                del clean_df
                if cleanup_fn is not None:
                    try:
                        cleanup_fn()
                    except Exception:
                        pass

    def _save_json(self, filename: str, data: Any) -> None:
        p = self.output_dir / filename

        def _json_default(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return str(obj)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def _generate_markdown_summary(
        self,
        provenance: DatasetProvenance,
        quality_report: DataQualityReport,
        logistic_report: dict[str, Any],
        world_model_report: dict[str, Any],
        unseen_report: dict[str, Any],
    ) -> None:
        title_prefix = "SYNTHETIC SOFTWARE VALIDATION" if self.is_synthetic else "Real Data Benchmark Summary"
        lines: list[str] = [
            f"# {title_prefix}: {self.dataset_name}",
            f"**Execution Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## 1. Provenance & Ingestion",
            f"- **Dataset:** {provenance.dataset_name}",
            f"- **Files Processed:** {len(provenance.source_files)}",
            f"- **Usable Records:** {provenance.row_count:,}",
            f"- **Total Bytes:** {provenance.total_bytes:,} bytes",
            "",
            "## 2. Data Quality Audit",
            f"- **Benign Samples:** {quality_report.benign_count:,}",
            f"- **Attack Samples:** {quality_report.attack_count:,} ({quality_report.attack_ratio * 100:.2f}%)",
            f"- **Attack Categories:** {quality_report.attack_categories}",
            f"- **Missing Values Cleaned:** {quality_report.missing_value_count:,}",
            f"- **Infinite Values Cleaned:** {quality_report.infinite_value_count:,}",
            "",
            "## 3. Logistic Regression Baseline",
            f"- **Precision:** {logistic_report.get('precision', 0.0):.4f}",
            f"- **Recall:** {logistic_report.get('recall', 0.0):.4f}",
            f"- **F1 Score:** {logistic_report.get('f1_score', 0.0):.4f}",
            f"- **False Positive Rate:** {logistic_report.get('false_positive_rate', 0.0):.4f}",
            f"- **ROC-AUC:** {logistic_report.get('roc_auc', 0.0) if logistic_report.get('roc_auc') is not None else 'N/A'}",
            "",
            "## 4. World Model Forecasting (Multi-Horizon)",
        ]

        if "results_by_horizon" in world_model_report:
            lines.append("| Horizon | Persistence MAE | World Model MAE | Linear MAE | Improvement over Persistence |")
            lines.append("|---|---|---|---|---|")
            for h_key, data in world_model_report["results_by_horizon"].items():
                p_mae = data.get("persistence_baseline", {}).get("mae", 0.0)
                w_mae = data.get("world_model", {}).get("mae", 0.0)
                lin_mae = data.get("linear_baseline", {}).get("mae")
                lin_str = f"{lin_mae:.4f}" if lin_mae is not None else "N/A"
                imp = data.get("improvement_over_persistence_pct", 0.0)
                lines.append(f"| {h_key} | {p_mae:.4f} | {w_mae:.4f} | {lin_str} | {imp:+.2f}% |")
        else:
            lines.append(f"- Status: {world_model_report.get('status', 'EVALUATED')}")

        lines.extend([
            "",
            "## 5. Unseen Attack Generalization",
            f"- **Status:** {unseen_report.get('status', 'COMPLETE')}",
        ])
        if "known_attack_metrics" in unseen_report:
            lines.append(f"- **Held-Out Category:** {unseen_report.get('held_out_category')}")
            lines.append(f"- **Known F1:** {unseen_report['known_attack_metrics'].get('f1_score', 0.0):.4f}")
            lines.append(f"- **Unseen F1:** {unseen_report['unseen_attack_metrics'].get('f1_score', 0.0):.4f}")
            lines.append(f"- **Delta F1:** {unseen_report['performance_degradation'].get('delta_f1', 0.0):+.4f}")

        summary_file = self.output_dir / "BENCHMARK_SUMMARY.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
