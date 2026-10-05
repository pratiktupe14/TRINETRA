"""TRINETRA RLM Offline Learning Pipeline & Candidate Model Evaluation Gate.

Provides controlled, offline training and validation loops:
    Learning Records -> Dataset Construction -> Candidate Training -> Evaluation Gate -> Explicit Release

SAFETY INVARIANTS:
1. Live incidents NEVER trigger training or modify weights.
2. Training produces candidate models in isolated subdirectories (candidates/).
3. Production model (artifacts/world_model/best_model.pt) is NEVER overwritten during training.
4. Candidates must pass the Evaluation Gate (candidate MAE <= current model MAE) to be eligible for release.
5. Production release requires explicit human/operator approval.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .network_state.future_dataset import FutureStatePairBatch, MultiHorizonFutureDataset
from .network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from .rlm_feedback import LearningRecord, RLMStore
from .world_model.candidates import GRUDynamicsWorldModel
from .world_model.checkpoint import load_world_model_checkpoint, save_world_model_checkpoint
from .world_model.evaluator import WorldModelEvaluator
from .world_model.model import StateResidualDynamicsModel
from .world_model.trainer import WorldModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluationReport:
    """Evaluation gate report comparing candidate model against current production model."""
    candidate_id: str
    created_at: str
    num_learning_records: int
    current_model_mae: float
    candidate_model_mae: float
    mae_improvement_pct: float
    evaluation_decision: str  # "ACCEPT", "REJECT", "INCONCLUSIVE"
    decision_reasons: List[str]
    candidate_artifact_dir: str
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class RLMLearningPipeline:
    """Controlled offline learning pipeline managing candidate model training and evaluation gates."""

    def __init__(
        self,
        artifacts_dir: Path | str = "artifacts/world_model",
        rlm_records_path: Path | str = "artifacts/rlm/learning_records.jsonl",
        candidates_dir: Optional[Path | str] = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.rlm_records_path = Path(rlm_records_path)
        self.candidates_dir = Path(candidates_dir) if candidates_dir is not None else self.artifacts_dir / "candidates"
        self.archive_dir = self.artifacts_dir / "archive"

    def load_and_validate_records(self) -> List[LearningRecord]:
        """Ingest all RLM records from the offline store and validate them."""
        store = RLMStore(self.rlm_records_path)
        return store.read_records()

    def build_candidate_dataset(
        self,
        records: List[LearningRecord],
        horizons: List[int] = [1, 3, 5],
        synthetic_supplement: bool = True,
        supplement_samples: int = 60,
        seed: int = 42,
    ) -> Tuple[MultiHorizonFutureDataset, MultiHorizonFutureDataset, MultiHorizonFutureDataset]:
        """Construct train, val, and test datasets from RLM records and synthetic baseline dynamics.
        
        Args:
            records: Validated RLM records.
            horizons: Horizons to forecast.
            synthetic_supplement: Whether to include baseline dynamics.
            supplement_samples: Number of baseline trajectory pairs to generate.
            seed: Random seed for deterministic reproducibility.
            
        Returns:
            Tuple of (train_ds, val_ds, test_ds).
        """
        np.random.seed(seed)

        # 1. Extract valid (current_state, observed_state) pairs from learning records
        extracted_curr: List[np.ndarray] = []
        extracted_targ: List[np.ndarray] = []

        for rec in records:
            if rec.state_references.current_state and rec.state_references.observed_state:
                if len(rec.state_references.current_state) == STATE_FEATURE_DIM and len(rec.state_references.observed_state) == STATE_FEATURE_DIM:
                    extracted_curr.append(np.array(rec.state_references.current_state, dtype=np.float32))
                    extracted_targ.append(np.array(rec.state_references.observed_state, dtype=np.float32))

        # 2. Build multi-horizon batches
        # If real incident pairs are insufficient, supplement deterministically
        n_real = len(extracted_curr)
        n_total = n_real + (supplement_samples if synthetic_supplement else 0)
        if n_total < 10:
            n_total = 30  # Minimum for split

        curr_states = np.random.randn(n_total, STATE_FEATURE_DIM).astype(np.float32)
        if n_real > 0:
            curr_states[:n_real] = np.vstack(extracted_curr)

        batches: Dict[int, FutureStatePairBatch] = {}
        for k in horizons:
            dts = (k * 2.5 + np.random.uniform(0.1, 0.5, size=n_total)).astype(np.float32)
            cur_ts = np.arange(n_total, dtype=float) * 10.0
            targ_ts = cur_ts + dts
            decay = 0.95 ** k
            drift = 0.05 * k * np.sin(curr_states)
            targ_states = (decay * curr_states + drift).astype(np.float32)
            if n_real > 0 and k == 1:
                targ_states[:n_real] = np.vstack(extracted_targ)

            batches[k] = FutureStatePairBatch(
                current_states=curr_states,
                target_states=targ_states,
                current_timestamps=cur_ts,
                target_timestamps=targ_ts,
                delta_times=dts,
                horizon=k,
                source_ids=["rlm-session"] * n_total,
                schema_version=STATE_SCHEMA_VERSION,
            )

        full_ds = MultiHorizonFutureDataset(
            batches=batches,
            horizons=horizons,
            total_states_input=n_total + max(horizons),
            dropped_tail_pairs={k: k for k in horizons},
            schema_version=STATE_SCHEMA_VERSION,
        )

        # 3. Chronological 70 / 15 / 15 split
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)

        train_batches: Dict[int, FutureStatePairBatch] = {}
        val_batches: Dict[int, FutureStatePairBatch] = {}
        test_batches: Dict[int, FutureStatePairBatch] = {}

        for k in horizons:
            b = full_ds.batches[k]
            train_batches[k] = FutureStatePairBatch(
                current_states=b.current_states[:n_train],
                target_states=b.target_states[:n_train],
                current_timestamps=b.current_timestamps[:n_train],
                target_timestamps=b.target_timestamps[:n_train],
                delta_times=b.delta_times[:n_train],
                horizon=k,
                source_ids=b.source_ids[:n_train],
                schema_version=STATE_SCHEMA_VERSION,
            )
            val_batches[k] = FutureStatePairBatch(
                current_states=b.current_states[n_train:n_train + n_val],
                target_states=b.target_states[n_train:n_train + n_val],
                current_timestamps=b.current_timestamps[n_train:n_train + n_val],
                target_timestamps=b.target_timestamps[n_train:n_train + n_val],
                delta_times=b.delta_times[n_train:n_train + n_val],
                horizon=k,
                source_ids=b.source_ids[n_train:n_train + n_val],
                schema_version=STATE_SCHEMA_VERSION,
            )
            test_batches[k] = FutureStatePairBatch(
                current_states=b.current_states[n_train + n_val:],
                target_states=b.target_states[n_train + n_val:],
                current_timestamps=b.current_timestamps[n_train + n_val:],
                target_timestamps=b.target_timestamps[n_train + n_val:],
                delta_times=b.delta_times[n_train + n_val:],
                horizon=k,
                source_ids=b.source_ids[n_train + n_val:],
                schema_version=STATE_SCHEMA_VERSION,
            )

        train_ds = MultiHorizonFutureDataset(train_batches, horizons, n_train, {k: 0 for k in horizons}, STATE_SCHEMA_VERSION)
        val_ds = MultiHorizonFutureDataset(val_batches, horizons, n_val, {k: 0 for k in horizons}, STATE_SCHEMA_VERSION)
        test_ds = MultiHorizonFutureDataset(test_batches, horizons, n_total - n_train - n_val, {k: 0 for k in horizons}, STATE_SCHEMA_VERSION)

        return train_ds, val_ds, test_ds

    def train_candidate(
        self,
        candidate_id: str,
        train_ds: MultiHorizonFutureDataset,
        val_ds: MultiHorizonFutureDataset,
        horizons: List[int] = [1, 3, 5],
        max_epochs: int = 5,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        seed: int = 42,
    ) -> Path:
        """Train a candidate model and save artifacts strictly into candidates/<candidate_id>/.
        
        CRITICAL: Never overwrites artifacts/world_model/best_model.pt.
        """
        dest_dir = self.candidates_dir / candidate_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Initialize fresh architecture (Phase 10: GRUDynamicsWorldModel candidate)
        candidate_model = GRUDynamicsWorldModel(
            state_dim=STATE_FEATURE_DIM,
            hidden_dim=128,
            num_layers=2,
            dropout=0.02,
            max_horizon=16,
            bidirectional=False,
        )

        trainer = WorldModelTrainer(
            model=candidate_model,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=3,
            seed=seed,
        )

        trainer.fit(train_ds, val_ds)

        # Save candidate artifacts
        config = {
            "model_type": "GRUDynamicsWorldModel",
            "state_dim": STATE_FEATURE_DIM,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_blocks": 2,
            "dropout": 0.02,
            "max_horizon": 16,
            "horizons": horizons,
            "bidirectional": False,
        }
        manifest_extra = {
            "model_type": "GRUDynamicsWorldModel",
            "candidate_id": candidate_id,
            "parent_model": "artifacts/world_model/best_model.pt",
            "best_val_loss": trainer.best_val_loss,
            "epochs_trained": len(trainer.history),
            "seed": seed,
            "batch_size": batch_size,
        }
        save_world_model_checkpoint(
            model=candidate_model,
            dest_dir=dest_dir,
            config=config,
            manifest_extra=manifest_extra,
        )

        # Save training history
        with open(dest_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(trainer.history, f, indent=2)

        logger.info("Candidate %s saved to %s", candidate_id, dest_dir)
        return dest_dir

    def evaluate_candidate(
        self,
        candidate_id: str,
        test_ds: MultiHorizonFutureDataset,
        num_records: int = 0,
    ) -> CandidateEvaluationReport:
        """Evaluate candidate model against current production model using WorldModelEvaluator.
        
        Evaluation Gate:
        - If candidate MAE <= current model MAE: ACCEPT
        - If candidate MAE > current model MAE: REJECT
        - If insufficient data: INCONCLUSIVE
        """
        cand_dir = self.candidates_dir / candidate_id
        cand_weights = cand_dir / "best_model.pt"
        if not cand_weights.exists():
            raise FileNotFoundError(f"Candidate model weights not found in {cand_dir}")

        cand_model, _, _ = load_world_model_checkpoint(cand_dir)
        cand_evaluator = WorldModelEvaluator(cand_model)
        cand_report = cand_evaluator.evaluate_multi_horizon(test_ds)

        # Compute average candidate MAE across horizons
        cand_maes = [cand_report["results_by_horizon"][f"k_{k}"]["world_model"]["mae"] for k in test_ds.horizons]
        candidate_mae = float(np.mean(cand_maes))

        # Check for current production model
        prod_weights = self.artifacts_dir / "best_model.pt"
        current_mae = 0.0
        if prod_weights.exists():
            prod_model, _, _ = load_world_model_checkpoint(self.artifacts_dir)
            prod_evaluator = WorldModelEvaluator(prod_model)
            prod_report = prod_evaluator.evaluate_multi_horizon(test_ds)
            prod_maes = [prod_report["results_by_horizon"][f"k_{k}"]["world_model"]["mae"] for k in test_ds.horizons]
            current_mae = float(np.mean(prod_maes))
        else:
            # No baseline production model yet
            current_mae = candidate_mae

        eps = 1e-7
        imp_pct = float((1.0 - (candidate_mae / (current_mae + eps))) * 100.0)

        # Evaluation Gate Decision
        reasons: List[str] = []
        if test_ds.total_usable_pairs < 5:
            decision = "INCONCLUSIVE"
            reasons.append(f"Insufficient test pairs ({test_ds.total_usable_pairs} < 5) to validate candidate.")
        elif candidate_mae <= current_mae or abs(candidate_mae - current_mae) < 1e-4:
            decision = "ACCEPT"
            reasons.append(
                f"Candidate MAE ({candidate_mae:.6f}) meets or beats current production MAE ({current_mae:.6f}) "
                f"by {imp_pct:+.2f}%."
            )
        else:
            decision = "REJECT"
            reasons.append(
                f"Regression detected: Candidate MAE ({candidate_mae:.6f}) is worse than current production MAE ({current_mae:.6f}) "
                f"by {imp_pct:+.2f}%."
            )

        report = CandidateEvaluationReport(
            candidate_id=candidate_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            num_learning_records=num_records,
            current_model_mae=current_mae,
            candidate_model_mae=candidate_mae,
            mae_improvement_pct=imp_pct,
            evaluation_decision=decision,
            decision_reasons=reasons,
            candidate_artifact_dir=str(cand_dir),
            evaluation_metrics=cand_report,
        )

        with open(cand_dir / "candidate_evaluation_report.json", "w", encoding="utf-8") as f:
            f.write(report.to_json())

        return report

    def promote_candidate(
        self,
        candidate_id: str,
        operator_approved: bool = False,
        force: bool = False,
    ) -> Path:
        """Promote a candidate model to production.
        
        CRITICAL SAFETY RULES:
        1. operator_approved must be True (explicit human approval).
        2. Candidate evaluation decision must be "ACCEPT" (unless force=True with explicit override).
        3. Archives existing production model to archive/<timestamp>/ before replacement.
        """
        if not operator_approved:
            raise PermissionError("Explicit operator approval is required to release candidate model to production.")

        cand_dir = self.candidates_dir / candidate_id
        eval_report_file = cand_dir / "candidate_evaluation_report.json"
        if not eval_report_file.exists():
            raise FileNotFoundError(f"Candidate {candidate_id} has not been evaluated through the Evaluation Gate.")

        with open(eval_report_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        decision = eval_data.get("evaluation_decision", "INCONCLUSIVE")
        if decision != "ACCEPT" and not force:
            raise ValueError(f"Cannot promote candidate {candidate_id}: Evaluation Gate decision was '{decision}'.")

        # Archive existing production model if present
        prod_weights = self.artifacts_dir / "best_model.pt"
        if prod_weights.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_dir = self.archive_dir / f"world_model_backup_{ts}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f_name in ["best_model.pt", "model_config.json", "model_manifest.json"]:
                p_file = self.artifacts_dir / f_name
                if p_file.exists():
                    shutil.copy2(p_file, backup_dir / f_name)
            logger.info("Archived previous production model to %s", backup_dir)

        # Copy candidate artifacts to production
        shutil.copy2(cand_dir / "best_model.pt", self.artifacts_dir / "best_model.pt")
        shutil.copy2(cand_dir / "model_config.json", self.artifacts_dir / "model_config.json")
        if (cand_dir / "model_manifest.json").exists():
            shutil.copy2(cand_dir / "model_manifest.json", self.artifacts_dir / "model_manifest.json")

        logger.info("Candidate %s successfully released to production at %s", candidate_id, self.artifacts_dir)
        return self.artifacts_dir
