"""Training Loop for the CyberHeal World Model.

Trains StateResidualDynamicsModel using Huber loss, AdamW, early stopping,
and deterministic seeding without data leakage.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..network_state.future_dataset import MultiHorizonFutureDataset
from .checkpoint import save_world_model_checkpoint
from .loss import WorldModelLoss
from .model import StateResidualDynamicsModel

logger = logging.getLogger(__name__)


def _future_dataset_to_tensors(
    dataset: MultiHorizonFutureDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine all horizon batches into flattened PyTorch tensors."""
    curr_list: list[np.ndarray] = []
    targ_list: list[np.ndarray] = []
    k_list: list[int] = []
    dt_list: list[np.ndarray] = []

    for k in dataset.horizons:
        b = dataset.batches[k]
        if b.num_pairs > 0:
            curr_list.append(b.current_states)
            targ_list.append(b.target_states)
            k_list.extend([k] * b.num_pairs)
            dt_list.append(b.delta_times)

    if not curr_list:
        raise ValueError("MultiHorizonFutureDataset contains zero usable pairs")

    X = torch.from_numpy(np.vstack(curr_list)).float()
    Y = torch.from_numpy(np.vstack(targ_list)).float()
    k_arr = torch.tensor(k_list, dtype=torch.long)
    dt_arr = torch.from_numpy(np.concatenate(dt_list)).float()

    return X, k_arr, dt_arr, Y


class WorldModelTrainer:
    """Trains StateResidualDynamicsModel with early stopping and metrics logging."""

    def __init__(
        self,
        model: StateResidualDynamicsModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 30,
        patience: int = 6,
        delta_huber: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.seed = seed

        # Deterministic seeding
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model.to(self.device)
        self.criterion = WorldModelLoss(delta=delta_huber)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

        self.best_val_loss: float = float("inf")
        self.best_state_dict: dict[str, Any] | None = None
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        train_dataset: MultiHorizonFutureDataset,
        val_dataset: MultiHorizonFutureDataset,
    ) -> list[dict[str, float]]:
        """Train model until convergence or max_epochs reached."""
        X_tr, k_tr, dt_tr, Y_tr = _future_dataset_to_tensors(train_dataset)
        X_va, k_va, dt_va, Y_va = _future_dataset_to_tensors(val_dataset)

        train_ds = TensorDataset(X_tr, k_tr, dt_tr, Y_tr)
        val_ds = TensorDataset(X_va, k_va, dt_va, Y_va)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=len(train_ds) > self.batch_size
        )
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        epochs_no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            # 1. Training pass
            self.model.train()
            train_loss_accum = 0.0
            n_train_batches = 0

            for bx, bk, bdt, by in train_loader:
                bx = bx.to(self.device)
                bk = bk.to(self.device)
                bdt = bdt.to(self.device)
                by = by.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(bx, bk, bdt)
                loss = self.criterion(pred, by)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss_accum += float(loss.item())
                n_train_batches += 1

            train_loss = train_loss_accum / max(1, n_train_batches)

            # 2. Validation pass
            self.model.eval()
            val_loss_accum = 0.0
            val_mae_accum = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for bx, bk, bdt, by in val_loader:
                    bx = bx.to(self.device)
                    bk = bk.to(self.device)
                    bdt = bdt.to(self.device)
                    by = by.to(self.device)

                    pred = self.model(bx, bk, bdt)
                    loss = self.criterion(pred, by)
                    metrics = self.criterion.compute_metrics(pred, by)

                    val_loss_accum += float(loss.item())
                    val_mae_accum += float(metrics["mae"])
                    n_val_batches += 1

            val_loss = val_loss_accum / max(1, n_val_batches)
            val_mae = val_mae_accum / max(1, n_val_batches)

            self.scheduler.step(val_loss)

            epoch_record = {
                "epoch": epoch,
                "train_huber": train_loss,
                "val_huber": val_loss,
                "val_mae": val_mae,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history.append(epoch_record)

            # Checkpoint best validation
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    def save(self, dest_dir: Path | str, horizons: list[int]) -> Path:
        """Save best model checkpoint and manifest."""
        config = {
            "state_dim": getattr(self.model, "state_dim", 280),
            "hidden_dim": getattr(self.model, "hidden_dim", 128),
            "num_blocks": getattr(self.model, "num_blocks", getattr(self.model, "num_layers", 2)),
            "max_horizon": getattr(self.model.conditioner, "max_horizon", 16) if hasattr(self.model, "conditioner") else 16,
            "horizons": horizons,
        }
        manifest_extra = {
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.history),
            "seed": self.seed,
            "batch_size": self.batch_size,
        }
        return save_world_model_checkpoint(
            model=self.model,
            dest_dir=dest_dir,
            config=config,
            manifest_extra=manifest_extra,
        )
