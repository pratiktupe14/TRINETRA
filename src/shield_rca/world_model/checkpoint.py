"""Model Checkpointing and Manifest Serialization for CyberHeal World Model."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..network_state.state_schema import STATE_FEATURE_DIM, STATE_SCHEMA_VERSION
from .model import StateResidualDynamicsModel

logger = logging.getLogger(__name__)


def save_world_model_checkpoint(
    model: StateResidualDynamicsModel,
    dest_dir: Path | str,
    config: dict[str, Any],
    manifest_extra: dict[str, Any] | None = None,
) -> Path:
    """Save model weights, configuration, and manifest.

    Args:
        model: Trained StateResidualDynamicsModel.
        dest_dir: Directory to save artifacts.
        config: Architecture configuration dictionary.
        manifest_extra: Optional extra fields (e.g. best_val_loss, epochs, seed).

    Returns:
        Path to the saved directory.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    weights_path = dest / "best_model.pt"
    config_path = dest / "model_config.json"
    manifest_path = dest / "model_manifest.json"

    # 1. Weights
    torch.save(model.state_dict(), weights_path)

    # 2. Config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 3. Manifest
    model_type = getattr(model, "__class__", type(model)).__name__
    manifest = {
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_dim": STATE_FEATURE_DIM,
        "weights_file": "best_model.pt",
        "config_file": "model_config.json",
    }
    if manifest_extra:
        manifest.update(manifest_extra)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved World Model checkpoint to %s", dest)
    return dest


def load_world_model_checkpoint(
    src_dir: Path | str,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load model weights, config, and manifest from directory.

    Returns:
        tuple of (model in eval mode, config, manifest).
    """
    src = Path(src_dir)
    weights_path = src / "best_model.pt"
    config_path = src / "model_config.json"
    manifest_path = src / "model_manifest.json"

    if not weights_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Missing World Model files in {src}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    model_type = manifest.get("model_type", config.get("model_type", "StateResidualDynamicsModel"))
    if model_type == "GRUDynamicsWorldModel":
        from .candidates import GRUDynamicsWorldModel

        model = GRUDynamicsWorldModel(
            state_dim=config.get("state_dim", STATE_FEATURE_DIM),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", config.get("num_blocks", 2)),
            dropout=config.get("dropout", 0.05),
            max_horizon=config.get("max_horizon", 16),
            bidirectional=config.get("bidirectional", False),
        )
    elif model_type == "TemporalContextResidualModel":
        from .candidates import TemporalContextResidualModel

        model = TemporalContextResidualModel(
            state_dim=config.get("state_dim", STATE_FEATURE_DIM),
            hidden_dim=config.get("hidden_dim", 128),
            num_blocks=config.get("num_blocks", 2),
            dropout=config.get("dropout", 0.05),
            max_horizon=config.get("max_horizon", 16),
        )
    else:
        model = StateResidualDynamicsModel(
            state_dim=config.get("state_dim", STATE_FEATURE_DIM),
            hidden_dim=config.get("hidden_dim", 256),
            num_blocks=config.get("num_blocks", 2),
            dropout=config.get("dropout", 0.05),
            max_horizon=config.get("max_horizon", 16),
        )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Loaded World Model checkpoint from %s", src)
    return model, config, manifest
