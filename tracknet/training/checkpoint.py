"""Checkpoint save/load helpers.

Checkpoint semantics are stable across training, resume, evaluation and
inference. `last.pt` is the latest full training state; `best.pt` is the full
state with the best validation metric/loss; `model_best.pt` contains only the
model weights for lightweight inference export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tracknet.utils.io import ensure_dir


@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    best_score: float
    metrics: dict[str, float]
    training: dict[str, Any] | None = None


def _model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        return model.module.state_dict()  # type: ignore[no-any-return]
    return model.state_dict()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    state: CheckpointState,
    config: dict[str, Any],
    scaler: Any | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    training = dict(state.training or {})
    if scaler is not None:
        training["amp_scaler_saved"] = True
    payload = {
        "format_version": 1,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_score": state.best_score,
        "metrics": state.metrics,
        "training": training,
        "config": config,
        "model_state_dict": _model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, path)


def save_model_only(path: str | Path, model: nn.Module, config: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"format_version": 1, "config": config, "metrics": metrics or {}, "model_state_dict": _model_state_dict(model)}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        return torch.load(path, map_location=map_location)


def load_model_weights(model: nn.Module, checkpoint: dict[str, Any], strict: bool = True) -> None:
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict")
    model_keys = list(model.state_dict().keys())
    state_keys = list(state.keys())
    if model_keys and state_keys:
        model_has_module = model_keys[0].startswith("module.")
        state_has_module = state_keys[0].startswith("module.")
        if model_has_module and not state_has_module:
            state = {f"module.{k}": v for k, v in state.items()}
        elif not model_has_module and state_has_module:
            state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
