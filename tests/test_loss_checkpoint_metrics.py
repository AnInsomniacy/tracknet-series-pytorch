from __future__ import annotations

from pathlib import Path

import torch

from tracknet.models import build_model
from tracknet.training.checkpoint import CheckpointState, load_checkpoint, load_model_weights, save_checkpoint
from tracknet.training.losses import build_loss
from tracknet.training.metrics import ConfusionCounts, classify_prediction


def test_wbce_and_trajectory_loss() -> None:
    wbce = build_loss("wbce")
    pred = torch.full((2, 3, 8, 8), 0.5)
    target = torch.zeros_like(pred)
    assert wbce(pred, target).item() > 0
    traj_loss = build_loss("trajectory_mse")
    out = torch.zeros(2, 2, 16)
    tgt = torch.zeros(2, 3, 16)
    tgt[:, 2] = 1
    assert traj_loss(out, tgt).item() == 0


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    model = build_model({"version": "v2", "sequence_length": 3, "base_channels": 4})
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "last.pt"
    save_checkpoint(ckpt_path, model, opt, None, CheckpointState(epoch=0, global_step=1, best_score=0.2, metrics={"val_loss": 0.2}), {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}})
    ckpt = load_checkpoint(ckpt_path)
    restored = build_model({"version": "v2", "sequence_length": 3, "base_channels": 4})
    load_model_weights(restored, ckpt)
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(a, b)


def test_evaluation_classification_metrics() -> None:
    counts = ConfusionCounts()
    for cls in [classify_prediction((10, 10), (12, 12), 4), classify_prediction(None, None, 4), classify_prediction(None, (1, 1), 4), classify_prediction((1, 1), None, 4), classify_prediction((10, 10), (30, 30), 4)]:
        counts.add(cls)
    metrics = counts.metrics()
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 2
