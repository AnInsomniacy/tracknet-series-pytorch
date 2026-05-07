"""Loss functions matching TrackNet paper semantics."""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedBinaryCrossEntropy(nn.Module):
    """TrackNetV2/V3/V4/V5 WBCE.

    The paper defines a pixel-adaptive weight `w` from the predicted heatmap and
    uses `(1-w)^2` for positive pixels and `w^2` for negative pixels. The input
    is expected to be a probability after Sigmoid, not raw logits.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(prediction.float(), self.eps, 1.0 - self.eps)
        tgt = target.float()
        if pred.shape != tgt.shape:
            raise ValueError(f"WBCE shape mismatch: prediction {tuple(pred.shape)} vs target {tuple(tgt.shape)}")
        w = pred.detach()
        loss = -(((1.0 - w) ** 2) * tgt * torch.log(pred) + (w**2) * (1.0 - tgt) * torch.log(1.0 - pred))
        return loss.mean()


class TrajectoryMaskedMSE(nn.Module):
    """MSE for V3 rectifier normalized trajectories.

    Target shape is [B,3,T] where channels are x, y and a valid-coordinate mask.
    Loss is computed only where ground-truth coordinates are valid.
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 3 or target.ndim != 3 or prediction.shape[1] != 2 or target.shape[1] != 3:
            raise ValueError(f"Expected prediction [B,2,T] and target [B,3,T], got {tuple(prediction.shape)} and {tuple(target.shape)}")
        coords = target[:, :2]
        mask = target[:, 2:3]
        denom = torch.clamp(mask.sum() * 2.0, min=1.0)
        return ((prediction - coords) ** 2 * mask).sum() / denom


def build_loss(name: str) -> nn.Module:
    key = name.lower()
    if key in {"wbce", "weighted_bce", "weighted_binary_cross_entropy"}:
        return WeightedBinaryCrossEntropy()
    if key in {"cross_entropy", "ce", "v1_cross_entropy"}:
        return nn.CrossEntropyLoss()
    if key in {"mse", "mean_squared_error"}:
        return nn.MSELoss()
    if key in {"trajectory_mse", "trajectory_masked_mse", "v3_rectifier_mse"}:
        return TrajectoryMaskedMSE()
    raise ValueError(f"Unknown loss: {name}")
