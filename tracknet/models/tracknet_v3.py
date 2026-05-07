"""TrackNetV3 tracking and trajectory rectification modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracknet.models.blocks import V2EncoderDecoder


class TrackNetV3Tracker(nn.Module):
    """U-Net tracking module using video frames concatenated with background.

    The V3 paper uses eight frames and a match-level median background as
    auxiliary input. `sequence_length` is configurable for tests but defaults to
    the paper value in configs.
    """

    def __init__(self, sequence_length: int = 8, input_channels: int | None = None, output_channels: int | None = None, dropout: float = 0.0, base_channels: int = 64):
        super().__init__()
        self.sequence_length = sequence_length
        in_ch = input_channels if input_channels is not None else sequence_length * 3 + 3
        out_ch = output_channels if output_channels is not None else sequence_length
        self.backbone = V2EncoderDecoder(in_ch, out_ch, dropout=dropout, base_channels=base_channels)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_logits(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class TrajectoryRectifier(nn.Module):
    """1D U-Net trajectory inpainting module from TrackNetV3.

    Input channels are `[x, y, visibility, mask]` over time. Coordinates should
    be normalized to [0, 1] before training/inference; the output is repaired
    `[x, y]` in the same normalized coordinate system.
    """

    def __init__(self, input_channels: int = 4, hidden_channels: int = 64, output_channels: int = 2):
        super().__init__()
        h = hidden_channels
        self.enc1 = nn.Sequential(nn.Conv1d(input_channels, h, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), nn.Conv1d(h, h, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv1d(h, h * 2, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), nn.Conv1d(h * 2, h * 2, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv1d(h * 2, h * 4, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), nn.Conv1d(h * 4, h * 4, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.dec2 = nn.Sequential(nn.Conv1d(h * 4 + h * 2, h * 2, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), nn.Conv1d(h * 2, h * 2, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))
        self.dec1 = nn.Sequential(nn.Conv1d(h * 2 + h, h, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), nn.Conv1d(h, h, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))
        self.out = nn.Conv1d(h, output_channels, 1)

    @staticmethod
    def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.shape[-1] == length:
            return x
        return F.interpolate(x, size=length, mode="linear", align_corners=False)

    def forward(self, trajectory_and_mask: torch.Tensor) -> torch.Tensor:
        # [B,4,T]
        e1 = self.enc1(trajectory_and_mask)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = F.interpolate(e3, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        # Sigmoid keeps normalized coordinates in range. The MSE target should
        # be normalized [0,1] coordinates.
        return torch.sigmoid(self.out(d1))
