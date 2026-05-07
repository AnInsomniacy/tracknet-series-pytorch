"""Reusable neural network blocks for TrackNet models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_block(in_channels: int, out_channels: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(conv_bn_relu(in_channels if i == 0 else out_channels, out_channels, 3))
    return nn.Sequential(*layers)


class V2EncoderDecoder(nn.Module):
    """TrackNetV2 U-Net style encoder-decoder backbone.

    This module returns raw heatmap logits. A caller applies Sigmoid, motion
    fusion, or residual refinement according to the target paper version.
    """

    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0, base_channels: int = 64):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = conv_block(input_channels, c1, 2)
        self.enc2 = conv_block(c1, c2, 2)
        self.enc3 = conv_block(c2, c3, 3)
        self.enc4 = conv_block(c3, c4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)
        self.dec1 = conv_block(c4 + c3, c3, 3)
        self.dec2 = conv_block(c3 + c2, c2, 2)
        self.dec3 = conv_block(c2 + c1, c1, 2)
        self.output = nn.Conv2d(c1, output_channels, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.dropout(self.enc4(self.pool(e3)))
        d1 = F.interpolate(bottleneck, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([d1, e3], dim=1))
        d2 = F.interpolate(d1, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d3 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d3 = self.dec3(torch.cat([d3, e1], dim=1))
        return d3

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(x)


def sinusoidal_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
