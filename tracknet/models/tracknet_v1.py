"""TrackNetV1 implementation following the paper layer table.

The first TrackNet paper uses VGG16-style encoding layers followed by a
DeconvNet-style decoder. The network predicts a 256-bin grayscale class at each
pixel; `CrossEntropyLoss` applies the pixel-wise softmax during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracknet.models.blocks import conv_bn_relu


def _conv_stack(in_channels: int, depths: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_channels
    for depth in depths:
        layers.append(conv_bn_relu(current, depth, 3))
        current = depth
    return nn.Sequential(*layers)


class TrackNetV1(nn.Module):
    def __init__(self, sequence_length: int = 3, input_channels: int | None = None, num_bins: int = 256, dropout: float = 0.0, base_channels: int = 64):
        super().__init__()
        in_ch = input_channels if input_channels is not None else sequence_length * 3
        self.sequence_length = sequence_length
        self.num_bins = num_bins
        self.encoder_depths = (
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
            base_channels * 8,
        )
        self.decoder_depths = (
            base_channels * 8,
            base_channels * 8,
            base_channels * 8,
            base_channels * 2,
            base_channels * 2,
            base_channels,
            base_channels,
            num_bins,
        )
        self.enc1 = _conv_stack(in_ch, self.encoder_depths[0:2])
        self.enc2 = _conv_stack(self.encoder_depths[1], self.encoder_depths[2:4])
        self.enc3 = _conv_stack(self.encoder_depths[3], self.encoder_depths[4:7])
        self.enc4 = _conv_stack(self.encoder_depths[6], self.encoder_depths[7:10])
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)
        self.dec1 = _conv_stack(self.encoder_depths[-1], self.decoder_depths[0:3])
        self.dec2 = _conv_stack(self.decoder_depths[2], self.decoder_depths[3:5])
        self.dec3 = _conv_stack(self.decoder_depths[4], self.decoder_depths[5:8])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.dropout(self.enc4(self.pool(e3)))
        z = F.interpolate(z, scale_factor=2, mode="nearest")
        z = self.dec1(z)
        z = F.interpolate(z, scale_factor=2, mode="nearest")
        z = self.dec2(z)
        z = F.interpolate(z, scale_factor=2, mode="nearest")
        z = self.dec3(z)
        return z  # [B,256,H,W] logits for CrossEntropyLoss

    @torch.no_grad()
    def heatmap_uint8(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=1).to(torch.uint8)
