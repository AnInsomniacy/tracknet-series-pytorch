"""TrackNetV1 implementation.

The first paper predicts a 256-bin grayscale heatmap per pixel using a softmax
and trains it with pixel-wise cross entropy. For three-frame TrackNetV1 the
three RGB frames are concatenated as channels, and the heatmap corresponds to
the last frame of the window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracknet.models.blocks import conv_block


class TrackNetV1(nn.Module):
    def __init__(self, sequence_length: int = 3, input_channels: int | None = None, num_bins: int = 256, dropout: float = 0.0, base_channels: int = 64):
        super().__init__()
        in_ch = input_channels if input_channels is not None else sequence_length * 3
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.sequence_length = sequence_length
        self.num_bins = num_bins
        self.enc1 = conv_block(in_ch, c1, 2)
        self.enc2 = conv_block(c1, c2, 2)
        self.enc3 = conv_block(c2, c3, 3)
        self.enc4 = conv_block(c3, c4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)
        # V1 follows a VGG16 + DeconvNet style decoder without U-Net skips.
        self.dec1 = conv_block(c4, c4, 3)
        self.dec2 = conv_block(c4, c2, 2)
        self.dec3 = conv_block(c2, c1, 2)
        self.final = nn.Sequential(
            nn.Conv2d(c1, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, num_bins, kernel_size=1),
        )

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
        return self.final(z)  # [B,256,H,W] logits for CrossEntropyLoss

    @torch.no_grad()
    def heatmap_uint8(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=1).to(torch.uint8)
