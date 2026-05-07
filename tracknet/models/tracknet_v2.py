"""TrackNetV2: 3-in/3-out U-Net with weighted BCE supervision."""

from __future__ import annotations

import torch
import torch.nn as nn

from tracknet.models.blocks import V2EncoderDecoder


class TrackNetV2(nn.Module):
    def __init__(self, sequence_length: int = 3, input_channels: int | None = None, output_channels: int | None = None, dropout: float = 0.0, base_channels: int = 64):
        super().__init__()
        self.sequence_length = sequence_length
        in_ch = input_channels if input_channels is not None else sequence_length * 3
        out_ch = output_channels if output_channels is not None else sequence_length
        self.backbone = V2EncoderDecoder(in_ch, out_ch, dropout=dropout, base_channels=base_channels)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_logits(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))
