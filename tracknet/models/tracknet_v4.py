"""TrackNetV4 motion-aware feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from tracknet.models.blocks import V2EncoderDecoder


class MotionPromptV4(nn.Module):
    def __init__(self, init_slope: float = 16.24, init_shift: float = 0.28):
        super().__init__()
        self.log_slope = nn.Parameter(torch.log(torch.tensor(float(init_slope))))
        self.shift = nn.Parameter(torch.tensor(float(init_shift)))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return motion attention maps [B,T-1,H,W] from frames [B,T,3,H,W]."""
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}")
        gray = frames.mean(dim=2)
        diff_abs = torch.abs(gray[:, 1:] - gray[:, :-1])
        slope = torch.exp(self.log_slope).clamp(1e-3, 100.0)
        return torch.sigmoid(slope * (diff_abs - self.shift))


class TrackNetV4(nn.Module):
    def __init__(self, sequence_length: int = 3, input_channels: int | None = None, output_channels: int | None = None, dropout: float = 0.0, base_channels: int = 64, fusion_variant: str = "eq4"):
        super().__init__()
        self.sequence_length = sequence_length
        in_ch = input_channels if input_channels is not None else sequence_length * 3
        out_ch = output_channels if output_channels is not None else sequence_length
        if out_ch != sequence_length:
            raise ValueError("TrackNetV4 motion fusion expects one output channel per input frame")
        if fusion_variant not in {"eq4", "mean"}:
            raise ValueError("fusion_variant must be 'eq4' or 'mean'")
        self.fusion_variant = fusion_variant
        self.backbone = V2EncoderDecoder(in_ch, out_ch, dropout=dropout, base_channels=base_channels)
        self.motion_prompt = MotionPromptV4()
        fusion_channels = base_channels if fusion_variant == "mean" else base_channels * sequence_length
        self.fusion_output = nn.Conv2d(fusion_channels, out_ch, kernel_size=1)

    def _frames_from_input(self, x: torch.Tensor) -> torch.Tensor:
        needed = self.sequence_length * 3
        if x.shape[1] < needed:
            raise ValueError(f"Input has {x.shape[1]} channels but needs at least {needed}")
        return x[:, :needed].reshape(x.shape[0], self.sequence_length, 3, x.shape[-2], x.shape[-1])

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        visual_features = self.backbone.forward_features(x)
        attention = self.motion_prompt(self._frames_from_input(x))
        attention = torch.nn.functional.interpolate(attention, size=visual_features.shape[-2:], mode="bilinear", align_corners=False)
        if self.fusion_variant == "mean":
            fused_feature = visual_features * attention.mean(dim=1, keepdim=True)
            return self.fusion_output(fused_feature)
        # Eq. (4) is a feature-level fusion: the motion maps modulate high-level
        # visual representations before the final heatmap projection.
        fused = [visual_features]
        for t in range(1, self.sequence_length):
            fused.append(visual_features * attention[:, t - 1 : t])
        return self.fusion_output(torch.cat(fused, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))
