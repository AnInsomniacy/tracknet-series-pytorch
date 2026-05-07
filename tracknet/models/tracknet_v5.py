"""TrackNetV5: MDD + residual spatio-temporal refinement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracknet.models.blocks import V2EncoderDecoder, sinusoidal_encoding


class MotionDirectionDecoupling(nn.Module):
    """MDD module from TrackNetV5.

    For three RGB frames it creates a 13-channel tensor:
    RGB(t-1), two polarity attention maps, RGB(t), two polarity attention maps,
    RGB(t+1). The polarity maps are computed from signed grayscale differences.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))
        self.eps = eps

    def attention_from_polarity(self, x: torch.Tensor) -> torch.Tensor:
        k = 5.0 / (0.45 * torch.abs(torch.tanh(self.alpha)) + self.eps)
        m = 0.6 * torch.tanh(self.beta)
        return torch.sigmoid(k * (torch.abs(x) - m))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[1] != 3 or frames.shape[2] != 3:
            raise ValueError(f"MDD expects frames [B,3,3,H,W], got {tuple(frames.shape)}")
        gray = frames.mean(dim=2)
        diff01 = gray[:, 1] - gray[:, 0]
        diff12 = gray[:, 2] - gray[:, 1]
        maps = []
        for diff in (diff01, diff12):
            pos = F.relu(diff)
            neg = F.relu(-diff)
            maps.append(torch.stack([self.attention_from_polarity(pos), self.attention_from_polarity(neg)], dim=1))
        att01, att12 = maps
        x_aug = torch.cat([frames[:, 0], att01, frames[:, 1], att12, frames[:, 2]], dim=1)
        motion_maps = torch.cat([att01, att12], dim=1)  # [B,4,H,W]
        return x_aug, motion_maps


class FactorizedSpatioTemporalBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout=dropout, batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout=dropout, batch_first=True, norm_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,D]
        b, t, n, d = x.shape
        temporal_in = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        temporal_out = self.temporal(temporal_in).reshape(b, n, t, d).permute(0, 2, 1, 3)
        spatial_in = temporal_out.reshape(b * t, n, d)
        spatial_out = self.spatial(spatial_in).reshape(b, t, n, d)
        return spatial_out


class TSATTHead(nn.Module):
    """Lightweight factorized spatio-temporal Transformer residual head."""

    def __init__(self, sequence_length: int = 3, patch_size: int = 16, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        patch_area = patch_size * patch_size
        self.patch_embed = nn.Linear(patch_area, embed_dim)
        self.blocks = nn.ModuleList([FactorizedSpatioTemporalBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.patch_decode = nn.Linear(embed_dim, patch_area)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        p = self.patch_size
        h, w = x.shape[-2:]
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    def forward(self, draft: torch.Tensor) -> torch.Tensor:
        if draft.ndim != 4:
            raise ValueError(f"TSATTHead expects [B,T,H,W], got {tuple(draft.shape)}")
        b, t, h0, w0 = draft.shape
        if t != self.sequence_length:
            raise ValueError(f"Expected T={self.sequence_length}, got {t}")
        x, (pad_h, pad_w) = self._pad_to_patch(draft)
        _, _, h, w = x.shape
        p = self.patch_size
        gh, gw = h // p, w // p
        # Non-overlapping patches per heatmap channel/time step.
        patches = F.unfold(x.reshape(b * t, 1, h, w), kernel_size=p, stride=p).transpose(1, 2)
        tokens = self.patch_embed(patches).reshape(b, t, gh * gw, self.embed_dim)
        # Factorized positional encodings: temporal + spatial grid.
        temporal_pe = sinusoidal_encoding(t, self.embed_dim, draft.device).view(1, t, 1, self.embed_dim)
        y_pe = sinusoidal_encoding(gh, self.embed_dim, draft.device)
        x_pe = sinusoidal_encoding(gw, self.embed_dim, draft.device)
        spatial_pe = (y_pe[:, None, :] + x_pe[None, :, :]).reshape(1, 1, gh * gw, self.embed_dim)
        tokens = tokens + temporal_pe + spatial_pe
        for block in self.blocks:
            tokens = block(tokens)
        decoded = self.patch_decode(tokens.reshape(b * t, gh * gw, self.embed_dim))
        residual_tiles = decoded.transpose(1, 2).reshape(b * t, p * p, gh, gw)
        residual = self.pixel_shuffle(residual_tiles).reshape(b, t, h, w)
        if pad_h or pad_w:
            residual = residual[:, :, :h0, :w0]
        return residual


class TrackNetV5(nn.Module):
    def __init__(
        self,
        sequence_length: int = 3,
        dropout: float = 0.0,
        base_channels: int = 64,
        rstr_patch_size: int = 16,
        rstr_embed_dim: int = 64,
        rstr_heads: int = 4,
        rstr_layers: int = 1,
        stochastic_context_dropout: float = 0.1,
        ablation: str = "full",
    ):
        super().__init__()
        if sequence_length != 3:
            raise ValueError("TrackNetV5 paper architecture is defined for exactly three frames")
        if ablation not in {"mdd", "rstr", "full"}:
            raise ValueError("ablation must be 'mdd', 'rstr', or 'full'")
        self.sequence_length = sequence_length
        self.ablation = ablation
        self.mdd = MotionDirectionDecoupling()
        backbone_in = 13 if ablation in {"mdd", "full"} else 9
        self.backbone = V2EncoderDecoder(input_channels=backbone_in, output_channels=sequence_length, dropout=dropout, base_channels=base_channels)
        self.motion_gate = nn.Conv2d(sequence_length + 4, sequence_length, kernel_size=1)
        self.context_dropout = nn.Dropout2d(stochastic_context_dropout)
        self.rstr = TSATTHead(sequence_length, patch_size=rstr_patch_size, embed_dim=rstr_embed_dim, num_heads=rstr_heads, num_layers=rstr_layers)

    def _frames_from_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 9:
            raise ValueError(f"TrackNetV5 needs three RGB frames (9 channels), got {x.shape[1]}")
        return x[:, :9].reshape(x.shape[0], 3, 3, x.shape[-2], x.shape[-1])

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        frames = self._frames_from_input(x)
        x_aug, motion_maps = self.mdd(frames)
        draft_input = x_aug if self.ablation in {"mdd", "full"} else x[:, :9]
        draft_logits = self.backbone.forward_logits(draft_input)
        if self.ablation == "mdd":
            return self.motion_gate(torch.cat([draft_logits, motion_maps], dim=1))
        draft_mdd = self.motion_gate(torch.cat([draft_logits, motion_maps], dim=1)) if self.ablation == "full" else draft_logits
        draft_for_residual = self.context_dropout(draft_mdd) if self.training else draft_mdd
        residual = self.rstr(draft_for_residual)
        # The V5 paper defines the training prediction as
        # sigmoid(Dropout(DraftMDD) + Delta_train). The clean draft is used at
        # inference only. Keeping this branch explicit prevents the refinement
        # head from being trained against a different formula than deployment.
        return draft_for_residual + residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))
