from __future__ import annotations

import torch
import torch.nn as nn

from tracknet.models import build_model
from tracknet.models.tracknet_v5 import TrackNetV5


def test_tracknet_v1_forward_shape() -> None:
    model = build_model({"version": "v1", "sequence_length": 3, "base_channels": 4})
    out = model(torch.rand(2, 9, 32, 32))
    assert out.shape == (2, 256, 32, 32)


def test_tracknet_v2_v3_v4_forward_shapes() -> None:
    v2 = build_model({"version": "v2", "sequence_length": 3, "base_channels": 4})
    assert v2(torch.rand(2, 9, 32, 32)).shape == (2, 3, 32, 32)
    v3 = build_model({"version": "v3", "sequence_length": 4, "base_channels": 4})
    assert v3(torch.rand(2, 15, 32, 32)).shape == (2, 4, 32, 32)
    v4 = build_model({"version": "v4", "sequence_length": 3, "base_channels": 4, "fusion_variant": "mean"})
    assert v4(torch.rand(2, 9, 32, 32)).shape == (2, 3, 32, 32)


def test_tracknet_v5_and_rectifier_forward_shapes() -> None:
    v5 = build_model(
        {
            "version": "v5",
            "base_channels": 4,
            "rstr_patch_size": 8,
            "rstr_embed_dim": 16,
            "rstr_heads": 2,
            "rstr_layers": 1,
        }
    )
    assert v5(torch.rand(1, 9, 32, 32)).shape == (1, 3, 32, 32)
    rectifier = build_model({"version": "v3_rectifier", "hidden_channels": 8})
    assert rectifier(torch.rand(2, 4, 16)).shape == (2, 2, 16)


def test_model_registry_passes_version_specific_kwargs() -> None:
    rectifier = build_model({"version": "v3_rectifier", "hidden_channels": 8})
    assert rectifier.enc1[0].out_channels == 8


class _ConstantBackbone(nn.Module):
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], 3, x.shape[-2], x.shape[-1])


class _PassDraft(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :3]


class _ZeroResidual(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 3, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)


class _ZeroMDD(nn.Module):
    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, _, h, w = frames.shape
        return torch.zeros(b, 13, h, w), torch.zeros(b, 4, h, w)


def test_tracknet_v5_training_uses_masked_draft_for_residual_sum() -> None:
    model = TrackNetV5(base_channels=4, stochastic_context_dropout=1.0)
    model.backbone = _ConstantBackbone()
    model.motion_gate = _PassDraft()
    model.rstr = _ZeroResidual()
    model.mdd = _ZeroMDD()
    model.train()

    output = model(torch.rand(1, 9, 16, 16))

    assert torch.allclose(output, torch.full_like(output, 0.5))
