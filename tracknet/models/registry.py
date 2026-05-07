from __future__ import annotations

from typing import Any

import torch.nn as nn

from tracknet.papers import get_paper_spec


def build_model(cfg: dict[str, Any]) -> nn.Module:
    version = str(cfg.get("version", cfg.get("model_version", "v2"))).lower()
    kwargs = dict(cfg.get("kwargs", {}))
    # Permit common model fields at top-level for concise configs.
    for key in [
        "sequence_length",
        "input_channels",
        "output_channels",
        "dropout",
        "base_channels",
        "rstr_patch_size",
        "rstr_embed_dim",
        "rstr_heads",
        "rstr_layers",
        "stochastic_context_dropout",
        "fusion_variant",
        "hidden_channels",
        "ablation",
    ]:
        if key in cfg and key not in kwargs:
            kwargs[key] = cfg[key]
    if version.endswith("_mdd"):
        kwargs["ablation"] = "mdd"
    elif version.endswith("_rstr"):
        kwargs["ablation"] = "rstr"
    spec = get_paper_spec(version)
    if spec.model_factory is None:
        raise ValueError(f"Unknown model version: {version}")
    return spec.model_factory(**kwargs)
