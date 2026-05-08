from __future__ import annotations

from typing import Any

import torch.nn as nn

from tracknet.papers import get_paper_spec, paper_id_from_model_config


def build_model(cfg: dict[str, Any]) -> nn.Module:
    version = paper_id_from_model_config(cfg).lower()
    spec = get_paper_spec(version)
    return spec.build_model(dict(cfg))
