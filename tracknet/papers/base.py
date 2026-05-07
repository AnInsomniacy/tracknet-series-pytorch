"""Shared protocol objects for TrackNet paper integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn


@dataclass(frozen=True)
class PaperSpec:
    """Version contract consumed by pipeline stages.

    A PaperSpec is intentionally declarative. Version packages own their model
    constructors and dataset defaults; engines consume the resulting contract
    without embedding paper-specific logic.
    """

    paper_id: str
    model_version: str
    loss_name: str
    model_factory: Callable[..., nn.Module] | None = None
    dataset_defaults: dict[str, Any] = field(default_factory=dict)
    postprocess_kind: str = "largest_blob"
    tolerance_pixels: float = 4.0
