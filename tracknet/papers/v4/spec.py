"""TrackNetV4 paper contract."""

from __future__ import annotations

from tracknet.models.tracknet_v4 import TrackNetV4
from tracknet.papers.base import HeatmapTargetPolicy, PaperSpec


def build_spec() -> PaperSpec:
    return PaperSpec(
        paper_id="v4",
        model_version="v4",
        loss_name="wbce",
        model_factory=TrackNetV4,
        target_policy=HeatmapTargetPolicy(target_frame_mode="all", heatmap_mode="gaussian", sigma=3.0),
        dataset_defaults={
            "sequence_length": 3,
        },
    )
