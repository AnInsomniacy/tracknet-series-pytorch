"""TrackNetV5 paper contract."""

from __future__ import annotations

from tracknet.models.tracknet_v5 import TrackNetV5
from tracknet.papers.base import HeatmapTargetPolicy, PaperSpec


def build_spec() -> PaperSpec:
    return PaperSpec(
        paper_id="v5",
        model_version="v5",
        loss_name="wbce",
        model_factory=TrackNetV5,
        target_policy=HeatmapTargetPolicy(target_frame_mode="all", heatmap_mode="binary_disk", radius=30.0),
        dataset_defaults={
            "sequence_length": 3,
        },
    )
