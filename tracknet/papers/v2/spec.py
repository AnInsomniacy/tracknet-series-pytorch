"""TrackNetV2 paper contract."""

from __future__ import annotations

from tracknet.models.tracknet_v2 import TrackNetV2
from tracknet.papers.base import PaperSpec


def build_spec() -> PaperSpec:
    return PaperSpec(
        paper_id="v2",
        model_version="v2",
        loss_name="wbce",
        model_factory=TrackNetV2,
        dataset_defaults={
            "sequence_length": 3,
            "model_version": "v2",
            "target_frame_mode": "all",
            "heatmap_mode": "gaussian",
            "sigma": 3.0,
        },
    )
