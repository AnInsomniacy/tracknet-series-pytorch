"""TrackNetV2 paper contract."""

from __future__ import annotations

from tracknet.data.targets import HeatmapTargetPolicy
from tracknet.models.tracknet_v2 import TrackNetV2
from tracknet.papers.base import EvaluationProtocol, PaperSpec


def build_spec() -> PaperSpec:
    return PaperSpec(
        paper_id="v2",
        model_version="v2",
        loss_name="wbce",
        model_factory=TrackNetV2,
        evaluation_protocol=EvaluationProtocol(threshold=0.5, tolerance_pixels=4.0, coordinate_space="model"),
        target_policy=HeatmapTargetPolicy(target_frame_mode="all", heatmap_mode="gaussian", sigma=3.0),
        dataset_defaults={
            "sequence_length": 3,
        },
    )
