"""TrackNetV1 paper contract."""

from __future__ import annotations

from tracknet.data.targets import HeatmapTargetPolicy
from tracknet.models.tracknet_v1 import TrackNetV1
from tracknet.papers.base import EvaluationProtocol, PaperSpec


def build_spec() -> PaperSpec:
    return PaperSpec(
        paper_id="v1",
        model_version="v1",
        loss_name="cross_entropy",
        model_factory=TrackNetV1,
        postprocess_kind="v1_hough",
        tolerance_pixels=5.0,
        evaluation_protocol=EvaluationProtocol(tolerance_pixels=5.0, hough_threshold=128, coordinate_space="model"),
        window_aggregation="last",
        target_policy=HeatmapTargetPolicy(target_frame_mode="last", heatmap_mode="v1_uint8", sigma=10.0**0.5),
        dataset_defaults={
            "sequence_length": 3,
        },
    )
