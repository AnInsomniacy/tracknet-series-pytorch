"""TrackNetV3 tracker and rectifier contracts."""

from __future__ import annotations

from tracknet.models.tracknet_v3 import TrackNetV3Tracker, TrajectoryRectifier
from tracknet.papers.base import HeatmapTargetPolicy, PaperSpec


def build_tracker_spec() -> PaperSpec:
    return PaperSpec(
        paper_id="v3",
        model_version="v3",
        loss_name="wbce",
        model_factory=TrackNetV3Tracker,
        target_policy=HeatmapTargetPolicy(
            target_frame_mode="all",
            heatmap_mode="binary_disk",
            radius=3.0,
            include_background=True,
            video_mixup_alpha=0.4,
            video_mixup_probability=0.5,
        ),
        dataset_defaults={
            "sequence_length": 8,
        },
    )


def build_rectifier_spec() -> PaperSpec:
    return PaperSpec(
        paper_id="v3_rectifier",
        model_version="v3_rectifier",
        loss_name="trajectory_mse",
        model_factory=TrajectoryRectifier,
        dataset_defaults={
            "type": "trajectory",
            "trajectory_length": 16,
            "mask_ratio": 0.3,
            "delta_y_pixels": 30.0,
        },
    )
