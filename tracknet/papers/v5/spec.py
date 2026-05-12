"""TrackNetV5 paper contract."""

from __future__ import annotations

from tracknet.data.targets import HeatmapTargetPolicy
from tracknet.models.tracknet_v5 import TrackNetV5
from tracknet.papers.base import EvaluationProtocol, PaperSpec


def build_spec(ablation: str = "full", paper_id: str = "v5") -> PaperSpec:
    return PaperSpec(
        paper_id=paper_id,
        model_version="v5",
        loss_name="wbce",
        model_factory=lambda **kwargs: TrackNetV5(ablation=ablation, **kwargs),
        evaluation_protocol=EvaluationProtocol(threshold=0.5, tolerance_pixels=4.0, coordinate_space="model"),
        target_policy=HeatmapTargetPolicy(target_frame_mode="all", heatmap_mode="binary_disk", radius=30.0),
        dataset_defaults={
            "sequence_length": 3,
        },
    )


def build_mdd_spec() -> PaperSpec:
    return build_spec(ablation="mdd", paper_id="v5_mdd")


def build_rstr_spec() -> PaperSpec:
    return build_spec(ablation="rstr", paper_id="v5_rstr")


def build_full_spec() -> PaperSpec:
    return build_spec(ablation="full", paper_id="v5_full")
