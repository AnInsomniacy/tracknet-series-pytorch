"""Shared protocol objects for TrackNet paper integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn as nn

PostprocessKind = Literal["v1_hough", "largest_blob"]


@dataclass(frozen=True)
class HeatmapTargetPolicy:
    """Paper-owned target and input semantics for heatmap datasets.

    The processed dataset is intentionally neutral: it knows how to read frames,
    annotations, backgrounds, and sliding windows. This policy injects the
    paper-specific target representation and optional input augmentation.
    Keeping this object outside the dataset prevents global data code from
    becoming a switchboard for TrackNetV1-V5 behavior.
    """

    target_frame_mode: str = "all"
    heatmap_mode: str = "gaussian"
    sigma: float = 3.0
    radius: float = 30.0
    include_background: bool = False
    video_mixup_alpha: float = 0.0
    video_mixup_probability: float = 0.0
    seed: int = 26


@dataclass(frozen=True)
class PaperSpec:
    """Version contract consumed by pipeline stages.

    Engines depend on this contract instead of embedding paper conditionals.
    Each paper package owns model construction, dataset target semantics,
    post-processing, and window aggregation policy.
    """

    paper_id: str
    model_version: str
    loss_name: str
    model_factory: Callable[..., nn.Module] | None = None
    dataset_defaults: dict[str, Any] = field(default_factory=dict)
    target_policy: HeatmapTargetPolicy | None = None
    postprocess_kind: PostprocessKind = "largest_blob"
    tolerance_pixels: float = 4.0

    def build_model(self, cfg: dict[str, Any]) -> nn.Module:
        if self.model_factory is None:
            raise ValueError(f"Paper spec {self.paper_id} does not define a model factory")
        kwargs = dict(cfg.get("kwargs", {}))
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
        return self.model_factory(**kwargs)

    def resolved_dataset_config(self, dataset_cfg: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(self.dataset_defaults)
        resolved.update(dataset_cfg)
        return resolved

    def build_heatmap_dataset(self, dataset_cfg: dict[str, Any]):
        from tracknet.data.dataset import ProcessedTrackNetDataset, TrackNetDatasetConfig

        resolved = self.resolved_dataset_config(dataset_cfg)
        policy = self.target_policy
        if policy is None:
            policy = HeatmapTargetPolicy()
        if "seed" in resolved:
            policy = HeatmapTargetPolicy(
                target_frame_mode=policy.target_frame_mode,
                heatmap_mode=policy.heatmap_mode,
                sigma=policy.sigma,
                radius=policy.radius,
                include_background=policy.include_background,
                video_mixup_alpha=policy.video_mixup_alpha,
                video_mixup_probability=policy.video_mixup_probability,
                seed=int(resolved["seed"]),
            )
        return ProcessedTrackNetDataset(TrackNetDatasetConfig.from_mapping(resolved), target_policy=policy)

    def build_training_dataset(self, dataset_cfg: dict[str, Any]):
        if self.paper_id.endswith("rectifier"):
            from tracknet.data.trajectory_dataset import TrajectoryRectifierDataset, TrajectoryRectifierDatasetConfig

            return TrajectoryRectifierDataset(TrajectoryRectifierDatasetConfig.from_mapping(self.resolved_dataset_config(dataset_cfg)))
        return self.build_heatmap_dataset(dataset_cfg)

    def aggregate_window_outputs(
        self,
        outputs: torch.Tensor,
        windows: list[list[int]],
        *,
        sequence_length: int,
        threshold: float,
        hough_threshold: int = 128,
    ):
        from tracknet.inference.aggregation import aggregate_window_outputs

        mode = self.target_policy.target_frame_mode if self.target_policy is not None else "all"
        return aggregate_window_outputs(
            outputs,
            windows,
            postprocess_kind=self.postprocess_kind,
            sequence_length=sequence_length,
            target_frame_mode=mode,
            threshold=threshold,
            hough_threshold=hough_threshold,
        )

    def video_windows(self, frame_count: int, sequence_length: int) -> list[list[int]]:
        from tracknet.inference.windowing import last_frame_windows, sliding_windows

        if self.postprocess_kind == "v1_hough" or (self.target_policy and self.target_policy.target_frame_mode == "last"):
            return last_frame_windows(frame_count, sequence_length)
        return sliding_windows(frame_count, sequence_length)
