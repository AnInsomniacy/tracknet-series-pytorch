"""Shared protocol objects for TrackNet paper integrations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import torch
import torch.nn as nn

from tracknet.data.targets import HeatmapTargetPolicy

PostprocessKind = Literal["v1_hough", "largest_blob"]
TrainingDatasetKind = Literal["heatmap", "trajectory"]
SplitStrategy = Literal["sequence", "trajectory_sequence"]
WindowAggregationKind = Literal["last", "weighted_heatmap"]


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
    training_dataset_kind: TrainingDatasetKind = "heatmap"
    split_strategy: SplitStrategy = "sequence"
    window_aggregation: WindowAggregationKind = "weighted_heatmap"

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
            policy = replace(policy, seed=int(resolved["seed"]))
        return ProcessedTrackNetDataset(TrackNetDatasetConfig.from_mapping(resolved), target_policy=policy)

    def build_training_dataset(self, dataset_cfg: dict[str, Any]):
        if self.training_dataset_kind == "trajectory":
            from tracknet.data.trajectory_dataset import TrajectoryRectifierDataset, TrajectoryRectifierDatasetConfig

            return TrajectoryRectifierDataset(TrajectoryRectifierDatasetConfig.from_mapping(self.resolved_dataset_config(dataset_cfg)))
        if self.training_dataset_kind != "heatmap":
            raise ValueError(f"Unsupported training dataset kind: {self.training_dataset_kind}")
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

        return aggregate_window_outputs(
            outputs,
            windows,
            postprocess_kind=self.postprocess_kind,
            sequence_length=sequence_length,
            aggregation_mode=self.window_aggregation,
            threshold=threshold,
            hough_threshold=hough_threshold,
        )

    def video_windows(self, frame_count: int, sequence_length: int) -> list[list[int]]:
        from tracknet.inference.windowing import last_frame_windows, sliding_windows

        if self.window_aggregation == "last":
            return last_frame_windows(frame_count, sequence_length)
        return sliding_windows(frame_count, sequence_length)
