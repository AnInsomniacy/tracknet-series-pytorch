"""Frame-level aggregation shared by evaluation and video inference.

TrackNet MIMO models produce several heatmaps per sliding window. A frame can
therefore receive multiple predictions from overlapping windows. Metrics must be
computed after reducing those window-level outputs to one prediction per raw
frame; otherwise validation counts the same frame multiple times and drifts from
video inference semantics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from tracknet.inference.postprocess import PostprocessKind, Prediction, decode_model_output


@dataclass(frozen=True)
class FramePrediction:
    frame: int
    prediction: Prediction


def weighted_average_heatmaps(items: list[tuple[np.ndarray, float]]) -> np.ndarray:
    if not items:
        raise ValueError("Cannot average an empty heatmap list")
    total_weight = float(sum(weight for _, weight in items))
    if total_weight <= 0.0:
        return sum(heatmap for heatmap, _ in items) / float(len(items))
    return sum(heatmap * weight for heatmap, weight in items) / total_weight


def aggregate_window_outputs(
    outputs: torch.Tensor,
    windows: Iterable[list[int]],
    *,
    postprocess_kind: PostprocessKind,
    sequence_length: int,
    aggregation_mode: str,
    threshold: float,
    hough_threshold: int = 128,
) -> list[FramePrediction]:
    """Decode batched window outputs into one prediction per frame id."""
    window_list = list(windows)
    if len(window_list) != int(outputs.shape[0]):
        raise ValueError(f"Expected {len(window_list)} model outputs, got {int(outputs.shape[0])}")
    kind = postprocess_kind
    if aggregation_mode == "last":
        predictions: dict[int, Prediction] = {}
        for sample, window in zip(outputs, window_list):
            frame_id = int(window[-1])
            predictions[frame_id] = decode_model_output(sample, postprocess_kind=kind, threshold=threshold, hough_threshold=hough_threshold)[0]
        return [FramePrediction(frame, predictions[frame]) for frame in sorted(predictions)]

    heatmaps_by_frame: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    center = (sequence_length - 1) / 2.0
    sigma = max(1.0, sequence_length / 4.0)
    for sample, window in zip(outputs, window_list):
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)
        for local_idx, frame_id in enumerate(window):
            if local_idx >= sample.shape[0]:
                continue
            weight = float(np.exp(-((local_idx - center) ** 2) / (2.0 * sigma * sigma)))
            heatmaps_by_frame[int(frame_id)].append((sample[local_idx].detach().cpu().numpy().astype(np.float32), weight))

    result: list[FramePrediction] = []
    for frame_id in sorted(heatmaps_by_frame):
        merged = weighted_average_heatmaps(heatmaps_by_frame[frame_id])
        result.append(FramePrediction(frame_id, decode_model_output(torch.from_numpy(merged), postprocess_kind=kind, threshold=threshold)[0]))
    return result
