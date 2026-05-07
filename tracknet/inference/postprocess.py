"""Post-processing shared by evaluation and video inference.

The papers differ in the final coordinate extraction step. Keeping this logic in
one file prevents training/evaluation/inference drift: the same heatmap threshold,
blob extraction, V1 Hough rule, and coordinate convention are used everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from tracknet.data.heatmaps import heatmap_largest_blob_centroid, v1_hough_coordinate

PostprocessKind = Literal["v1_hough", "largest_blob"]


@dataclass(frozen=True)
class Prediction:
    """A prediction in model-coordinate space unless explicitly mapped later."""

    visibility: int
    x: float
    y: float
    score: float

    @property
    def coordinate(self) -> tuple[float, float] | None:
        if self.visibility != 1 or self.x < 0 or self.y < 0:
            return None
        return float(self.x), float(self.y)


def _to_numpy_2d(heatmap: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().float().cpu().numpy()
    arr = np.asarray(heatmap)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap [H,W], got shape {arr.shape}")
    return arr


def decode_heatmap(
    heatmap: torch.Tensor | np.ndarray,
    *,
    kind: PostprocessKind = "largest_blob",
    threshold: float = 0.5,
    hough_threshold: int = 128,
) -> Prediction:
    """Convert one heatmap to a coordinate.

    V2/V3/V4/V5 use thresholded connected components and largest-blob centroid.
    V1 uses its original grayscale-threshold-plus-Hough rule and reports no ball
    unless exactly one circle is found.
    """
    arr = _to_numpy_2d(heatmap)
    score = float(np.max(arr)) if arr.size else 0.0
    if kind == "v1_hough":
        coord = v1_hough_coordinate(arr.astype(np.uint8), threshold=int(hough_threshold))
    elif kind == "largest_blob":
        coord = heatmap_largest_blob_centroid(arr.astype(np.float32), threshold=float(threshold))
    else:
        raise ValueError(f"Unsupported postprocess kind: {kind}")
    if coord is None:
        return Prediction(visibility=0, x=-1.0, y=-1.0, score=score)
    return Prediction(visibility=1, x=float(coord[0]), y=float(coord[1]), score=score)


def decode_model_output(
    output: torch.Tensor,
    *,
    postprocess_kind: PostprocessKind,
    threshold: float = 0.5,
    hough_threshold: int = 128,
) -> list[Prediction]:
    """Decode a single sample model output into per-target predictions.

    Expected input shapes:
    - V1: [256,H,W] logits for one target frame. The predicted grayscale class
      map is decoded by the V1 Hough procedure.
    - V2/V3/V4/V5: [T,H,W] probability heatmaps.
    """
    if output.ndim == 4 and output.shape[0] == 1:
        output = output[0]
    kind = postprocess_kind
    if kind == "v1_hough":
        if output.ndim != 3 or output.shape[0] != 256:
            raise ValueError(f"V1 output must be [256,H,W], got {tuple(output.shape)}")
        class_map = torch.argmax(output, dim=0).to(torch.uint8)
        return [decode_heatmap(class_map, kind="v1_hough", hough_threshold=hough_threshold)]
    if output.ndim == 2:
        output = output.unsqueeze(0)
    if output.ndim != 3:
        raise ValueError(f"Heatmap model output must be [T,H,W], got {tuple(output.shape)}")
    return [decode_heatmap(output[i], kind=kind, threshold=threshold, hough_threshold=hough_threshold) for i in range(output.shape[0])]
