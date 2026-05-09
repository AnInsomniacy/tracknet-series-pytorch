"""Target policies used by processed datasets.

The processed dataset owns neutral frame and annotation loading. A target
policy describes how a caller wants those annotations converted into tensors
without forcing the dataset to know about TrackNet paper versions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeatmapTargetPolicy:
    """Heatmap target and optional sample-mixing semantics.

    This object is intentionally data-layer vocabulary. Paper packages may
    construct policies, but the shared paper registry should not own heatmap
    modes, background flags, or augmentation knobs.
    """

    target_frame_mode: str = "all"
    heatmap_mode: str = "gaussian"
    sigma: float = 3.0
    radius: float = 30.0
    include_background: bool = False
    video_mixup_alpha: float = 0.0
    video_mixup_probability: float = 0.0
    seed: int = 26
