"""Trajectory windows for the TrackNetV3 rectification module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tracknet.constants import PROCESSED_FRAME_COL, PROCESSED_VISIBILITY_COL, PROCESSED_X_MODEL_COL, PROCESSED_Y_MODEL_COL
from tracknet.utils.io import read_json


@dataclass(frozen=True)
class TrajectoryRectifierDatasetConfig:
    processed_root: Path
    trajectory_length: int = 16
    frame_stride: int = 1
    mask_ratio: float = 0.3
    delta_y_pixels: float = 30.0
    seed: int = 26
    split_file: Path | None = None

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "TrajectoryRectifierDatasetConfig":
        split = cfg.get("split_file")
        return cls(
            processed_root=Path(cfg["processed_root"]),
            trajectory_length=int(cfg.get("trajectory_length", 16)),
            frame_stride=max(1, int(cfg.get("frame_stride", 1))),
            mask_ratio=float(cfg.get("mask_ratio", 0.3)),
            delta_y_pixels=float(cfg.get("delta_y_pixels", 30.0)),
            seed=int(cfg.get("seed", 26)),
            split_file=Path(split) if split else None,
        )


class TrajectoryRectifierDataset(Dataset[dict[str, Any]]):
    """Synthetic inpainting samples for V3 rectifier training.

    The tracking network produces intermittent missed detections. The V3 paper
    trains the rectifier by masking valid trajectory points and optimizing MSE
    against the original normalized coordinates. The fourth input channel is the
    inpainting mask; masked positions have their coordinate/visibility inputs
    zeroed so the network must infer them from context.
    """

    def __init__(self, cfg: TrajectoryRectifierDatasetConfig):
        if not 0.0 <= cfg.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0,1)")
        if cfg.trajectory_length <= 1:
            raise ValueError("trajectory_length must be greater than one")
        self.cfg = cfg
        manifest_path = cfg.processed_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")
        self.manifest = read_json(manifest_path)
        self.width = int(self.manifest["target_width"])
        self.height = int(self.manifest["target_height"])
        self.sequences: list[tuple[str, pd.DataFrame]] = []
        allowed: set[str] | None = None
        if cfg.split_file is not None:
            allowed = {line.strip() for line in cfg.split_file.read_text().splitlines() if line.strip()}
        for meta in self.manifest["sequences"]:
            sid = str(meta["sequence_id"])
            if allowed is not None and sid not in allowed:
                continue
            ann = pd.read_csv(cfg.processed_root / "sequences" / sid / "annotations.csv").sort_values(PROCESSED_FRAME_COL).reset_index(drop=True)
            self.sequences.append((sid, ann))
        self.windows: list[tuple[int, int]] = []
        for seq_idx, (_, ann) in enumerate(self.sequences):
            n = len(ann)
            for start in range(0, max(0, n - cfg.trajectory_length + 1), cfg.frame_stride):
                self.windows.append((seq_idx, start))
        if not self.windows:
            raise ValueError("No trajectory windows available")

    def __len__(self) -> int:
        return len(self.windows)

    def _mask_positions(self, coords: np.ndarray, visible: np.ndarray, idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed + idx)
        mask = np.zeros(self.cfg.trajectory_length, dtype=np.float32)
        visible_indices = np.flatnonzero(visible > 0.5)
        if len(visible_indices) == 0:
            return mask
        n_to_mask = int(round(len(visible_indices) * self.cfg.mask_ratio))
        if n_to_mask > 0:
            chosen = rng.choice(visible_indices, size=min(n_to_mask, len(visible_indices)), replace=False)
            mask[chosen] = 1.0
        # Preserve the V3 inference prior in training: only repair gaps whose
        # bounding detections have similar vertical position (delta_y criterion).
        invisible = np.flatnonzero(visible < 0.5)
        for i in invisible:
            prev_candidates = visible_indices[visible_indices < i]
            next_candidates = visible_indices[visible_indices > i]
            if len(prev_candidates) == 0 or len(next_candidates) == 0:
                continue
            p = prev_candidates[-1]
            n = next_candidates[0]
            if abs((coords[1, p] - coords[1, n]) * (self.height - 1)) < self.cfg.delta_y_pixels:
                mask[i] = 1.0
        return mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start = self.windows[idx]
        sid, ann = self.sequences[seq_idx]
        rows = ann.iloc[start : start + self.cfg.trajectory_length].reset_index(drop=True)
        visible = (rows[PROCESSED_VISIBILITY_COL].to_numpy(dtype=np.float32) == 1).astype(np.float32)
        x = rows[PROCESSED_X_MODEL_COL].to_numpy(dtype=np.float32)
        y = rows[PROCESSED_Y_MODEL_COL].to_numpy(dtype=np.float32)
        x = np.where(visible > 0, np.clip(x / max(1, self.width - 1), 0.0, 1.0), 0.0)
        y = np.where(visible > 0, np.clip(y / max(1, self.height - 1), 0.0, 1.0), 0.0)
        coords = np.stack([x, y], axis=0).astype(np.float32)
        mask = self._mask_positions(coords, visible, idx)
        observed = visible * (1.0 - mask)
        input_coords = coords * observed[None, :]
        inp = np.concatenate([input_coords, observed[None, :], mask[None, :]], axis=0).astype(np.float32)
        target = np.concatenate([coords, visible[None, :]], axis=0).astype(np.float32)
        return {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(target),
            "sequence_id": sid,
            "start": start,
            "frame_indices": [int(v) for v in rows[PROCESSED_FRAME_COL].tolist()],
        }
