"""Processed TrackNet datasets and sliding-window sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tracknet.constants import (
    PROCESSED_FRAME_COL,
    PROCESSED_VISIBILITY_COL,
    PROCESSED_X_MODEL_COL,
    PROCESSED_Y_MODEL_COL,
)
from tracknet.data.heatmaps import make_heatmap
from tracknet.data.targets import HeatmapTargetPolicy
from tracknet.data.transforms import image_to_tensor_chw_uint8_rgb
from tracknet.utils.io import read_json


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    match_name: str
    sequence_name: str
    sequence_root: Path
    background_path: Path | None
    width: int
    height: int
    annotations: pd.DataFrame


@dataclass(frozen=True)
class WindowRecord:
    sequence_index: int
    start: int
    length: int


@dataclass(frozen=True)
class TrackNetDatasetConfig:
    processed_root: Path
    sequence_length: int
    frame_stride: int = 1
    split_file: Path | None = None

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "TrackNetDatasetConfig":
        split = cfg.get("split_file")
        return cls(
            processed_root=Path(cfg["processed_root"]),
            sequence_length=int(cfg.get("sequence_length", 3)),
            frame_stride=max(1, int(cfg.get("frame_stride", 1))),
            split_file=Path(split) if split else None,
        )


class ProcessedTrackNetDataset(Dataset[dict[str, Any]]):
    """Sliding-window dataset over the processed structure.

    Returned tensors:
    - frames: [T, 3, H, W], float in [0, 1]
    - input: [C, H, W], where C=3*T plus optional 3-channel background
    - target: [T,H,W] for MIMO or [H,W]/[T,H,W] depending on target mode
    - target_classes: [H,W] for V1 softmax targets when heatmap_mode=v1_uint8
    """

    def __init__(self, cfg: TrackNetDatasetConfig, target_policy: HeatmapTargetPolicy | None = None):
        if cfg.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.cfg = cfg
        self.target_policy = target_policy or HeatmapTargetPolicy()
        self.root = cfg.processed_root
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Processed manifest not found: {manifest_path}. Run preprocessing first.")
        self.manifest = read_json(manifest_path)
        self.sequences = self._load_sequences()
        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(f"No valid {cfg.sequence_length}-frame windows found in {self.root}")

    @property
    def width(self) -> int:
        return int(self.manifest["target_width"])

    @property
    def height(self) -> int:
        return int(self.manifest["target_height"])

    def _load_sequences(self) -> list[SequenceRecord]:
        records: list[SequenceRecord] = []
        allowed_sequence_ids: set[str] | None = None
        if self.cfg.split_file is not None:
            if not self.cfg.split_file.exists():
                raise FileNotFoundError(f"Split file not found: {self.cfg.split_file}")
            allowed_sequence_ids = {line.strip() for line in self.cfg.split_file.read_text().splitlines() if line.strip()}
        for meta in self.manifest["sequences"]:
            sid = meta["sequence_id"]
            if allowed_sequence_ids is not None and sid not in allowed_sequence_ids:
                continue
            seq_root = self.root / "sequences" / sid
            ann_path = seq_root / "annotations.csv"
            if not ann_path.exists():
                raise FileNotFoundError(f"Processed annotations missing: {ann_path}")
            ann = pd.read_csv(ann_path).sort_values(PROCESSED_FRAME_COL).reset_index(drop=True)
            background_key = str(meta.get("background_key", meta["match_name"]))
            bg_path = self.root / "backgrounds" / f"{background_key}.png"
            records.append(
                SequenceRecord(
                    sequence_id=sid,
                    match_name=meta["match_name"],
                    sequence_name=meta["sequence_name"],
                    sequence_root=seq_root,
                    background_path=bg_path if bg_path.exists() else None,
                    width=int(meta["target_width"]),
                    height=int(meta["target_height"]),
                    annotations=ann,
                )
            )
        return records

    def _build_windows(self) -> list[WindowRecord]:
        windows: list[WindowRecord] = []
        for seq_idx, seq in enumerate(self.sequences):
            n = len(seq.annotations)
            if n < self.cfg.sequence_length:
                continue
            for start in range(0, n - self.cfg.sequence_length + 1, self.cfg.frame_stride):
                windows.append(WindowRecord(seq_idx, start, self.cfg.sequence_length))
        return windows

    def _load_rgb_tensor(self, path: Path) -> torch.Tensor:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read processed frame: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image_to_tensor_chw_uint8_rgb(img_rgb))

    def _load_background(self, seq: SequenceRecord) -> torch.Tensor:
        if seq.background_path is None:
            return torch.zeros(3, seq.height, seq.width, dtype=torch.float32)
        return self._load_rgb_tensor(seq.background_path)

    def _target_indices(self) -> list[int]:
        mode = self.target_policy.target_frame_mode
        if mode == "all":
            return list(range(self.cfg.sequence_length))
        if mode == "last":
            return [self.cfg.sequence_length - 1]
        if mode == "center":
            return [self.cfg.sequence_length // 2]
        raise ValueError(f"Unsupported target_frame_mode: {mode}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._load_single(idx)
        if self._should_apply_video_mixup(idx):
            partner_idx, lam = self._mixup_partner_and_lambda(idx)
            partner = self._load_single(partner_idx)
            sample["input"] = sample["input"] * lam + partner["input"] * (1.0 - lam)
            sample["frames"] = sample["frames"] * lam + partner["frames"] * (1.0 - lam)
            if sample["background"] is not None and partner["background"] is not None:
                sample["background"] = sample["background"] * lam + partner["background"] * (1.0 - lam)
            sample["target"] = sample["target"].float() * lam + partner["target"].float() * (1.0 - lam)
            sample["mixup"] = {"enabled": True, "lambda": lam, "partner_index": partner_idx}
        else:
            sample["mixup"] = {"enabled": False, "lambda": 1.0, "partner_index": None}
        return sample

    def _should_apply_video_mixup(self, idx: int) -> bool:
        policy = self.target_policy
        if policy.video_mixup_alpha <= 0.0 or policy.video_mixup_probability <= 0.0 or len(self.windows) < 2:
            return False
        rng = np.random.default_rng(policy.seed + idx * 1009)
        return bool(rng.random() < policy.video_mixup_probability)

    def _mixup_partner_and_lambda(self, idx: int) -> tuple[int, float]:
        policy = self.target_policy
        rng = np.random.default_rng(policy.seed + idx * 1009 + 17)
        partner_idx = int(rng.integers(0, len(self.windows) - 1))
        if partner_idx >= idx:
            partner_idx += 1
        alpha = float(policy.video_mixup_alpha)
        lam = float(rng.beta(alpha, alpha))
        return partner_idx, lam

    def _load_single(self, idx: int) -> dict[str, Any]:
        win = self.windows[idx]
        seq = self.sequences[win.sequence_index]
        rows = seq.annotations.iloc[win.start : win.start + win.length].reset_index(drop=True)
        frames = []
        for _, row in rows.iterrows():
            frame_path = seq.sequence_root / str(row["frame_file"])
            frames.append(self._load_rgb_tensor(frame_path))
        frames_t = torch.stack(frames, dim=0)  # [T,3,H,W]
        input_parts = [frames_t.reshape(win.length * 3, seq.height, seq.width)]
        background = None
        policy = self.target_policy
        if policy.include_background:
            background = self._load_background(seq)
            input_parts.append(background)
        model_input = torch.cat(input_parts, dim=0)

        targets: list[np.ndarray] = []
        target_info: list[dict[str, Any]] = []
        for local_idx in self._target_indices():
            row = rows.iloc[local_idx]
            h = make_heatmap(
                seq.width,
                seq.height,
                int(row[PROCESSED_VISIBILITY_COL]),
                float(row[PROCESSED_X_MODEL_COL]),
                float(row[PROCESSED_Y_MODEL_COL]),
                mode=policy.heatmap_mode,  # type: ignore[arg-type]
                sigma=policy.sigma,
                radius=policy.radius,
            )
            targets.append(h)
            target_info.append(
                {
                    "frame": int(row[PROCESSED_FRAME_COL]),
                    "visibility": int(row[PROCESSED_VISIBILITY_COL]),
                    "x_model": float(row[PROCESSED_X_MODEL_COL]),
                    "y_model": float(row[PROCESSED_Y_MODEL_COL]),
                    "local_index": local_idx,
                }
            )
        if policy.heatmap_mode == "v1_uint8":
            target_tensor = torch.from_numpy(np.stack(targets, axis=0).astype(np.int64))
            if target_tensor.shape[0] == 1:
                target_tensor = target_tensor[0]
        else:
            target_tensor = torch.from_numpy(np.stack(targets, axis=0).astype(np.float32))
            if target_tensor.shape[0] == 1:
                target_tensor = target_tensor[0]

        frame_indices = [int(v) for v in rows[PROCESSED_FRAME_COL].tolist()]
        return {
            "input": model_input,
            "frames": frames_t,
            "background": background,
            "target": target_tensor,
            "sequence_id": seq.sequence_id,
            "match_name": seq.match_name,
            "sequence_name": seq.sequence_name,
            "start": win.start,
            "frame_indices": frame_indices,
            "target_info": target_info,
        }

    def get_window(self, idx: int) -> WindowRecord:
        return self.windows[idx]
