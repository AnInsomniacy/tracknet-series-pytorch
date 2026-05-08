"""TrackNetV3 trajectory rectification for predicted coordinate sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tracknet.inference.postprocess import Prediction
from tracknet.models import build_model
from tracknet.training.checkpoint import load_checkpoint, load_model_weights
from tracknet.utils.device import select_device


@dataclass(frozen=True)
class RectificationConfig:
    checkpoint_path: Path
    model: dict[str, Any] | None = None
    trajectory_length: int = 16
    delta_y_pixels: float = 30.0
    batch_size: int = 16
    device: str = "auto"


def _model_config_from_checkpoint(checkpoint_path: Path, explicit_model: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if explicit_model is not None:
        return explicit_model, ckpt
    cfg = ckpt.get("config") or {}
    if isinstance(cfg, dict) and "model" in cfg:
        return dict(cfg["model"]), ckpt
    return {"version": "v3_rectifier"}, ckpt


def build_v3_inpainting_mask(predictions: list[Prediction], *, delta_y_pixels: float) -> np.ndarray:
    """Mask missing intervals using the V3 height-threshold rule.

    The V3 paper marks a missing interval as inpaintable when both bounding
    detections are near the top of the image, using `p_f_y < delta` and
    `p_b_y < delta`. This distinguishes likely occlusion/missed detections
    from trajectories that have left the field of view.
    """
    n = len(predictions)
    mask = np.zeros(n, dtype=np.float32)
    visible = np.array([p.visibility == 1 and p.y >= 0 for p in predictions], dtype=bool)
    visible_indices = np.flatnonzero(visible)
    for i in range(n):
        if visible[i]:
            continue
        prev_candidates = visible_indices[visible_indices < i]
        next_candidates = visible_indices[visible_indices > i]
        if len(prev_candidates) == 0 or len(next_candidates) == 0:
            continue
        p = prev_candidates[-1]
        q = next_candidates[0]
        if predictions[p].y < delta_y_pixels and predictions[q].y < delta_y_pixels:
            mask[i] = 1.0
    return mask


def _window_starts(n: int, length: int) -> list[int]:
    if n <= length:
        return [0]
    return list(range(0, n - length + 1))


def _window_indices(start: int, n: int, length: int) -> list[int]:
    return [min(max(i, 0), n - 1) for i in range(start, start + length)]


def rectify_predictions(predictions: list[Prediction], *, raw_width: int, raw_height: int, cfg: RectificationConfig) -> list[Prediction]:
    if not predictions:
        return []
    if raw_width <= 1 or raw_height <= 1:
        raise ValueError("raw_width/raw_height must be greater than one for coordinate normalization")
    model_cfg, ckpt = _model_config_from_checkpoint(Path(cfg.checkpoint_path), cfg.model)
    model = build_model(model_cfg)
    load_model_weights(model, ckpt, strict=True)
    device = select_device(cfg.device)
    model.to(device).eval()

    mask = build_v3_inpainting_mask(predictions, delta_y_pixels=float(cfg.delta_y_pixels))
    n = len(predictions)
    starts = _window_starts(n, int(cfg.trajectory_length))
    inputs = []
    windows = []
    for start in starts:
        idxs = _window_indices(start, n, int(cfg.trajectory_length))
        windows.append(idxs)
        x = np.zeros(len(idxs), dtype=np.float32)
        y = np.zeros(len(idxs), dtype=np.float32)
        observed = np.zeros(len(idxs), dtype=np.float32)
        m = np.zeros(len(idxs), dtype=np.float32)
        for j, frame_idx in enumerate(idxs):
            p = predictions[frame_idx]
            if p.visibility == 1 and p.x >= 0 and p.y >= 0:
                x[j] = np.clip(p.x / (raw_width - 1), 0.0, 1.0)
                y[j] = np.clip(p.y / (raw_height - 1), 0.0, 1.0)
                observed[j] = 1.0
            m[j] = mask[frame_idx]
            if m[j] > 0:
                observed[j] = 0.0
                x[j] = 0.0
                y[j] = 0.0
        inputs.append(torch.from_numpy(np.stack([x, y, observed, m], axis=0)))

    repaired_by_frame: dict[int, list[tuple[float, float, float]]] = {i: [] for i in range(n)}
    center = (int(cfg.trajectory_length) - 1) / 2.0
    sigma = max(1.0, int(cfg.trajectory_length) / 4.0)
    with torch.no_grad():
        for start in range(0, len(inputs), int(cfg.batch_size)):
            batch = torch.stack(inputs[start : start + int(cfg.batch_size)], dim=0).to(device).float()
            out = model(batch).detach().cpu().numpy()  # [B,2,T]
            for b, traj in enumerate(out):
                idxs = windows[start + b]
                for local_idx, frame_idx in enumerate(idxs):
                    weight = float(np.exp(-((local_idx - center) ** 2) / (2.0 * sigma * sigma)))
                    repaired_by_frame[frame_idx].append((float(traj[0, local_idx]), float(traj[1, local_idx]), weight))
    repaired = list(predictions)
    for i, original in enumerate(predictions):
        if mask[i] <= 0 or not repaired_by_frame.get(i):
            continue
        total_w = sum(w for _, _, w in repaired_by_frame[i]) or 1.0
        x_norm = sum(x * w for x, _, w in repaired_by_frame[i]) / total_w
        y_norm = sum(y * w for _, y, w in repaired_by_frame[i]) / total_w
        repaired[i] = Prediction(visibility=1, x=float(x_norm * (raw_width - 1)), y=float(y_norm * (raw_height - 1)), score=original.score)
    return repaired
