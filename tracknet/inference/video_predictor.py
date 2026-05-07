"""Video inference with frame-accurate CSV and optional visualization output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from tracknet.constants import CSV_OUTPUT_COLUMNS, RAW_FRAME_COL, RAW_VISIBILITY_COL, RAW_X_COL, RAW_Y_COL
from tracknet.data.transforms import LetterboxTransform, image_to_tensor_chw_uint8_rgb, letterbox_image
from tracknet.inference.aggregation import aggregate_window_outputs, weighted_average_heatmaps
from tracknet.inference.postprocess import PostprocessKind, Prediction
from tracknet.inference.rectification import RectificationConfig, rectify_predictions
from tracknet.inference.visualization import draw_prediction, update_trail
from tracknet.models import build_model
from tracknet.papers import get_paper_spec
from tracknet.training.checkpoint import load_checkpoint, load_model_weights
from tracknet.utils.device import select_device
from tracknet.utils.io import ensure_dir


@dataclass(frozen=True)
class VideoPredictionConfig:
    video_path: Path
    checkpoint_path: Path
    output_csv: Path
    output_video: Path | None = None
    model: dict[str, Any] | None = None
    target_width: int = 512
    target_height: int = 288
    sequence_length: int = 3
    target_frame_mode: str = "all"
    include_background: bool = False
    threshold: float = 0.5
    hough_threshold: int = 128
    batch_size: int = 4
    device: str = "auto"
    frame_index_base: int = 0
    trail_length: int = 12
    rectifier_checkpoint_path: Path | None = None
    rectifier_model: dict[str, Any] | None = None
    rectifier_sequence_length: int = 16
    rectifier_delta_y_pixels: float = 30.0
    num_threads: int | None = None
    interop_threads: int | None = None


def _read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    h, w = frames[0].shape[:2]
    return frames, fps, w, h


def _estimate_background(model_frames_rgb: list[np.ndarray]) -> np.ndarray:
    # A whole-video median is robust enough for inference without raw match
    # context. V3 training uses match/rally medians; this is the closest
    # self-contained estimate when only one input video is provided.
    stack = np.stack(model_frames_rgb, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _model_config_from_checkpoint(checkpoint_path: Path, explicit_model: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if explicit_model is not None:
        return explicit_model, ckpt
    cfg = ckpt.get("config") or {}
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise KeyError("Checkpoint does not contain config.model; pass --model-version/--model-kwargs via a config file")
    return dict(cfg["model"]), ckpt


def _window_indices_for_frame(frame_index: int, n_frames: int, sequence_length: int) -> list[int]:
    # V1 predicts the last frame. Prefix padding preserves early frames instead
    # of silently dropping them.
    start = frame_index - sequence_length + 1
    return [min(max(i, 0), n_frames - 1) for i in range(start, frame_index + 1)]


def _sliding_window_indices(n_frames: int, sequence_length: int) -> list[list[int]]:
    if n_frames < sequence_length:
        return [_window_indices_for_frame(i, n_frames, sequence_length) for i in range(n_frames)]
    return [list(range(start, start + sequence_length)) for start in range(0, n_frames - sequence_length + 1)]


def _build_input_tensor(
    frames_rgb_model: list[np.ndarray],
    indices: list[int],
    *,
    background_rgb_model: np.ndarray | None,
) -> torch.Tensor:
    tensors = [torch.from_numpy(image_to_tensor_chw_uint8_rgb(frames_rgb_model[i])) for i in indices]
    frames_t = torch.stack(tensors, dim=0)
    parts = [frames_t.reshape(len(indices) * 3, frames_t.shape[-2], frames_t.shape[-1])]
    if background_rgb_model is not None:
        parts.append(torch.from_numpy(image_to_tensor_chw_uint8_rgb(background_rgb_model)))
    return torch.cat(parts, dim=0)


def _weighted_average_heatmaps(items: list[tuple[np.ndarray, float]]) -> np.ndarray:
    return weighted_average_heatmaps(items)


def _aggregate_outputs(
    model: torch.nn.Module,
    windows: list[list[int]],
    *,
    frames_rgb_model: list[np.ndarray],
    background_rgb_model: np.ndarray | None,
    cfg: VideoPredictionConfig,
    postprocess_kind: PostprocessKind,
    device: torch.device,
) -> list[Prediction]:
    n_frames = max(max(w) for w in windows) + 1
    all_outputs: list[torch.Tensor] = []
    batched_windows: list[list[int]] = []
    for start in range(0, len(windows), cfg.batch_size):
        batch_windows = windows[start : start + cfg.batch_size]
        batch_inputs = [_build_input_tensor(frames_rgb_model, w, background_rgb_model=background_rgb_model) for w in batch_windows]
        batch = torch.stack(batch_inputs, dim=0).to(device)
        with torch.no_grad():
            out = model(batch).detach().cpu()
        all_outputs.append(out)
        batched_windows.extend(batch_windows[: len(out)])
    aggregated = aggregate_window_outputs(
        torch.cat(all_outputs, dim=0),
        batched_windows,
        postprocess_kind=postprocess_kind,
        sequence_length=cfg.sequence_length,
        target_frame_mode=cfg.target_frame_mode,
        threshold=cfg.threshold,
        hough_threshold=cfg.hough_threshold,
    )
    by_frame = {item.frame: item.prediction for item in aggregated}
    preds: list[Prediction] = []
    for i in range(n_frames):
        preds.append(by_frame.get(i, Prediction(visibility=0, x=-1.0, y=-1.0, score=0.0)))
    return preds


def _map_predictions_to_raw(predictions: list[Prediction], transform: LetterboxTransform) -> list[Prediction]:
    out: list[Prediction] = []
    for p in predictions:
        if p.visibility != 1 or p.x < 0 or p.y < 0:
            out.append(Prediction(visibility=0, x=-1.0, y=-1.0, score=p.score))
            continue
        x_raw, y_raw = transform.model_to_raw(p.x, p.y)
        x_raw = float(np.clip(x_raw, 0, transform.original_width - 1))
        y_raw = float(np.clip(y_raw, 0, transform.original_height - 1))
        out.append(Prediction(visibility=1, x=x_raw, y=y_raw, score=p.score))
    return out


def run_video_prediction(cfg: VideoPredictionConfig) -> pd.DataFrame:
    if cfg.num_threads is not None:
        torch.set_num_threads(int(cfg.num_threads))
    if cfg.interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(cfg.interop_threads))
        except RuntimeError:
            pass
    frames_bgr, fps, raw_w, raw_h = _read_video_frames(Path(cfg.video_path))
    frames_rgb_model: list[np.ndarray] = []
    transform: LetterboxTransform | None = None
    for frame in frames_bgr:
        rgb, t = letterbox_image(frame, cfg.target_width, cfg.target_height)
        frames_rgb_model.append(rgb)
        if transform is None:
            transform = t
    assert transform is not None
    if raw_w != transform.original_width or raw_h != transform.original_height:
        raise RuntimeError("Video frame size changed during decoding; variable-resolution videos are not supported")

    model_cfg, ckpt = _model_config_from_checkpoint(Path(cfg.checkpoint_path), cfg.model)
    model_version = str(model_cfg.get("version", model_cfg.get("model_version", "v2"))).lower()
    paper_spec = get_paper_spec(model_version)
    model = build_model(model_cfg)
    load_model_weights(model, ckpt, strict=True)
    device = select_device(cfg.device)
    model.to(device).eval()

    background = _estimate_background(frames_rgb_model) if cfg.include_background else None
    if paper_spec.postprocess_kind == "v1_hough" or cfg.target_frame_mode == "last":
        windows = [_window_indices_for_frame(i, len(frames_rgb_model), cfg.sequence_length) for i in range(len(frames_rgb_model))]
    else:
        windows = _sliding_window_indices(len(frames_rgb_model), cfg.sequence_length)
    model_preds = _aggregate_outputs(
        model,
        windows,
        frames_rgb_model=frames_rgb_model,
        background_rgb_model=background,
        cfg=cfg,
        postprocess_kind=paper_spec.postprocess_kind,
        device=device,
    )
    raw_preds = _map_predictions_to_raw(model_preds[: len(frames_bgr)], transform)
    if cfg.rectifier_checkpoint_path is not None:
        raw_preds = rectify_predictions(
            raw_preds,
            raw_width=raw_w,
            raw_height=raw_h,
            cfg=RectificationConfig(
                checkpoint_path=Path(cfg.rectifier_checkpoint_path),
                model=cfg.rectifier_model,
                trajectory_length=int(cfg.rectifier_sequence_length),
                delta_y_pixels=float(cfg.rectifier_delta_y_pixels),
                device=cfg.device,
            ),
        )

    rows = []
    for i, p in enumerate(raw_preds):
        rows.append(
            {
                RAW_FRAME_COL: int(i + cfg.frame_index_base),
                RAW_VISIBILITY_COL: int(p.visibility),
                RAW_X_COL: int(round(p.x)) if p.visibility == 1 else -1,
                RAW_Y_COL: int(round(p.y)) if p.visibility == 1 else -1,
            }
        )
    df = pd.DataFrame(rows, columns=CSV_OUTPUT_COLUMNS)
    ensure_dir(Path(cfg.output_csv).parent)
    df.to_csv(cfg.output_csv, index=False)

    if cfg.output_video is not None:
        ensure_dir(Path(cfg.output_video).parent)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(cfg.output_video), fourcc, fps, (raw_w, raw_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {cfg.output_video}")
        trail: deque[tuple[int, int]] = deque()
        for i, frame in enumerate(frames_bgr):
            update_trail(trail, raw_preds[i], cfg.trail_length)
            writer.write(draw_prediction(frame, raw_preds[i], frame_index=i + cfg.frame_index_base, trail=trail))
        writer.release()
    return df
