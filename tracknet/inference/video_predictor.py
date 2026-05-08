"""Video inference with frame-accurate CSV and optional visualization output."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from tracknet.constants import CSV_OUTPUT_COLUMNS, RAW_FRAME_COL, RAW_VISIBILITY_COL, RAW_X_COL, RAW_Y_COL
from tracknet.data.transforms import LetterboxTransform, image_to_tensor_chw_uint8_rgb, letterbox_image
from tracknet.inference.aggregation import FramePrediction, weighted_average_heatmaps
from tracknet.inference.postprocess import Prediction
from tracknet.inference.rectification import RectificationConfig, rectify_predictions
from tracknet.inference.visualization import draw_prediction, update_trail
from tracknet.papers import get_paper_spec, paper_id_from_model_config
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
    background_sample_stride: int = 8
    max_background_samples: int = 512
    num_threads: int | None = None
    interop_threads: int | None = None


def _open_video(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def _video_metadata(video_path: Path) -> tuple[int, float, int, int]:
    cap = _open_video(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0 or raw_w <= 0 or raw_h <= 0:
        frame_count = 0
        raw_w = raw_h = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if raw_w == 0 or raw_h == 0:
                raw_h, raw_w = frame.shape[:2]
    cap.release()
    if frame_count <= 0 or raw_w <= 0 or raw_h <= 0:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    return frame_count, fps, raw_w, raw_h


def _iter_video_frames(video_path: Path):
    cap = _open_video(video_path)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield index, frame
        index += 1
    cap.release()


def _estimate_background(model_frames_rgb: list[np.ndarray]) -> np.ndarray:
    # A whole-video median is robust enough for inference without raw match
    # context. V3 training uses match/rally medians; this is the closest
    # self-contained estimate when only one input video is provided.
    stack = np.stack(model_frames_rgb, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _sample_background(video_path: Path, cfg: VideoPredictionConfig) -> tuple[np.ndarray, LetterboxTransform]:
    samples: list[np.ndarray] = []
    transform: LetterboxTransform | None = None
    stride = max(1, int(cfg.background_sample_stride))
    max_samples = max(1, int(cfg.max_background_samples))
    for frame_index, frame in _iter_video_frames(video_path):
        if frame_index % stride != 0:
            continue
        rgb, t = letterbox_image(frame, cfg.target_width, cfg.target_height)
        transform = t if transform is None else transform
        samples.append(rgb)
        if len(samples) >= max_samples:
            break
    if transform is None or not samples:
        raise ValueError(f"Could not sample frames for background estimation: {video_path}")
    return _estimate_background(samples), transform


def _model_config_from_checkpoint(checkpoint_path: Path, explicit_model: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if explicit_model is not None:
        return explicit_model, ckpt
    cfg = ckpt.get("config") or {}
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise KeyError("Checkpoint does not contain config.model; pass --model-version/--model-kwargs via a config file")
    return dict(cfg["model"]), ckpt


def _sliding_window_indices(n_frames: int, sequence_length: int) -> list[list[int]]:
    from tracknet.inference.windowing import sliding_windows

    return sliding_windows(n_frames, sequence_length)


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


class StreamingWindowAggregator:
    """Incrementally reduce window outputs to frame-level predictions.

    Evaluation can aggregate a whole sequence at once, but video inference must
    not keep every frame or every window output resident. This reducer preserves
    the same center-weighted heatmap policy while finalizing frames once no
    later sliding window can still contribute to them.
    """

    def __init__(
        self,
        *,
        paper_spec: Any,
        sequence_length: int,
        threshold: float,
        hough_threshold: int,
    ):
        self.paper_spec = paper_spec
        self.sequence_length = int(sequence_length)
        self.threshold = float(threshold)
        self.hough_threshold = int(hough_threshold)
        self._pending_heatmaps: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
        self._last_predictions: dict[int, Prediction] = {}
        self._finalized: dict[int, Prediction] = {}

    def add_batch(self, outputs: torch.Tensor, windows: list[list[int]]) -> None:
        if self.paper_spec.window_aggregation == "last":
            for item in self.paper_spec.aggregate_window_outputs(
                outputs,
                windows,
                sequence_length=self.sequence_length,
                threshold=self.threshold,
                hough_threshold=self.hough_threshold,
            ):
                self._last_predictions[int(item.frame)] = item.prediction
            return

        center = (self.sequence_length - 1) / 2.0
        sigma = max(1.0, self.sequence_length / 4.0)
        for sample, window in zip(outputs, windows):
            if sample.ndim == 2:
                sample = sample.unsqueeze(0)
            for local_idx, frame_id in enumerate(window):
                if local_idx >= sample.shape[0]:
                    continue
                weight = float(np.exp(-((local_idx - center) ** 2) / (2.0 * sigma * sigma)))
                self._pending_heatmaps[int(frame_id)].append((sample[local_idx].detach().cpu().numpy().astype(np.float32), weight))

    def finalize_until(self, frame_exclusive: int) -> list[FramePrediction]:
        ready: list[FramePrediction] = []
        for frame_id in sorted([frame for frame in self._pending_heatmaps if frame < frame_exclusive]):
            ready.append(self._finalize_heatmap_frame(frame_id))
        for frame_id in sorted([frame for frame in self._last_predictions if frame < frame_exclusive]):
            ready.append(FramePrediction(frame_id, self._last_predictions.pop(frame_id)))
        return sorted(ready, key=lambda item: item.frame)

    def finalize_all(self) -> list[FramePrediction]:
        ready: list[FramePrediction] = []
        for frame_id in sorted(list(self._pending_heatmaps)):
            ready.append(self._finalize_heatmap_frame(frame_id))
        for frame_id in sorted(list(self._last_predictions)):
            ready.append(FramePrediction(frame_id, self._last_predictions.pop(frame_id)))
        return sorted(ready, key=lambda item: item.frame)

    def _finalize_heatmap_frame(self, frame_id: int) -> FramePrediction:
        from tracknet.inference.postprocess import decode_model_output

        merged = weighted_average_heatmaps(self._pending_heatmaps.pop(frame_id))
        prediction = decode_model_output(
            torch.from_numpy(merged),
            postprocess_kind=self.paper_spec.postprocess_kind,
            threshold=self.threshold,
            hough_threshold=self.hough_threshold,
        )[0]
        self._finalized[frame_id] = prediction
        return FramePrediction(frame_id, prediction)


def _predict_streaming(
    model: torch.nn.Module,
    *,
    video_path: Path,
    frame_count: int,
    background_rgb_model: np.ndarray | None,
    cfg: VideoPredictionConfig,
    paper_spec: Any,
    device: torch.device,
) -> list[Prediction]:
    windows = paper_spec.video_windows(frame_count, cfg.sequence_length)
    if not windows:
        return []

    max_buffer = int(cfg.sequence_length)
    frame_buffer: deque[tuple[int, np.ndarray]] = deque()
    batch_windows: list[list[int]] = []
    batch_inputs: list[torch.Tensor] = []
    aggregator = StreamingWindowAggregator(
        paper_spec=paper_spec,
        sequence_length=cfg.sequence_length,
        threshold=cfg.threshold,
        hough_threshold=cfg.hough_threshold,
    )
    predictions_by_frame: dict[int, Prediction] = {}
    window_cursor = 0

    def buffered_frame(frame_id: int) -> np.ndarray:
        for buffered_id, buffered_rgb in frame_buffer:
            if buffered_id == frame_id:
                return buffered_rgb
        raise RuntimeError(f"Window requested frame {frame_id}, but it is no longer buffered")

    def flush_batch() -> None:
        if not batch_inputs:
            return
        batch = torch.stack(batch_inputs, dim=0).to(device)
        with torch.no_grad():
            out = model(batch).detach().cpu()
        aggregator.add_batch(out, batch_windows[: len(out)])
        if batch_windows:
            next_start = batch_windows[-1][0] + 1
            for item in aggregator.finalize_until(next_start):
                predictions_by_frame[int(item.frame)] = item.prediction
        batch_windows.clear()
        batch_inputs.clear()

    for frame_index, frame in _iter_video_frames(video_path):
        rgb, _ = letterbox_image(frame, cfg.target_width, cfg.target_height)
        frame_buffer.append((frame_index, rgb))
        while window_cursor < len(windows) and max(windows[window_cursor]) <= frame_index:
            window = windows[window_cursor]
            batch_windows.append(window)
            batch_inputs.append(_build_input_tensor([buffered_frame(i) for i in window], list(range(len(window))), background_rgb_model=background_rgb_model))
            window_cursor += 1
            if len(batch_inputs) >= int(cfg.batch_size):
                flush_batch()
        while len(frame_buffer) > max_buffer:
            frame_buffer.popleft()
    if window_cursor != len(windows):
        raise RuntimeError(f"Video ended before all windows were processed: {window_cursor}/{len(windows)}")
    flush_batch()
    for item in aggregator.finalize_all():
        predictions_by_frame[int(item.frame)] = item.prediction
    return [predictions_by_frame.get(i, Prediction(visibility=0, x=-1.0, y=-1.0, score=0.0)) for i in range(frame_count)]


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
    model_cfg, ckpt = _model_config_from_checkpoint(Path(cfg.checkpoint_path), cfg.model)
    paper_spec = get_paper_spec(paper_id_from_model_config(model_cfg))
    frame_count, fps, raw_w, raw_h = _video_metadata(Path(cfg.video_path))
    transform = LetterboxTransform(
        original_width=raw_w,
        original_height=raw_h,
        target_width=cfg.target_width,
        target_height=cfg.target_height,
        scale=min(cfg.target_width / raw_w, cfg.target_height / raw_h),
        pad_x=(cfg.target_width - int(round(raw_w * min(cfg.target_width / raw_w, cfg.target_height / raw_h)))) // 2,
        pad_y=(cfg.target_height - int(round(raw_h * min(cfg.target_width / raw_w, cfg.target_height / raw_h)))) // 2,
        resized_width=int(round(raw_w * min(cfg.target_width / raw_w, cfg.target_height / raw_h))),
        resized_height=int(round(raw_h * min(cfg.target_width / raw_w, cfg.target_height / raw_h))),
    )
    model = paper_spec.build_model(model_cfg)
    load_model_weights(model, ckpt, strict=True)
    device = select_device(cfg.device)
    model.to(device).eval()

    background = None
    if paper_spec.target_policy is not None and paper_spec.target_policy.include_background:
        background, transform = _sample_background(Path(cfg.video_path), cfg)
    model_preds = _predict_streaming(
        model,
        video_path=Path(cfg.video_path),
        frame_count=frame_count,
        background_rgb_model=background,
        cfg=cfg,
        paper_spec=paper_spec,
        device=device,
    )
    raw_preds = _map_predictions_to_raw(model_preds[:frame_count], transform)
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
        for i, frame in _iter_video_frames(Path(cfg.video_path)):
            update_trail(trail, raw_preds[i], cfg.trail_length)
            writer.write(draw_prediction(frame, raw_preds[i], frame_index=i + cfg.frame_index_base, trail=trail))
        writer.release()
    return df
