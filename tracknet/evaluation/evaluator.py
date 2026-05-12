"""Evaluation loop using the same model, heatmap and post-processing semantics as inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracknet.data.transforms import LetterboxTransform
from tracknet.inference.aggregation import FramePrediction
from tracknet.inference.postprocess import Prediction
from tracknet.inference.rectification import RectificationConfig, rectify_predictions
from tracknet.papers import get_paper_spec, paper_id_from_model_config
from tracknet.papers.base import CoordinateSpace, EvaluationProtocol, PaperSpec
from tracknet.training.checkpoint import load_checkpoint, load_model_weights
from tracknet.training.metrics import ConfusionCounts, classify_prediction
from tracknet.utils.device import select_device
from tracknet.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: Path
    output_dir: Path
    dataset: dict[str, Any]
    model: dict[str, Any] | None = None
    batch_size: int = 4
    workers: int = 2
    device: str = "auto"
    threshold: float | None = None
    hough_threshold: int | None = None
    tolerance_pixels: float | None = None
    coordinate_space: CoordinateSpace | None = None
    rectifier_checkpoint_path: Path | None = None
    rectifier_model: dict[str, Any] | None = None
    rectifier_sequence_length: int = 16
    rectifier_delta_y_pixels: float = 30.0
    num_threads: int | None = None
    interop_threads: int | None = None
    progress: bool = True


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    predictions_csv: Path
    metrics_json: Path
    sequence_metrics_json: Path
    protocol_json: Path
    resolved_config_json: Path
    checkpoint_json: Path


def _collate_eval(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": torch.stack([s["input"] for s in samples], dim=0),
        "target": torch.stack([s["target"] for s in samples], dim=0) if isinstance(samples[0]["target"], torch.Tensor) else [s["target"] for s in samples],
        "meta": samples,
    }


def _model_config_from_checkpoint(checkpoint_path: Path, explicit_model: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if explicit_model is not None:
        return explicit_model, ckpt
    cfg = ckpt.get("config") or {}
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise KeyError("Checkpoint does not contain config.model; provide evaluation.model in the config")
    return dict(cfg["model"]), ckpt


def _resolved_dataset_config(dataset_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    return get_paper_spec(paper_id_from_model_config(model_cfg)).resolved_dataset_config(dataset_cfg)


def _target_infos(sample_meta: dict[str, Any]) -> list[dict[str, Any]]:
    infos = sample_meta["target_info"]
    if isinstance(infos, list):
        return infos
    raise TypeError(f"Unexpected target_info type: {type(infos).__name__}")


def _gt_prediction(info: dict[str, Any]) -> Prediction:
    vis = int(info["visibility"])
    if vis != 1 or float(info["x_model"]) < 0 or float(info["y_model"]) < 0:
        return Prediction(visibility=0, x=-1.0, y=-1.0, score=1.0)
    return Prediction(visibility=1, x=float(info["x_model"]), y=float(info["y_model"]), score=1.0)


def _gt_raw_prediction(info: dict[str, Any]) -> Prediction:
    vis = int(info["visibility"])
    if vis != 1 or float(info["x_raw"]) < 0 or float(info["y_raw"]) < 0:
        return Prediction(visibility=0, x=-1.0, y=-1.0, score=1.0)
    return Prediction(visibility=1, x=float(info["x_raw"]), y=float(info["y_raw"]), score=1.0)


def _window_frames(sample_meta: dict[str, Any]) -> list[int]:
    infos = _target_infos(sample_meta)
    by_local_index = {int(info["local_index"]): int(info["frame"]) for info in infos}
    if len(by_local_index) == len(sample_meta["frame_indices"]):
        return [by_local_index[i] for i in range(len(by_local_index))]
    return [int(v) for v in sample_meta["frame_indices"]]


def _ground_truth_by_frame(samples: list[dict[str, Any]]) -> dict[tuple[str, int], Prediction]:
    gt: dict[tuple[str, int], Prediction] = {}
    for sample in samples:
        for info in _target_infos(sample):
            key = (str(sample["sequence_id"]), int(info["frame"]))
            gt[key] = _gt_prediction(info)
    return gt


def _ground_truth_raw_by_frame(samples: list[dict[str, Any]]) -> dict[tuple[str, int], Prediction]:
    gt: dict[tuple[str, int], Prediction] = {}
    for sample in samples:
        for info in _target_infos(sample):
            key = (str(sample["sequence_id"]), int(info["frame"]))
            gt[key] = _gt_raw_prediction(info)
    return gt


def _sequence_transform(sample_meta: dict[str, Any]) -> LetterboxTransform:
    sequence_meta = sample_meta.get("sequence_meta")
    if not isinstance(sequence_meta, dict) or not isinstance(sequence_meta.get("transform"), dict):
        raise KeyError("Evaluation requires sequence_meta.transform for raw-coordinate metrics")
    return LetterboxTransform.from_dict(sequence_meta["transform"])


def _prediction_to_raw(prediction: Prediction, transform: LetterboxTransform) -> Prediction:
    if prediction.visibility != 1 or prediction.x < 0 or prediction.y < 0:
        return Prediction(visibility=0, x=-1.0, y=-1.0, score=prediction.score)
    x_raw, y_raw = transform.model_to_raw(prediction.x, prediction.y)
    return Prediction(visibility=1, x=float(x_raw), y=float(y_raw), score=prediction.score)


def _prediction_to_model(prediction: Prediction, transform: LetterboxTransform) -> Prediction:
    if prediction.visibility != 1 or prediction.x < 0 or prediction.y < 0:
        return Prediction(visibility=0, x=-1.0, y=-1.0, score=prediction.score)
    x_model, y_model = transform.raw_to_model(prediction.x, prediction.y)
    x_model, y_model = transform.clamp_model(x_model, y_model)
    return Prediction(visibility=1, x=float(x_model), y=float(y_model), score=prediction.score)


def _prediction_for_space(model_prediction: Prediction, raw_prediction: Prediction, coordinate_space: CoordinateSpace) -> Prediction:
    if coordinate_space == "model":
        return model_prediction
    if coordinate_space == "raw":
        return raw_prediction
    raise ValueError(f"Unsupported coordinate space: {coordinate_space}")


def _counts_metrics(counts: ConfusionCounts) -> dict[str, float]:
    return counts.metrics()


def resolve_evaluation_protocol(cfg: EvaluationConfig, model_cfg: dict[str, Any] | None = None) -> EvaluationProtocol:
    resolved_model = model_cfg or cfg.model
    if resolved_model is None:
        raise ValueError("model_cfg is required when EvaluationConfig.model is not provided")
    paper_spec = get_paper_spec(paper_id_from_model_config(resolved_model))
    return paper_spec.evaluation_protocol.with_overrides(
        threshold=cfg.threshold,
        hough_threshold=cfg.hough_threshold,
        tolerance_pixels=cfg.tolerance_pixels,
        coordinate_space=cfg.coordinate_space,
        supports_rectifier=True if cfg.rectifier_checkpoint_path is not None else None,
    )


def _checkpoint_summary(checkpoint_path: Path, ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "format_version": ckpt.get("format_version"),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "best_score": ckpt.get("best_score"),
        "metrics": ckpt.get("metrics", {}),
    }


def _resolved_config_summary(cfg: EvaluationConfig, model_cfg: dict[str, Any], dataset_cfg: dict[str, Any], protocol: EvaluationProtocol) -> dict[str, Any]:
    return {
        "checkpoint_path": str(cfg.checkpoint_path),
        "output_dir": str(cfg.output_dir),
        "model": model_cfg,
        "dataset": dataset_cfg,
        "batch_size": int(cfg.batch_size),
        "workers": int(cfg.workers),
        "device": cfg.device,
        "progress": bool(cfg.progress),
        "protocol": protocol.as_dict(),
        "rectifier": {
            "checkpoint_path": str(cfg.rectifier_checkpoint_path) if cfg.rectifier_checkpoint_path is not None else None,
            "model": cfg.rectifier_model,
            "trajectory_length": int(cfg.rectifier_sequence_length),
            "delta_y_pixels": float(cfg.rectifier_delta_y_pixels),
        },
    }


@dataclass
class _EvaluationAccumulator:
    protocol: EvaluationProtocol
    counts: ConfusionCounts
    sequence_counts: dict[str, ConfusionCounts]
    rows: list[dict[str, Any]]


def _record_frame_predictions(
    *,
    accumulator: _EvaluationAccumulator,
    sequence_id: str,
    sample_metas: list[dict[str, Any]],
    frame_predictions: list[FramePrediction],
    rectifier_cfg: RectificationConfig | None,
) -> None:
    if not frame_predictions:
        return
    transform = _sequence_transform(sample_metas[0])
    model_predictions_by_frame = {int(item.frame): item.prediction for item in frame_predictions}
    raw_predictions_by_frame = {frame: _prediction_to_raw(prediction, transform) for frame, prediction in model_predictions_by_frame.items()}
    if rectifier_cfg is not None:
        ordered_frames = sorted(raw_predictions_by_frame)
        repaired = rectify_predictions(
            [raw_predictions_by_frame[frame] for frame in ordered_frames],
            raw_width=int(transform.original_width),
            raw_height=int(transform.original_height),
            cfg=rectifier_cfg,
        )
        raw_predictions_by_frame = {frame: pred for frame, pred in zip(ordered_frames, repaired, strict=True)}
        model_predictions_by_frame = {frame: _prediction_to_model(pred, transform) for frame, pred in raw_predictions_by_frame.items()}

    gt_by_frame = _ground_truth_by_frame(sample_metas)
    gt_raw_by_frame = _ground_truth_raw_by_frame(sample_metas)
    sequence_count = accumulator.sequence_counts.setdefault(sequence_id, ConfusionCounts())
    for frame_prediction in frame_predictions:
        key = (sequence_id, int(frame_prediction.frame))
        gt = gt_by_frame.get(key)
        gt_raw = gt_raw_by_frame.get(key)
        if gt is None or gt_raw is None:
            continue
        pred = model_predictions_by_frame[int(frame_prediction.frame)]
        pred_raw = raw_predictions_by_frame[int(frame_prediction.frame)]
        scored_pred = _prediction_for_space(pred, pred_raw, accumulator.protocol.coordinate_space)
        scored_gt = _prediction_for_space(gt, gt_raw, accumulator.protocol.coordinate_space)
        outcome = classify_prediction(scored_pred.coordinate, scored_gt.coordinate, tolerance=float(accumulator.protocol.tolerance_pixels))
        accumulator.counts.add(outcome)
        sequence_count.add(outcome)
        accumulator.rows.append(
            {
                "sequence_id": sequence_id,
                "frame": int(frame_prediction.frame),
                "gt_visibility": int(gt.visibility),
                "gt_x_model": float(gt.x),
                "gt_y_model": float(gt.y),
                "gt_x_raw": float(gt_raw.x),
                "gt_y_raw": float(gt_raw.y),
                "pred_visibility": int(pred.visibility),
                "pred_x_model": float(pred.x),
                "pred_y_model": float(pred.y),
                "pred_x_raw": float(pred_raw.x),
                "pred_y_raw": float(pred_raw.y),
                "score": float(pred.score),
                "coordinate_space": accumulator.protocol.coordinate_space,
                "outcome": outcome,
            }
        )


def _rectifier_config(cfg: EvaluationConfig) -> RectificationConfig | None:
    if cfg.rectifier_checkpoint_path is None:
        return None
    return RectificationConfig(
        checkpoint_path=Path(cfg.rectifier_checkpoint_path),
        model=cfg.rectifier_model,
        trajectory_length=int(cfg.rectifier_sequence_length),
        delta_y_pixels=float(cfg.rectifier_delta_y_pixels),
        device=cfg.device,
    )


def evaluate_checkpoint(cfg: EvaluationConfig) -> EvaluationResult:
    if cfg.num_threads is not None:
        torch.set_num_threads(int(cfg.num_threads))
    if cfg.interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(cfg.interop_threads))
        except RuntimeError:
            pass
    ensure_dir(cfg.output_dir)
    model_cfg, ckpt = _model_config_from_checkpoint(Path(cfg.checkpoint_path), cfg.model)
    paper_spec = get_paper_spec(paper_id_from_model_config(model_cfg))
    protocol = resolve_evaluation_protocol(cfg, model_cfg)
    device = select_device(cfg.device)
    model = paper_spec.build_model(model_cfg)
    load_model_weights(model, ckpt, strict=True)
    model.to(device).eval()

    resolved_dataset_cfg = _resolved_dataset_config(cfg.dataset, model_cfg)
    ds = paper_spec.build_heatmap_dataset(resolved_dataset_cfg)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_eval,
    )
    counts = ConfusionCounts()
    sequence_counts: dict[str, ConfusionCounts] = {}
    rows: list[dict[str, Any]] = []
    accumulator = _EvaluationAccumulator(protocol=protocol, counts=counts, sequence_counts=sequence_counts, rows=rows)
    sequence_batches: dict[str, list[tuple[torch.Tensor, dict[str, Any]]]] = {}
    rectifier_cfg = _rectifier_config(cfg)

    def flush_sequence(sequence_id: str) -> None:
        items = sequence_batches.pop(sequence_id, [])
        if not items:
            return
        outputs = torch.cat([output for output, _ in items], dim=0)
        sample_metas = [meta for _, meta in items]
        windows = [_window_frames(meta) for meta in sample_metas]
        frame_predictions = paper_spec.aggregate_window_outputs(
            outputs,
            windows,
            sequence_length=int(resolved_dataset_cfg.get("sequence_length", model_cfg.get("sequence_length", 3))),
            threshold=float(protocol.threshold),
            hough_threshold=int(protocol.hough_threshold),
        )
        _record_frame_predictions(
            accumulator=accumulator,
            sequence_id=sequence_id,
            sample_metas=sample_metas,
            frame_predictions=frame_predictions,
            rectifier_cfg=rectifier_cfg,
        )

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", unit="batch", total=len(loader), dynamic_ncols=True, disable=not cfg.progress)
        for batch in progress_bar:
            x = batch["input"].to(device, non_blocking=True).float()
            outputs = model(x).detach().cpu()
            for sample_idx, sample_meta in enumerate(batch["meta"]):
                sequence_id = str(sample_meta["sequence_id"])
                if paper_spec.window_aggregation == "last":
                    frame_predictions = paper_spec.aggregate_window_outputs(
                        outputs[sample_idx : sample_idx + 1],
                        [_window_frames(sample_meta)],
                        sequence_length=int(resolved_dataset_cfg.get("sequence_length", model_cfg.get("sequence_length", 3))),
                        threshold=float(protocol.threshold),
                        hough_threshold=int(protocol.hough_threshold),
                    )
                    _record_frame_predictions(
                        accumulator=accumulator,
                        sequence_id=sequence_id,
                        sample_metas=[sample_meta],
                        frame_predictions=frame_predictions,
                        rectifier_cfg=rectifier_cfg,
                    )
                    continue
                active_ids = set(sequence_batches)
                if active_ids and sequence_id not in active_ids:
                    for old_sequence_id in sorted(active_ids):
                        flush_sequence(old_sequence_id)
                sequence_batches.setdefault(sequence_id, []).append((outputs[sample_idx : sample_idx + 1], sample_meta))
        progress_bar.close()
    for sequence_id in sorted(list(sequence_batches)):
        flush_sequence(sequence_id)
    metrics = counts.metrics()
    predictions_csv = cfg.output_dir / "predictions.csv"
    metrics_json = cfg.output_dir / "metrics.json"
    sequence_metrics_json = cfg.output_dir / "metrics.by_sequence.json"
    protocol_json = cfg.output_dir / "protocol.json"
    resolved_config_json = cfg.output_dir / "evaluation.resolved.json"
    checkpoint_json = cfg.output_dir / "checkpoint.json"
    pd.DataFrame(rows).to_csv(predictions_csv, index=False)
    write_json(metrics_json, metrics)
    write_json(sequence_metrics_json, {sequence_id: _counts_metrics(sequence_count) for sequence_id, sequence_count in sorted(sequence_counts.items())})
    write_json(protocol_json, protocol.as_dict())
    write_json(resolved_config_json, _resolved_config_summary(cfg, model_cfg, resolved_dataset_cfg, protocol))
    write_json(checkpoint_json, _checkpoint_summary(Path(cfg.checkpoint_path), ckpt))
    return EvaluationResult(
        metrics=metrics,
        predictions_csv=predictions_csv,
        metrics_json=metrics_json,
        sequence_metrics_json=sequence_metrics_json,
        protocol_json=protocol_json,
        resolved_config_json=resolved_config_json,
        checkpoint_json=checkpoint_json,
    )
