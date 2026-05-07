"""Evaluation loop using the same model, heatmap and post-processing semantics as inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from tracknet.inference.postprocess import Prediction
from tracknet.papers import get_paper_spec, paper_id_from_model_config
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
    threshold: float = 0.5
    hough_threshold: int = 128
    tolerance_pixels: float = 4.0
    num_threads: int | None = None
    interop_threads: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    predictions_csv: Path
    metrics_json: Path


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
    rows: list[dict[str, Any]] = []
    sequence_outputs: dict[str, list[torch.Tensor]] = {}
    sequence_metas: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True).float()
            outputs = model(x).detach().cpu()
            for sample_idx, sample_meta in enumerate(batch["meta"]):
                sequence_id = str(sample_meta["sequence_id"])
                sequence_outputs.setdefault(sequence_id, []).append(outputs[sample_idx : sample_idx + 1])
                sequence_metas.setdefault(sequence_id, []).append(sample_meta)
    for sequence_id, sample_metas in sequence_metas.items():
        outputs = torch.cat(sequence_outputs[sequence_id], dim=0)
        windows = [_window_frames(meta) for meta in sample_metas]
        frame_predictions = paper_spec.aggregate_window_outputs(
            outputs,
            windows,
            sequence_length=int(cfg.dataset.get("sequence_length", model_cfg.get("sequence_length", 3))),
            threshold=float(cfg.threshold),
            hough_threshold=int(cfg.hough_threshold),
        )
        gt_by_frame = _ground_truth_by_frame(sample_metas)
        for frame_prediction in frame_predictions:
            key = (sequence_id, int(frame_prediction.frame))
            gt = gt_by_frame.get(key)
            if gt is None:
                continue
            pred = frame_prediction.prediction
            outcome = classify_prediction(pred.coordinate, gt.coordinate, tolerance=float(cfg.tolerance_pixels))
            counts.add(outcome)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame": int(frame_prediction.frame),
                    "gt_visibility": int(gt.visibility),
                    "gt_x_model": float(gt.x),
                    "gt_y_model": float(gt.y),
                    "pred_visibility": int(pred.visibility),
                    "pred_x_model": float(pred.x),
                    "pred_y_model": float(pred.y),
                    "score": float(pred.score),
                    "outcome": outcome,
                }
            )
    metrics = counts.metrics()
    predictions_csv = cfg.output_dir / "predictions.csv"
    metrics_json = cfg.output_dir / "metrics.json"
    pd.DataFrame(rows).to_csv(predictions_csv, index=False)
    write_json(metrics_json, metrics)
    return EvaluationResult(metrics=metrics, predictions_csv=predictions_csv, metrics_json=metrics_json)
