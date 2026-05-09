"""TensorBoard observability for training runs.

The trainer owns optimization. This module owns how training state is exposed
to TensorBoard so visual diagnostics, experiment metadata, and optional heavy
logging can evolve without turning the training loop into a dashboard script.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class TensorBoardRunLogger:
    """High-level TensorBoard writer for TrackNet training.

    Expensive diagnostics are opt-in and rate-limited. Scalars and run metadata
    stay cheap enough to keep enabled by default, while sample images,
    histograms, model graphs, and hparams can be controlled from YAML configs.
    """

    def __init__(
        self,
        *,
        writer: Any | None,
        cfg: dict[str, Any],
        output_dir: Path,
        is_main: bool,
        device: torch.device,
        split_metadata: dict[str, Any],
        hardware: dict[str, Any],
    ) -> None:
        self.writer = writer
        self.cfg = cfg
        self.output_dir = output_dir
        self.is_main = is_main
        self.device = device
        self.split_metadata = split_metadata
        self.hardware = hardware
        self.train_cfg = cfg.get("train", {})
        self.enabled = writer is not None and is_main
        self.image_enabled = bool(self.train_cfg.get("tensorboard_images", True))
        self.histogram_enabled = bool(self.train_cfg.get("tensorboard_histograms", False))
        self.graph_enabled = bool(self.train_cfg.get("tensorboard_model_graph", False))
        self.hparams_enabled = bool(self.train_cfg.get("tensorboard_hparams", True))
        self.histogram_interval = max(1, int(self.train_cfg.get("tensorboard_histogram_interval", 5)))
        self.image_interval = max(1, int(self.train_cfg.get("tensorboard_image_interval", 1)))
        self.max_images = max(1, int(self.train_cfg.get("tensorboard_max_images", 2)))
        self.log_every_steps = max(1, int(self.train_cfg.get("log_every_steps", 20)))
        self._loss_ema: float | None = None
        self._run_start = time.perf_counter()

    def log_run_metadata(self) -> None:
        if not self.enabled:
            return
        assert self.writer is not None
        self.writer.add_text("config/resolved", _json_block(self.cfg), 0)
        self.writer.add_text("config/hardware", _json_block(self.hardware), 0)
        self.writer.add_text("config/splits", _json_block(self.split_metadata), 0)
        self.writer.add_text("config/output_dir", str(self.output_dir), 0)

    def log_model_graph(self, model: torch.nn.Module, sample_batch: dict[str, Any]) -> None:
        if not self.enabled or not self.graph_enabled:
            return
        assert self.writer is not None
        try:
            model_for_graph = model.module if hasattr(model, "module") else model
            sample_input = sample_batch["input"][:1].to(self.device, non_blocking=True).float()
            self.writer.add_graph(model_for_graph, sample_input)
        except Exception as exc:  # pragma: no cover - graph tracing is model/backend dependent.
            self.writer.add_text("warnings/model_graph", f"Model graph export failed: {exc}", 0)

    def log_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        epoch_seconds: float,
        train_samples: int,
        model: torch.nn.Module,
        val_sample: dict[str, Any] | None,
    ) -> None:
        if not self.enabled:
            return
        assert self.writer is not None
        samples_per_second = train_samples / max(epoch_seconds, 1e-9)
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        self.writer.add_scalar("optim/lr", lr, epoch)
        self.writer.add_scalar("train/global_step", global_step, epoch)
        self.writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch)
        self.writer.add_scalar("train/samples_per_second", samples_per_second, epoch)
        self.writer.add_scalar("train/elapsed_seconds", time.perf_counter() - self._run_start, epoch)
        if val_loss < float("inf"):
            self.writer.add_scalar("checkpoint/best_score_candidate", val_loss, epoch)
        if self.histogram_enabled and epoch % self.histogram_interval == 0:
            self.log_parameter_histograms(model, epoch)
        if self.image_enabled and val_sample is not None and epoch % self.image_interval == 0:
            self.log_validation_samples(model, val_sample, epoch)
        self.writer.flush()

    def log_train_step(
        self,
        *,
        global_step: int,
        loss: float,
        lr: float,
        step_seconds: float,
        batch_size: int,
    ) -> None:
        if not self.enabled:
            return
        if global_step <= 1 or global_step % self.log_every_steps == 0:
            assert self.writer is not None
            ema_alpha = float(self.train_cfg.get("tensorboard_loss_ema_alpha", 0.1))
            self._loss_ema = loss if self._loss_ema is None else (ema_alpha * loss + (1.0 - ema_alpha) * self._loss_ema)
            self.writer.add_scalar("loss/train_step", loss, global_step)
            self.writer.add_scalar("loss/train_ema", self._loss_ema, global_step)
            self.writer.add_scalar("optim/lr_step", lr, global_step)
            self.writer.add_scalar("train/step_seconds", step_seconds, global_step)
            self.writer.add_scalar("train/samples_per_second_step", batch_size / max(step_seconds, 1e-9), global_step)

    def log_checkpoint(self, *, epoch: int, global_step: int, metrics: Mapping[str, float], is_best: bool) -> None:
        if not self.enabled:
            return
        assert self.writer is not None
        payload = {"epoch": epoch, "global_step": global_step, "is_best": is_best, "metrics": dict(metrics)}
        self.writer.add_text("checkpoint/latest", _json_block(payload), epoch)
        if is_best:
            self.writer.add_scalar("checkpoint/best_val_loss", float(metrics.get("val_loss", 0.0)), epoch)
            self.writer.add_text("checkpoint/best", _json_block(payload), epoch)

    def log_hparams(self, metrics: Mapping[str, float]) -> None:
        if not self.enabled or not self.hparams_enabled:
            return
        assert self.writer is not None
        hparams = _flatten_hparams({"model": self.cfg.get("model", {}), "train": self.train_cfg})
        numeric_metrics = {f"hparam/{key}": float(value) for key, value in metrics.items() if isinstance(value, int | float)}
        if not hparams or not numeric_metrics:
            return
        self.writer.add_hparams(hparams, numeric_metrics, run_name="hparams")

    def make_profiler(self) -> Any | None:
        if not self.enabled or not bool(self.train_cfg.get("tensorboard_profiler", False)):
            return None
        wait = int(self.train_cfg.get("tensorboard_profiler_wait", 1))
        warmup = int(self.train_cfg.get("tensorboard_profiler_warmup", 1))
        active = int(self.train_cfg.get("tensorboard_profiler_active", 2))
        repeat = int(self.train_cfg.get("tensorboard_profiler_repeat", 1))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile_dir = self.output_dir / "tensorboard" / "profile"
        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=bool(self.train_cfg.get("tensorboard_profiler_record_shapes", True)),
            profile_memory=bool(self.train_cfg.get("tensorboard_profiler_memory", True)),
            with_stack=bool(self.train_cfg.get("tensorboard_profiler_stack", False)),
        )

    def log_parameter_histograms(self, model: torch.nn.Module, epoch: int) -> None:
        if not self.enabled:
            return
        assert self.writer is not None
        model_for_logging = model.module if hasattr(model, "module") else model
        max_tensors = max(1, int(self.train_cfg.get("tensorboard_max_histograms", 12)))
        for index, (name, param) in enumerate(model_for_logging.named_parameters()):
            if index >= max_tensors:
                break
            clean_name = name.replace(".", "/")
            self.writer.add_histogram(f"parameters/{clean_name}", param.detach().float().cpu(), epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{clean_name}", param.grad.detach().float().cpu(), epoch)
                self.writer.add_scalar(f"grad_norm/{clean_name}", float(param.grad.detach().float().norm().cpu()), epoch)

    @torch.no_grad()
    def log_validation_samples(self, model: torch.nn.Module, batch: dict[str, Any], epoch: int) -> None:
        if not self.enabled:
            return
        assert self.writer is not None
        if "input" not in batch or "target" not in batch:
            return
        x = batch["input"][: self.max_images].to(self.device, non_blocking=True).float()
        target = batch["target"][: self.max_images].detach().cpu()
        was_training = model.training
        model.eval()
        output = model(x).detach().cpu()
        model.train(was_training)
        rgb = _extract_rgb_preview(x.detach().cpu())
        pred_heat = _prediction_preview(output)
        target_heat = _target_preview(target)
        for index in range(rgb.shape[0]):
            panel = _make_sample_panel(rgb[index], target_heat[index], pred_heat[index])
            self.writer.add_image(f"samples/validation_{index}", panel, epoch)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _flatten_hparams(value: Mapping[str, Any], prefix: str = "") -> dict[str, int | float | str | bool]:
    flat: dict[str, int | float | str | bool] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(_flatten_hparams(item, name))
        elif isinstance(item, bool | int | float | str):
            flat[name] = item
    return flat


def _extract_rgb_preview(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] < 3:
        return torch.zeros(max(1, x.shape[0] if x.ndim > 0 else 1), 3, 16, 16)
    rgb = x[:, :3].float().clamp(0.0, 1.0)
    return rgb


def _prediction_preview(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 4:
        if output.shape[1] == 256:
            heat = torch.argmax(output, dim=1).float() / 255.0
        else:
            heat = output.float().max(dim=1).values
        return _normalize_heatmap(heat)
    if output.ndim == 3:
        return _trajectory_preview(output)
    return torch.zeros(max(1, output.shape[0] if output.ndim > 0 else 1), 16, 16)


def _target_preview(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4:
        return _normalize_heatmap(target.float().max(dim=1).values)
    if target.ndim == 3 and target.shape[1] == 3:
        return _trajectory_preview(target[:, :2])
    if target.ndim == 3:
        return _normalize_heatmap(target.float())
    return torch.zeros(max(1, target.shape[0] if target.ndim > 0 else 1), 16, 16)


def _normalize_heatmap(heat: torch.Tensor) -> torch.Tensor:
    heat = heat.float()
    flat = heat.reshape(heat.shape[0], -1)
    min_v = flat.min(dim=1).values.reshape(-1, 1, 1)
    max_v = flat.max(dim=1).values.reshape(-1, 1, 1)
    return ((heat - min_v) / torch.clamp(max_v - min_v, min=1e-6)).clamp(0.0, 1.0)


def _trajectory_preview(coords: torch.Tensor, size: int = 128) -> torch.Tensor:
    batch = coords.shape[0]
    canvas = torch.zeros(batch, size, size)
    xy = coords[:, :2].detach().float().clamp(0.0, 1.0)
    for b in range(batch):
        for t in range(xy.shape[-1]):
            x = int(round(float(xy[b, 0, t]) * (size - 1)))
            y = int(round(float(xy[b, 1, t]) * (size - 1)))
            canvas[b, max(0, y - 1) : min(size, y + 2), max(0, x - 1) : min(size, x + 2)] = 1.0
    return canvas


def _make_sample_panel(rgb: torch.Tensor, target_heat: torch.Tensor, pred_heat: torch.Tensor) -> torch.Tensor:
    h, w = rgb.shape[-2:]
    target_rgb = _heatmap_to_rgb(_resize_heatmap(target_heat, h, w))
    pred_rgb = _heatmap_to_rgb(_resize_heatmap(pred_heat, h, w))
    overlay = (rgb * 0.65 + pred_rgb * 0.35).clamp(0.0, 1.0)
    return torch.cat([rgb, target_rgb, pred_rgb, overlay], dim=2)


def _resize_heatmap(heat: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if heat.shape[-2:] == (height, width):
        return heat.float()
    resized = F.interpolate(heat.reshape(1, 1, *heat.shape[-2:]).float(), size=(height, width), mode="bilinear", align_corners=False)
    return resized[0, 0]


def _heatmap_to_rgb(heat: torch.Tensor) -> torch.Tensor:
    heat = heat.float().clamp(0.0, 1.0)
    return torch.stack([heat, torch.sqrt(heat), 1.0 - heat], dim=0).clamp(0.0, 1.0)
