"""Training loop for TrackNet tracking models."""

from __future__ import annotations

import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from tracknet.papers import get_paper_spec, paper_id_from_model_config
from tracknet.papers.base import PaperSpec
from tracknet.training.checkpoint import CheckpointState, load_checkpoint, load_model_weights, save_checkpoint, save_model_only
from tracknet.training.losses import build_loss
from tracknet.training.tensorboard import TensorBoardRunLogger
from tracknet.utils.device import select_device
from tracknet.utils.hardware import collect_hardware_report
from tracknet.utils.io import ensure_dir, write_json
from tracknet.utils.seeding import seed_everything


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()



def _collate_train(samples: list[dict[str, Any]]) -> dict[str, Any]:
    # Training only needs tensors. Keeping metadata out of the default collate
    # avoids failures on optional fields such as background=None.
    return {
        "input": torch.stack([s["input"] for s in samples], dim=0),
        "target": torch.stack([s["target"] for s in samples], dim=0),
    }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _start_tensorboard_process(cmd: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)


@dataclass
class TrainResult:
    output_dir: Path
    best_loss: float
    last_epoch: int
    tensorboard_url: str | None = None


@dataclass
class TensorBoardServer:
    process: subprocess.Popen[str]
    url: str
    logdir: Path
    port: int


class TrackNetTrainer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.train_cfg = cfg.get("train", {})
        if "num_threads" in self.train_cfg:
            torch.set_num_threads(int(self.train_cfg["num_threads"]))
        if "interop_threads" in self.train_cfg:
            try:
                torch.set_num_interop_threads(int(self.train_cfg["interop_threads"]))
            except RuntimeError:
                pass
        self.distributed, self.rank, self.local_rank, self.world_size = setup_distributed()
        self.is_main = self.rank == 0
        seed_everything(int(self.train_cfg.get("seed", 26)), deterministic=bool(self.train_cfg.get("deterministic", False)))
        device_name = str(self.train_cfg.get("device", "auto"))
        if self.distributed and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = select_device(device_name)
        self.output_dir = self._resolve_output_dir()
        if self.is_main:
            ensure_dir(self.output_dir / "checkpoints")
            write_json(self.output_dir / "config.resolved.json", cfg)
        if self.distributed:
            dist.barrier()
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.amp_requested = bool(self.train_cfg.get("amp", False))
        self.amp_enabled = self.amp_requested and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.split_metadata: dict[str, Any] = {}
        self.paper_spec = self._resolve_paper_spec()
        self.tensorboard_enabled = bool(self.train_cfg.get("tensorboard", False))
        self.tensorboard_writer: Any | None = None
        self.tensorboard_server: TensorBoardServer | None = None

    def _resolve_paper_spec(self) -> PaperSpec:
        model_cfg = self.cfg.get("model", {})
        return get_paper_spec(paper_id_from_model_config(model_cfg))

    def _resolve_output_dir(self) -> Path:
        resume = self.train_cfg.get("resume")
        if resume:
            output_dir = Path(resume)
        else:
            out_root = Path(self.train_cfg.get("output_root", "outputs"))
            name = str(self.train_cfg.get("experiment_name", "tracknet"))
            from datetime import datetime

            output_dir = out_root / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.distributed:
            shared = [str(output_dir) if self.is_main else ""]
            dist.broadcast_object_list(shared, src=0)
            output_dir = Path(shared[0])
        return output_dir

    def _make_dataset(self) -> tuple[Any, Any]:
        dataset_section = self.cfg["dataset"]
        train_split_file = dataset_section.get("train_split_file")
        val_split_file = dataset_section.get("val_split_file")
        if bool(train_split_file) != bool(val_split_file):
            raise ValueError("Training requires both dataset.train_split_file and dataset.val_split_file")
        if train_split_file and val_split_file:
            train_cfg = dict(dataset_section)
            val_cfg = dict(dataset_section)
            train_cfg["split_file"] = train_split_file
            val_cfg["split_file"] = val_split_file
            train_ds = self.paper_spec.build_training_dataset(train_cfg)
            val_ds = self.paper_spec.build_training_dataset(val_cfg)
            train_ids = self._dataset_sequence_ids(train_ds)
            val_ids = self._dataset_sequence_ids(val_ds)
            if not train_ids or not val_ids:
                raise ValueError("Explicit train/validation split files produced an empty dataset")
            if set(train_ids) & set(val_ids):
                raise ValueError("Explicit train/validation split files must be disjoint")
            self.split_metadata = {
                "strategy": "explicit_split_files",
                "train_split_file": str(train_split_file),
                "val_split_file": str(val_split_file),
                "train_sequences": sorted(train_ids),
                "val_sequences": sorted(val_ids),
            }
            return train_ds, val_ds
        raise ValueError(
            "Training requires explicit dataset.train_split_file and dataset.val_split_file. "
            "Run preprocessing first or provide deterministic split files."
        )

    def _dataset_sequence_ids(self, dataset: Any) -> list[str]:
        if hasattr(dataset, "sequences"):
            values: list[str] = []
            for item in dataset.sequences:
                if hasattr(item, "sequence_id"):
                    values.append(str(item.sequence_id))
                elif isinstance(item, tuple) and item:
                    values.append(str(item[0]))
            return values
        return []

    def _loader_generator(self, train: bool) -> torch.Generator:
        generator = torch.Generator()
        offset = 0 if train else 10_000_000
        generator.manual_seed(int(self.train_cfg.get("seed", 26)) + offset + self.rank)
        return generator

    def _loader_metadata(self) -> dict[str, Any]:
        workers = int(self.train_cfg.get("workers", 4))
        return {
            "batch_size": int(self.train_cfg.get("batch_size", 2)),
            "workers": workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": bool(self.train_cfg.get("persistent_workers", workers > 0)),
            "prefetch_factor": int(self.train_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
            "drop_last": bool(self.train_cfg.get("drop_last", False)),
        }

    def _make_loader(self, dataset: Any, train: bool) -> DataLoader:
        batch_size = int(self.train_cfg.get("batch_size", 2))
        workers = int(self.train_cfg.get("workers", 4))
        sampler = DistributedSampler(dataset, shuffle=train) if self.distributed else None
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": train and sampler is None,
            "sampler": sampler,
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
            "drop_last": train and bool(self.train_cfg.get("drop_last", False)),
            "collate_fn": _collate_train,
            "generator": self._loader_generator(train),
            "worker_init_fn": _seed_worker,
        }
        if workers > 0:
            kwargs["persistent_workers"] = bool(self.train_cfg.get("persistent_workers", True))
            kwargs["prefetch_factor"] = int(self.train_cfg.get("prefetch_factor", 2))
        return DataLoader(
            dataset,
            **kwargs,
        )

    def _make_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        opt_name = str(self.train_cfg.get("optimizer", "Adadelta")).lower()
        lr = self.train_cfg.get("lr")
        wd = float(self.train_cfg.get("weight_decay", 0.0))
        if opt_name == "adadelta":
            return torch.optim.Adadelta(model.parameters(), lr=float(1.0 if lr is None else lr), weight_decay=wd)
        if opt_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=float(1e-3 if lr is None else lr), weight_decay=wd)
        if opt_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=float(1e-4 if lr is None else lr), weight_decay=wd)
        if opt_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=float(1e-2 if lr is None else lr), momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {opt_name}")

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> Any | None:
        name = self.train_cfg.get("scheduler")
        if not name or str(name).lower() == "none":
            return None
        if str(name).lower() == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.train_cfg.get("factor", 0.5)),
                patience=int(self.train_cfg.get("patience", 3)),
                min_lr=float(self.train_cfg.get("min_lr", 1e-6)),
            )
        if str(name).lower() == "multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(self.train_cfg.get("milestones", [20, 25])),
                gamma=float(self.train_cfg.get("gamma", 0.1)),
            )
        raise ValueError(f"Unknown scheduler: {name}")

    def _make_tensorboard_writer(self) -> Any | None:
        if not self.tensorboard_enabled or not self.is_main:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise RuntimeError("TensorBoard logging requires the 'tensorboard' package. Install project requirements before enabling train.tensorboard.") from exc
        return SummaryWriter(
            log_dir=str(self.output_dir / "tensorboard"),
            max_queue=int(self.train_cfg.get("tensorboard_max_queue", 100)),
            flush_secs=int(self.train_cfg.get("tensorboard_flush_secs", 30)),
        )

    def _find_open_port(self, preferred_port: int) -> int:
        for port in range(preferred_port, preferred_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError(f"Could not find an open TensorBoard port starting at {preferred_port}")

    def _launch_tensorboard_if_needed(self) -> TensorBoardServer | None:
        if not self.tensorboard_enabled or not self.is_main:
            return None
        if not bool(self.train_cfg.get("launch_tensorboard", True)):
            return None
        port = self._find_open_port(int(self.train_cfg.get("tensorboard_port", 6006)))
        logdir = Path(self.train_cfg.get("tensorboard_logdir", self.output_dir / "tensorboard"))
        cmd = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(logdir),
            "--host",
            str(self.train_cfg.get("tensorboard_host", "127.0.0.1")),
            "--port",
            str(port),
        ]
        if bool(self.train_cfg.get("tensorboard_profile_plugin", False)):
            cmd.append("--load_fast=false")
        process = _start_tensorboard_process(cmd)
        url = f"http://localhost:{port}"
        print(f"TensorBoard: {url}")
        print(f"TensorBoard logdir: {logdir}")
        return TensorBoardServer(process=process, url=url, logdir=logdir, port=port)

    def _restore_if_needed(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any | None) -> None:
        resume = self.train_cfg.get("resume_checkpoint") or (self.output_dir / "checkpoints" / "last.pt" if self.train_cfg.get("resume") else None)
        if not resume:
            return
        ckpt = load_checkpoint(resume, map_location="cpu")
        load_model_weights(model, ckpt, strict=True)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_loss = float(ckpt.get("best_score", float("inf")))

    def _step_loss(self, model: torch.nn.Module, criterion: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
        x = batch["input"].to(self.device, non_blocking=True).float()
        target = batch["target"].to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=self.amp_enabled):
            output = model(x)
        return criterion(output, target)

    def _run_epoch(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        profiler: Any | None = None,
        tensorboard: TensorBoardRunLogger | None = None,
    ) -> tuple[float, int]:
        train = optimizer is not None
        model.train(train)
        total = 0.0
        count = 0
        pbar = tqdm(loader, desc=("train" if train else "val") + f" epoch {epoch+1}", disable=not self.is_main, leave=False)
        for batch in pbar:
            step_started = time.perf_counter()
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss = self._step_loss(model, criterion, batch)
            batch_size = int(batch["input"].shape[0])
            if train:
                self.scaler.scale(loss).backward()
                if float(self.train_cfg.get("grad_clip_norm", 0.0)) > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.train_cfg.get("grad_clip_norm", 0.0)))
                self.scaler.step(optimizer)
                self.scaler.update()
                self.global_step += 1
                if profiler is not None:
                    profiler.step()
                if tensorboard is not None:
                    tensorboard.log_train_step(
                        global_step=self.global_step,
                        loss=float(loss.detach().cpu()),
                        lr=float(optimizer.param_groups[0].get("lr", 0.0)),
                        step_seconds=time.perf_counter() - step_started,
                        batch_size=batch_size,
                    )
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
            pbar.set_postfix(loss=f"{total / max(1, count):.5f}")
        avg = total / max(1, count)
        if self.distributed:
            t = torch.tensor([total, float(count)], dtype=torch.float32, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg = float((t[0] / torch.clamp(t[1], min=1.0)).item())
            count = int(t[1].item())
        return avg, count

    def _sample_batch(self, loader: DataLoader) -> dict[str, Any] | None:
        try:
            return next(iter(loader))
        except StopIteration:
            return None

    def fit(self) -> TrainResult:
        train_ds, val_ds = self._make_dataset()
        train_loader = self._make_loader(train_ds, train=True)
        val_loader = self._make_loader(val_ds, train=False)
        model = self.paper_spec.build_model(self.cfg["model"]).to(self.device)
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank] if self.device.type == "cuda" else None)
        default_loss = self.paper_spec.loss_name
        criterion = build_loss(str(self.train_cfg.get("loss", default_loss))).to(self.device)
        optimizer = self._make_optimizer(model)
        scheduler = self._make_scheduler(optimizer)
        self.tensorboard_writer = self._make_tensorboard_writer()
        self.tensorboard_server = self._launch_tensorboard_if_needed()
        self._restore_if_needed(model, optimizer, scheduler)
        hardware = collect_hardware_report()
        tensorboard = TensorBoardRunLogger(
            writer=self.tensorboard_writer,
            cfg=self.cfg,
            output_dir=self.output_dir,
            is_main=self.is_main,
            device=self.device,
            split_metadata=self.split_metadata,
            hardware=hardware,
        )
        tensorboard.log_run_metadata()
        val_sample = self._sample_batch(val_loader) if self.is_main else None
        if val_sample is not None:
            tensorboard.log_model_graph(model, val_sample)
        profiler = tensorboard.make_profiler()
        if profiler is not None:
            profiler.__enter__()
        epochs = int(self.train_cfg.get("epochs", 30))
        final_metrics: dict[str, float] = {}
        try:
            for epoch in range(self.start_epoch, epochs):
                if self.distributed and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                epoch_started = time.perf_counter()
                train_loss, train_samples = self._run_epoch(model, criterion, train_loader, optimizer, epoch, profiler, tensorboard)
                with torch.no_grad():
                    val_loss, _ = self._run_epoch(model, criterion, val_loader, None, epoch)
                epoch_seconds = time.perf_counter() - epoch_started
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                if self.is_main:
                    metrics = {"train_loss": train_loss, "val_loss": val_loss}
                    final_metrics = metrics
                    lr = float(optimizer.param_groups[0].get("lr", 0.0))
                    tensorboard.log_epoch(
                        epoch=epoch,
                        global_step=self.global_step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        lr=lr,
                        epoch_seconds=epoch_seconds,
                        train_samples=train_samples,
                        model=model,
                        val_sample=val_sample,
                    )
                    training_meta = {
                        "amp_requested": self.amp_requested,
                        "amp_enabled": self.amp_enabled,
                        "device": str(self.device),
                        "world_size": self.world_size,
                        "hardware": hardware,
                        "loader": self._loader_metadata(),
                        "tensorboard_enabled": self.tensorboard_writer is not None,
                        "tensorboard_url": self.tensorboard_server.url if self.tensorboard_server is not None else None,
                        "split": self.split_metadata,
                    }
                    state = CheckpointState(epoch=epoch, global_step=self.global_step, best_score=min(self.best_loss, val_loss), metrics=metrics, training=training_meta)
                    save_checkpoint(self.output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, state, self.cfg, scaler=self.scaler)
                    is_best = val_loss < self.best_loss
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        state.best_score = self.best_loss
                        save_checkpoint(self.output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, state, self.cfg, scaler=self.scaler)
                        save_model_only(self.output_dir / "checkpoints" / "model_best.pt", model, self.cfg, metrics)
                    tensorboard.log_checkpoint(epoch=epoch, global_step=self.global_step, metrics=metrics, is_best=is_best)
                    write_json(self.output_dir / "metrics.last.json", metrics)
                if self.distributed:
                    dist.barrier()
        finally:
            if profiler is not None:
                profiler.__exit__(None, None, None)
            if self.is_main:
                tensorboard.log_hparams(final_metrics)
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()
        return TrainResult(
            self.output_dir,
            self.best_loss,
            epochs - 1,
            self.tensorboard_server.url if self.tensorboard_server is not None else None,
        )
