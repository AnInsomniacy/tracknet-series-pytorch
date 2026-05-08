"""Training loop for TrackNet tracking models."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from tracknet.papers import get_paper_spec, paper_id_from_model_config
from tracknet.papers.base import PaperSpec
from tracknet.training.checkpoint import CheckpointState, load_checkpoint, load_model_weights, save_checkpoint, save_model_only
from tracknet.training.losses import build_loss
from tracknet.utils.device import select_device
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

@dataclass
class TrainResult:
    output_dir: Path
    best_loss: float
    last_epoch: int


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

    def _resolve_paper_spec(self) -> PaperSpec:
        model_cfg = self.cfg.get("model", {})
        return get_paper_spec(paper_id_from_model_config(model_cfg))

    def _resolve_output_dir(self) -> Path:
        resume = self.train_cfg.get("resume")
        if resume:
            return Path(resume)
        out_root = Path(self.train_cfg.get("output_root", "outputs"))
        name = str(self.train_cfg.get("experiment_name", "tracknet"))
        from datetime import datetime

        return out_root / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _split_indices_by_sequence(self, dataset: Any, val_split: float) -> tuple[list[int], list[int]]:
        """Split windows by sequence id to avoid validation leakage.

        Consecutive windows from the same rally are almost identical. A random
        window-level split makes validation see near-duplicates of training
        samples and gives a misleading score. The split unit is therefore the
        semantic sequence when the dataset exposes `windows` and `sequences`.
        """
        if not hasattr(dataset, "windows") or not hasattr(dataset, "sequences"):
            raise TypeError("Sequence-safe splitting requires a dataset with 'windows' and 'sequences' attributes")
        sequence_ids = [str(seq.sequence_id) for seq in dataset.sequences]
        if len(sequence_ids) == 1:
            return self._split_single_sequence_windows(dataset, val_split, sequence_ids[0])
        generator = torch.Generator().manual_seed(int(self.train_cfg.get("seed", 26)))
        order = torch.randperm(len(sequence_ids), generator=generator).tolist()
        val_count = max(1, int(round(len(sequence_ids) * val_split)))
        if val_count >= len(sequence_ids):
            val_count = len(sequence_ids) - 1
        val_sequence_indices = set(order[:val_count])
        train_indices: list[int] = []
        val_indices: list[int] = []
        for window_idx, window in enumerate(dataset.windows):
            target = val_indices if int(window.sequence_index) in val_sequence_indices else train_indices
            target.append(window_idx)
        self.split_metadata = {
            "strategy": "sequence",
            "train_sequences": sorted(sequence_ids[i] for i in set(range(len(sequence_ids))) - val_sequence_indices),
            "val_sequences": sorted(sequence_ids[i] for i in val_sequence_indices),
        }
        return train_indices, val_indices

    def _split_indices_by_trajectory_sequence(self, dataset: Any, val_split: float) -> tuple[list[int], list[int]]:
        if not hasattr(dataset, "windows") or not hasattr(dataset, "sequences"):
            raise TypeError("Trajectory sequence-safe splitting requires 'windows' and 'sequences' attributes")
        sequence_ids = [str(item[0]) for item in dataset.sequences]
        if len(sequence_ids) == 1:
            return self._split_single_sequence_windows(dataset, val_split, sequence_ids[0])
        generator = torch.Generator().manual_seed(int(self.train_cfg.get("seed", 26)))
        order = torch.randperm(len(sequence_ids), generator=generator).tolist()
        val_count = max(1, int(round(len(sequence_ids) * val_split)))
        if val_count >= len(sequence_ids):
            val_count = len(sequence_ids) - 1
        val_sequence_indices = set(order[:val_count])
        train_indices: list[int] = []
        val_indices: list[int] = []
        for window_idx, (sequence_idx, _) in enumerate(dataset.windows):
            target = val_indices if int(sequence_idx) in val_sequence_indices else train_indices
            target.append(window_idx)
        self.split_metadata = {
            "strategy": "sequence",
            "train_sequences": sorted(sequence_ids[i] for i in set(range(len(sequence_ids))) - val_sequence_indices),
            "val_sequences": sorted(sequence_ids[i] for i in val_sequence_indices),
        }
        return train_indices, val_indices

    def _split_single_sequence_windows(self, dataset: Any, val_split: float, sequence_id: str) -> tuple[list[int], list[int]]:
        """Fallback for tiny smoke datasets that contain only one sequence.

        Real experiments should use at least two rallies and therefore use the
        sequence-level split above. For synthetic smoke tests we still need a
        deterministic train/validation split; a contiguous split is less leaky
        than random interleaving and keeps the limitation visible in metadata.
        """
        n = len(dataset)
        val_size = max(1, int(round(n * val_split)))
        if val_size >= n:
            val_size = n - 1
        train_indices = list(range(0, n - val_size))
        val_indices = list(range(n - val_size, n))
        self.split_metadata = {
            "strategy": "single_sequence_contiguous_window",
            "warning": "Only one sequence is available; use multiple sequences for leakage-safe validation.",
            "train_sequences": [sequence_id],
            "val_sequences": [sequence_id],
        }
        return train_indices, val_indices

    def _make_dataset(self) -> tuple[Any, Any]:
        dataset_section = self.cfg["dataset"]
        dataset = self.paper_spec.build_training_dataset(dataset_section)
        val_split = float(self.train_cfg.get("val_split", 0.2))
        if not 0.0 < val_split < 1.0:
            raise ValueError("train.val_split must be between 0 and 1")
        val_size = max(1, int(round(len(dataset) * val_split)))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("Dataset is too small for the requested validation split")
        if self.paper_spec.split_strategy == "trajectory_sequence":
            train_indices, val_indices = self._split_indices_by_trajectory_sequence(dataset, val_split)
        elif self.paper_spec.split_strategy == "sequence":
            train_indices, val_indices = self._split_indices_by_sequence(dataset, val_split)
        else:
            raise ValueError(f"Unsupported split strategy: {self.paper_spec.split_strategy}")
        if not train_indices or not val_indices:
            raise ValueError("Sequence-safe split produced an empty train or validation set")
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    def _make_loader(self, dataset: Any, train: bool) -> DataLoader:
        batch_size = int(self.train_cfg.get("batch_size", 2))
        workers = int(self.train_cfg.get("workers", 4))
        sampler = DistributedSampler(dataset, shuffle=train) if self.distributed else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(train and sampler is None),
            sampler=sampler,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            drop_last=train and bool(self.train_cfg.get("drop_last", False)),
            collate_fn=_collate_train,
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

    def _run_epoch(self, model: torch.nn.Module, criterion: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, epoch: int) -> float:
        train = optimizer is not None
        model.train(train)
        total = 0.0
        count = 0
        pbar = tqdm(loader, desc=("train" if train else "val") + f" epoch {epoch+1}", disable=not self.is_main, leave=False)
        for batch in pbar:
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss = self._step_loss(model, criterion, batch)
            if train:
                self.scaler.scale(loss).backward()
                if float(self.train_cfg.get("grad_clip_norm", 0.0)) > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.train_cfg.get("grad_clip_norm", 0.0)))
                self.scaler.step(optimizer)
                self.scaler.update()
                self.global_step += 1
            total += float(loss.detach().cpu())
            count += 1
            pbar.set_postfix(loss=f"{total / max(1, count):.5f}")
        avg = total / max(1, count)
        if self.distributed:
            t = torch.tensor([avg], dtype=torch.float32, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = float(t.item())
        return avg

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
        self._restore_if_needed(model, optimizer, scheduler)
        epochs = int(self.train_cfg.get("epochs", 30))
        for epoch in range(self.start_epoch, epochs):
            if self.distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            train_loss = self._run_epoch(model, criterion, train_loader, optimizer, epoch)
            with torch.no_grad():
                val_loss = self._run_epoch(model, criterion, val_loader, None, epoch)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            if self.is_main:
                metrics = {"train_loss": train_loss, "val_loss": val_loss}
                training_meta = {
                    "amp_requested": self.amp_requested,
                    "amp_enabled": self.amp_enabled,
                    "device": str(self.device),
                    "world_size": self.world_size,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "split": self.split_metadata,
                }
                state = CheckpointState(epoch=epoch, global_step=self.global_step, best_score=min(self.best_loss, val_loss), metrics=metrics, training=training_meta)
                save_checkpoint(self.output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, state, self.cfg, scaler=self.scaler)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    state.best_score = self.best_loss
                    save_checkpoint(self.output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, state, self.cfg, scaler=self.scaler)
                    save_model_only(self.output_dir / "checkpoints" / "model_best.pt", model, self.cfg, metrics)
                write_json(self.output_dir / "metrics.last.json", metrics)
            if self.distributed:
                dist.barrier()
        return TrainResult(self.output_dir, self.best_loss, epochs - 1)
