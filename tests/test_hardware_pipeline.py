from __future__ import annotations

import copy
import os
from pathlib import Path

import torch

from tracknet.data.preprocessing import PreprocessConfig, preprocess_dataset
from tracknet.training.trainer import TrackNetTrainer
from tracknet.utils.hardware import collect_hardware_report


def test_parallel_preprocess_matches_single_worker_manifest(tmp_path: Path, synthetic_tracknet_domain_raw_root: Path) -> None:
    single_root = tmp_path / "single"
    parallel_root = tmp_path / "parallel"

    single = preprocess_dataset(
        PreprocessConfig(
            raw_root=synthetic_tracknet_domain_raw_root,
            output_root=single_root,
            target_width=32,
            target_height=32,
            overwrite=True,
            adapter="tracknet_domain",
            val_fraction=0.5,
            workers=1,
        )
    )
    parallel = preprocess_dataset(
        PreprocessConfig(
            raw_root=synthetic_tracknet_domain_raw_root,
            output_root=parallel_root,
            target_width=32,
            target_height=32,
            overwrite=True,
            adapter="tracknet_domain",
            val_fraction=0.5,
            workers=2,
        )
    )

    assert [item["sequence_id"] for item in parallel["sequences"]] == [item["sequence_id"] for item in single["sequences"]]
    assert parallel["splits"] == single["splits"]
    assert parallel["preprocess"]["workers"] == 2
    assert (parallel_root / "backgrounds" / "Professional__match1.png").exists()


def test_trainer_loader_uses_seeded_generator_and_worker_tuning(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    split_dir = synthetic_processed_two_sequences_root / "splits"
    split_dir.mkdir(exist_ok=True)
    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    train_file.write_text("match1__rally1\n", encoding="utf-8")
    val_file.write_text("match1__rally2\n", encoding="utf-8")
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "loader_tuning",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 2,
            "persistent_workers": True,
            "prefetch_factor": 3,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 123,
        },
    }
    trainer = TrackNetTrainer(cfg)
    train_ds, _ = trainer._make_dataset()

    loader = trainer._make_loader(train_ds, train=True)

    assert loader.generator is not None
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
    assert loader.worker_init_fn is not None


def test_output_dir_is_broadcast_from_rank_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr("tracknet.training.trainer.setup_distributed", lambda: (True, 1, 1, 2))
    monkeypatch.setattr("tracknet.training.trainer.dist.barrier", lambda: None)
    monkeypatch.setattr("tracknet.training.trainer.dist.broadcast_object_list", lambda obj, src: obj.__setitem__(0, str(tmp_path / "rank0_dir")))

    trainer = TrackNetTrainer(
        {
            "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
            "dataset": {"processed_root": str(tmp_path)},
            "train": {"experiment_name": "ddp_dir", "output_root": str(tmp_path / "outputs"), "device": "cpu"},
        }
    )

    assert trainer.output_dir == tmp_path / "rank0_dir"


def test_hardware_report_contains_accelerator_fields() -> None:
    report = collect_hardware_report()

    assert "platform" in report
    assert "torch" in report
    assert "cuda" in report
    assert "mps" in report
    assert isinstance(report["cuda"]["available"], bool)
    assert isinstance(report["mps"]["available"], bool)


def test_training_metadata_records_loader_and_hardware(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    split_dir = synthetic_processed_two_sequences_root / "splits"
    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    train_file.write_text("match1__rally1\n", encoding="utf-8")
    val_file.write_text("match1__rally2\n", encoding="utf-8")
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "metadata",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
        },
    }

    result = TrackNetTrainer(copy.deepcopy(cfg)).fit()
    ckpt = torch.load(result.output_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)

    assert ckpt["training"]["loader"]["batch_size"] == 2
    assert ckpt["training"]["loader"]["workers"] == 0
    assert "hardware" in ckpt["training"]
