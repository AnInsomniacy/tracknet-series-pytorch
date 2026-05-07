from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from tracknet.data.trajectory_dataset import TrajectoryRectifierDataset, TrajectoryRectifierDatasetConfig
from tracknet.evaluation import EvaluationConfig, evaluate_checkpoint
from tracknet.inference import VideoPredictionConfig, run_video_prediction
from tracknet.inference.postprocess import Prediction, decode_heatmap
from tracknet.inference.rectification import build_v3_inpainting_mask
from tracknet.models import build_model
from tracknet.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from tracknet.training.trainer import TrackNetTrainer


def _write_synthetic_video(path: Path, frames: int = 5, width: int = 32, height: int = 24) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (5 + i, 8), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_postprocess_decode_heatmap() -> None:
    heat = np.zeros((16, 16), dtype=np.float32)
    heat[7:10, 4:7] = 1.0
    pred = decode_heatmap(heat, threshold=0.5)
    assert pred.visibility == 1
    assert pred.x == 5
    assert pred.y == 8


def test_video_prediction_outputs_all_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "in.mp4"
    _write_synthetic_video(video_path, frames=5)
    model_cfg = {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "model.pt"
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(ckpt_path, model, opt, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)
    out_csv = tmp_path / "pred.csv"
    out_video = tmp_path / "vis.mp4"
    df = run_video_prediction(
        VideoPredictionConfig(
            video_path=video_path,
            checkpoint_path=ckpt_path,
            output_csv=out_csv,
            output_video=out_video,
            model=model_cfg["model"],
            target_width=32,
            target_height=32,
            sequence_length=3,
            batch_size=2,
            threshold=1.1,
            device="cpu",
        )
    )
    assert len(df) == 5
    assert df["Frame"].tolist() == [0, 1, 2, 3, 4]
    assert df["Visibility"].tolist() == [0, 0, 0, 0, 0]
    assert out_video.exists()


def test_evaluation_runs_on_synthetic_processed(tmp_path: Path, synthetic_processed_root: Path) -> None:
    model_cfg = {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "eval_model.pt"
    save_checkpoint(ckpt_path, model, None, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)
    result = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=ckpt_path,
            output_dir=tmp_path / "eval",
            dataset={"processed_root": str(synthetic_processed_root), "sequence_length": 3},
            model=model_cfg["model"],
            batch_size=2,
            workers=0,
            device="cpu",
            threshold=1.1,
        )
    )
    assert result.metrics["total"] > 0
    assert result.predictions_csv.exists()
    assert pd.read_csv(result.predictions_csv).shape[0] > 0


def test_train_smoke_saves_checkpoint(tmp_path: Path, synthetic_processed_root: Path) -> None:
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {"processed_root": str(synthetic_processed_root), "sequence_length": 3},
        "train": {
            "experiment_name": "smoke",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "val_split": 0.34,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
        },
    }
    result = TrackNetTrainer(cfg).fit()
    assert (result.output_dir / "checkpoints" / "last.pt").exists()
    assert (result.output_dir / "checkpoints" / "best.pt").exists()
    assert (result.output_dir / "checkpoints" / "model_best.pt").exists()


def test_train_split_is_sequence_safe(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {"processed_root": str(synthetic_processed_two_sequences_root), "sequence_length": 3},
        "train": {
            "experiment_name": "split_safe",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "val_split": 0.5,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
        },
    }
    trainer = TrackNetTrainer(cfg)
    train_ds, val_ds = trainer._make_dataset()
    train_sequences = {train_ds.dataset[window_idx]["sequence_id"] for window_idx in train_ds.indices}
    val_sequences = {val_ds.dataset[window_idx]["sequence_id"] for window_idx in val_ds.indices}
    assert train_sequences
    assert val_sequences
    assert train_sequences.isdisjoint(val_sequences)


def test_train_amp_setting_is_recorded_in_checkpoint(tmp_path: Path, synthetic_processed_root: Path) -> None:
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {"processed_root": str(synthetic_processed_root), "sequence_length": 3},
        "train": {
            "experiment_name": "amp_smoke",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "val_split": 0.34,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "amp": True,
        },
    }
    result = TrackNetTrainer(cfg).fit()
    ckpt = load_checkpoint(result.output_dir / "checkpoints" / "last.pt")
    assert ckpt["training"]["amp_requested"] is True
    assert ckpt["training"]["amp_enabled"] is False


def test_v3_trajectory_dataset_and_inpainting_mask(synthetic_processed_root: Path) -> None:
    ds = TrajectoryRectifierDataset(TrajectoryRectifierDatasetConfig(processed_root=synthetic_processed_root, trajectory_length=4, mask_ratio=0.5, seed=1))
    sample = ds[0]
    assert sample["input"].shape == (4, 4)
    assert sample["target"].shape == (3, 4)
    preds = [Prediction(1, 0, 10, 1), Prediction(0, -1, -1, 0), Prediction(1, 2, 15, 1), Prediction(0, -1, -1, 0)]
    mask = build_v3_inpainting_mask(preds, delta_y_pixels=10)
    assert mask[1] == 1
    assert mask[3] == 0


def test_evaluation_counts_each_frame_once(tmp_path: Path, synthetic_processed_root: Path) -> None:
    model_cfg = {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "eval_model.pt"
    save_checkpoint(ckpt_path, model, None, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)
    result = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=ckpt_path,
            output_dir=tmp_path / "eval_once",
            dataset={"processed_root": str(synthetic_processed_root), "sequence_length": 3},
            model=model_cfg["model"],
            batch_size=2,
            workers=0,
            device="cpu",
            threshold=1.1,
        )
    )
    predictions = pd.read_csv(result.predictions_csv)
    assert predictions["frame"].is_unique
    assert result.metrics["total"] == len(predictions)
