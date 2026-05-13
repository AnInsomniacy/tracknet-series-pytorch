from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tracknet.data.preprocessing import PreprocessConfig, preprocess_dataset
from tracknet.data.trajectory_dataset import TrajectoryRectifierDataset, TrajectoryRectifierDatasetConfig
from tracknet.evaluation import EvaluationConfig, evaluate_checkpoint
import tracknet.evaluation.evaluator as evaluator_module
from tracknet.inference import VideoPredictionConfig, run_video_prediction
import tracknet.inference.video_predictor as video_predictor_module
from tracknet.inference.postprocess import Prediction, decode_heatmap
from tracknet.inference.rectification import build_v3_inpainting_mask
from tracknet.models import build_model
from tracknet.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
import tracknet.training.trainer as trainer_module
from tracknet.training.trainer import TrackNetTrainer
from tracknet.tools.collect_evaluations import write_evaluation_report


def _write_split_files(processed_root: Path, train_ids: list[str], val_ids: list[str]) -> tuple[Path, Path]:
    splits_dir = processed_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    train_file.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    return train_file, val_file


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


def test_video_prediction_config_enables_progress_and_verbose_output_by_default() -> None:
    import inspect

    signature = inspect.signature(VideoPredictionConfig)

    assert signature.parameters["progress"].default is True
    assert signature.parameters["verbose"].default is True


def test_video_prediction_reports_prediction_and_overlay_progress(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "in.mp4"
    _write_synthetic_video(video_path, frames=5)
    model_cfg = {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(ckpt_path, model, None, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)
    progress_runs: list[dict[str, object]] = []

    class RecordingProgress:
        def __init__(self, iterable=None, **kwargs):  # noqa: ANN001
            self.iterable = iterable
            self.kwargs = kwargs
            self.updated = 0
            self.closed = False
            progress_runs.append({"kwargs": kwargs, "bar": self})

        def __iter__(self):
            assert self.iterable is not None
            yield from self.iterable

        def update(self, n: int) -> None:
            self.updated += int(n)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(video_predictor_module, "tqdm", RecordingProgress)
    out_csv = tmp_path / "pred.csv"
    out_video = tmp_path / "vis.mp4"

    run_video_prediction(
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
            verbose=False,
        )
    )

    by_desc = {str(item["kwargs"]["desc"]): item["bar"] for item in progress_runs}
    assert set(by_desc) == {"Predicting windows", "Writing overlay"}
    assert by_desc["Predicting windows"].updated == 3
    assert by_desc["Writing overlay"].updated == 5
    assert by_desc["Predicting windows"].closed is True
    assert by_desc["Writing overlay"].closed is True


def test_video_prediction_rejects_sequence_length_that_does_not_match_checkpoint(tmp_path: Path) -> None:
    video_path = tmp_path / "in.mp4"
    _write_synthetic_video(video_path, frames=5)
    model_cfg = {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(ckpt_path, model, None, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)

    try:
        run_video_prediction(
            VideoPredictionConfig(
                video_path=video_path,
                checkpoint_path=ckpt_path,
                output_csv=tmp_path / "pred.csv",
                model=model_cfg["model"],
                target_width=32,
                target_height=32,
                sequence_length=8,
                batch_size=2,
                threshold=1.1,
                device="cpu",
                progress=False,
                verbose=False,
            )
        )
    except ValueError as exc:
        assert "sequence_length" in str(exc)
        assert "checkpoint" in str(exc)
        return

    raise AssertionError("prediction must reject a sequence_length mismatch before model execution")


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


def test_train_smoke_saves_checkpoint(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "smoke",
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
    result = TrackNetTrainer(cfg).fit()
    assert (result.output_dir / "checkpoints" / "last.pt").exists()
    assert (result.output_dir / "checkpoints" / "best.pt").exists()
    assert (result.output_dir / "checkpoints" / "model_best.pt").exists()


def test_train_split_is_sequence_safe(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "split_safe",
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
    trainer = TrackNetTrainer(cfg)
    train_ds, val_ds = trainer._make_dataset()
    train_sequences = _dataset_sequence_ids(train_ds)
    val_sequences = _dataset_sequence_ids(val_ds)
    assert train_sequences
    assert val_sequences
    assert train_sequences.isdisjoint(val_sequences)


def _dataset_sequence_ids(dataset) -> set[str]:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return {dataset.dataset[window_idx]["sequence_id"] for window_idx in dataset.indices}
    if hasattr(dataset, "sequences"):
        return {str(seq.sequence_id) for seq in dataset.sequences}
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def test_train_uses_explicit_processed_split_files(tmp_path: Path, synthetic_tracknet_domain_raw_root: Path) -> None:
    processed = tmp_path / "processed"
    preprocess_dataset(
        PreprocessConfig(
            raw_root=synthetic_tracknet_domain_raw_root,
            output_root=processed,
            target_width=32,
            target_height=32,
            overwrite=True,
            adapter="tracknet_domain",
            val_fraction=0.5,
        )
    )
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(processed),
            "sequence_length": 3,
            "train_split_file": str(processed / "splits" / "train.txt"),
            "val_split_file": str(processed / "splits" / "val.txt"),
        },
        "train": {
            "experiment_name": "explicit_split",
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

    train_ds, val_ds = TrackNetTrainer(cfg)._make_dataset()

    assert _dataset_sequence_ids(train_ds) == set((processed / "splits" / "train.txt").read_text().split())
    assert _dataset_sequence_ids(val_ds) == set((processed / "splits" / "val.txt").read_text().split())


def test_train_requires_explicit_split_files(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {"processed_root": str(synthetic_processed_two_sequences_root), "sequence_length": 3},
        "train": {
            "experiment_name": "split_required",
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

    try:
        TrackNetTrainer(cfg)._make_dataset()
    except ValueError as exc:
        assert "train_split_file" in str(exc)
        assert "val_split_file" in str(exc)
        return

    raise AssertionError("training must require explicit train/validation split files")


def test_train_writes_tensorboard_scalars_when_enabled(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "tensorboard_smoke",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "tensorboard": True,
        },
    }

    result = TrackNetTrainer(cfg).fit()

    assert list((result.output_dir / "tensorboard").glob("events.out.tfevents.*"))


def test_train_writes_rich_tensorboard_observability(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "tensorboard_rich",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "tensorboard": True,
            "launch_tensorboard": False,
            "tensorboard_images": True,
            "tensorboard_histograms": True,
            "tensorboard_hparams": True,
            "tensorboard_model_graph": False,
        },
    }

    result = TrackNetTrainer(cfg).fit()
    accumulator = EventAccumulator(str(result.output_dir / "tensorboard"))
    accumulator.Reload()
    tags = accumulator.Tags()

    assert {"loss/train", "loss/val", "optim/lr", "train/global_step", "train/epoch_seconds", "train/samples_per_second"}.issubset(set(tags["scalars"]))
    tensor_tags = set(tags["tensors"])
    assert {"config/resolved/text_summary", "config/hardware/text_summary", "config/splits/text_summary", "checkpoint/latest/text_summary"}.issubset(tensor_tags)
    assert any(tag.startswith("samples/validation") for tag in tags["images"])
    assert any(tag.startswith("parameters/") for tag in tags["histograms"])


def test_train_writes_tensorboard_step_scalars_at_configured_interval(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "tensorboard_steps",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "tensorboard": True,
            "launch_tensorboard": False,
            "log_every_steps": 2,
            "tensorboard_flush_secs": 30,
            "tensorboard_max_queue": 100,
        },
    }

    result = TrackNetTrainer(cfg).fit()
    accumulator = EventAccumulator(str(result.output_dir / "tensorboard"))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    assert {"loss/train_step", "loss/train_ema", "optim/lr_step", "train/step_seconds", "train/samples_per_second_step"}.issubset(tags)
    assert [event.step for event in accumulator.Scalars("loss/train_step")] == [1, 2]


def test_tensorboard_launch_can_enable_profiler(monkeypatch, tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_start_tensorboard_process(cmd):  # noqa: ANN001
        launched["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr(trainer_module, "_start_tensorboard_process", fake_start_tensorboard_process)
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "tensorboard_profiler",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "tensorboard": True,
            "tensorboard_profile_plugin": True,
        },
    }

    TrackNetTrainer(cfg).fit()

    assert "--load_fast=false" in launched["cmd"]


def test_tensorboard_profiler_steps_during_training(monkeypatch, tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    calls = {"enter": 0, "step": 0, "exit": 0}

    class FakeProfiler:
        def __enter__(self):
            calls["enter"] += 1
            return self

        def step(self):
            calls["step"] += 1

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            calls["exit"] += 1

    monkeypatch.setattr("tracknet.training.tensorboard.torch.profiler.profile", lambda **kwargs: FakeProfiler())
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "tensorboard_profiler_steps",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "tensorboard": True,
            "launch_tensorboard": False,
            "tensorboard_profiler": True,
        },
    }

    TrackNetTrainer(cfg).fit()

    assert calls == {"enter": 1, "step": 3, "exit": 1}


def test_train_launches_tensorboard_by_default_when_logging_is_enabled(monkeypatch, tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_start_tensorboard_process(cmd):  # noqa: ANN001
        launched["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr(trainer_module, "_start_tensorboard_process", fake_start_tensorboard_process)
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "tensorboard_launch",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "tensorboard": True,
        },
    }

    result = TrackNetTrainer(cfg).fit()

    assert launched["cmd"][:3] == [torch.sys.executable, "-m", "tensorboard.main"]
    assert "--logdir" in launched["cmd"]
    assert "--port" in launched["cmd"]
    port_index = launched["cmd"].index("--port") + 1
    assert result.tensorboard_url == f"http://localhost:{launched['cmd'][port_index]}"
    logdir_index = launched["cmd"].index("--logdir") + 1
    assert Path(launched["cmd"][logdir_index]) == result.output_dir / "tensorboard"


def test_train_can_disable_tensorboard_launch(monkeypatch, tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])

    def fail_start_tensorboard_process(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("TensorBoard process should not launch")

    monkeypatch.setattr(trainer_module, "_start_tensorboard_process", fail_start_tensorboard_process)
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "tensorboard_no_launch",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "wbce",
            "device": "cpu",
            "seed": 7,
            "tensorboard": True,
            "launch_tensorboard": False,
        },
    }

    result = TrackNetTrainer(cfg).fit()

    assert result.tensorboard_url is None


def test_train_amp_setting_is_recorded_in_checkpoint(tmp_path: Path, synthetic_processed_two_sequences_root: Path) -> None:
    train_file, val_file = _write_split_files(synthetic_processed_two_sequences_root, ["match1__rally1"], ["match1__rally2"])
    cfg = {
        "model": {"version": "v2", "sequence_length": 3, "base_channels": 4},
        "dataset": {
            "processed_root": str(synthetic_processed_two_sequences_root),
            "sequence_length": 3,
            "train_split_file": str(train_file),
            "val_split_file": str(val_file),
        },
        "train": {
            "experiment_name": "amp_smoke",
            "output_root": str(tmp_path / "outputs"),
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
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
    preds = [Prediction(1, 0, 5, 1), Prediction(0, -1, -1, 0), Prediction(1, 2, 15, 1), Prediction(0, -1, -1, 0)]
    mask = build_v3_inpainting_mask(preds, delta_y_pixels=10)
    assert mask[1] == 0
    assert mask[3] == 0


def test_v3_inpainting_mask_matches_paper_height_threshold() -> None:
    paper_repairable = [
        Prediction(1, 0, 8, 1),
        Prediction(0, -1, -1, 0),
        Prediction(1, 2, 9, 1),
    ]
    visually_similar_but_out_of_field = [
        Prediction(1, 0, 50, 1),
        Prediction(0, -1, -1, 0),
        Prediction(1, 2, 55, 1),
    ]

    assert build_v3_inpainting_mask(paper_repairable, delta_y_pixels=30).tolist() == [0.0, 1.0, 0.0]
    assert build_v3_inpainting_mask(visually_similar_but_out_of_field, delta_y_pixels=30).tolist() == [0.0, 0.0, 0.0]


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


def test_evaluation_uses_paper_default_tolerance_when_config_omits_override() -> None:
    v1_cfg = EvaluationConfig(
        checkpoint_path=Path("unused.pt"),
        output_dir=Path("unused"),
        dataset={"processed_root": "unused"},
        model={"version": "v1"},
    )
    v2_cfg = EvaluationConfig(
        checkpoint_path=Path("unused.pt"),
        output_dir=Path("unused"),
        dataset={"processed_root": "unused"},
        model={"version": "v2"},
    )

    assert evaluator_module.resolve_evaluation_protocol(v1_cfg).tolerance_pixels == 5.0
    assert evaluator_module.resolve_evaluation_protocol(v2_cfg).tolerance_pixels == 4.0


def test_evaluation_config_can_override_protocol_tolerance_and_coordinate_space() -> None:
    cfg = EvaluationConfig(
        checkpoint_path=Path("unused.pt"),
        output_dir=Path("unused"),
        dataset={"processed_root": "unused"},
        model={"version": "v5"},
        tolerance_pixels=7.0,
        coordinate_space="raw",
    )

    protocol = evaluator_module.resolve_evaluation_protocol(cfg)

    assert protocol.tolerance_pixels == 7.0
    assert protocol.coordinate_space == "raw"


def test_evaluation_writes_auditable_protocol_files(tmp_path: Path, synthetic_processed_root: Path) -> None:
    model_cfg = {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "eval_model.pt"
    save_checkpoint(
        ckpt_path,
        model,
        None,
        None,
        CheckpointState(epoch=12, global_step=34, best_score=0.25, metrics={"val_loss": 0.25}),
        model_cfg,
    )

    result = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=ckpt_path,
            output_dir=tmp_path / "eval_audit",
            dataset={"processed_root": str(synthetic_processed_root), "sequence_length": 3},
            model=model_cfg["model"],
            batch_size=2,
            workers=0,
            device="cpu",
            threshold=1.1,
        )
    )

    assert result.protocol_json.exists()
    assert result.resolved_config_json.exists()
    assert result.checkpoint_json.exists()
    assert result.sequence_metrics_json.exists()
    predictions = pd.read_csv(result.predictions_csv)
    assert {"gt_x_raw", "gt_y_raw", "pred_x_raw", "pred_y_raw", "coordinate_space"}.issubset(predictions.columns)


def test_evaluation_can_score_in_raw_coordinate_space(monkeypatch, tmp_path: Path, synthetic_raw_root: Path) -> None:
    processed = tmp_path / "processed_scaled"
    preprocess_dataset(PreprocessConfig(raw_root=synthetic_raw_root, output_root=processed, target_width=16, target_height=16, overwrite=True))
    model_cfg = {"model": {"version": "v2", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "eval_model.pt"
    save_checkpoint(ckpt_path, model, None, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)

    def fake_predict(*args, **kwargs):
        return [evaluator_module.FramePrediction(frame=2, prediction=Prediction(1, 6.5, 6.0, 1.0))]

    monkeypatch.setattr(evaluator_module.PaperSpec, "aggregate_window_outputs", fake_predict)

    model_result = evaluate_checkpoint(
        EvaluationConfig(
                checkpoint_path=ckpt_path,
                output_dir=tmp_path / "eval_model_space",
                dataset={"processed_root": str(processed), "sequence_length": 3},
            model=model_cfg["model"],
            workers=0,
            device="cpu",
            coordinate_space="model",
            tolerance_pixels=2.0,
        )
    )
    raw_result = evaluate_checkpoint(
        EvaluationConfig(
                checkpoint_path=ckpt_path,
                output_dir=tmp_path / "eval_raw_space",
                dataset={"processed_root": str(processed), "sequence_length": 3},
            model=model_cfg["model"],
            workers=0,
            device="cpu",
            coordinate_space="raw",
            tolerance_pixels=2.0,
        )
    )

    assert model_result.metrics["tp"] == 1
    assert raw_result.metrics["fp1"] == 1


def test_v3_full_evaluation_applies_rectifier_when_configured(monkeypatch, tmp_path: Path, synthetic_processed_root: Path) -> None:
    model_cfg = {"model": {"version": "v3", "sequence_length": 3, "base_channels": 4}}
    model = build_model(model_cfg["model"])
    ckpt_path = tmp_path / "tracker.pt"
    rectifier_path = tmp_path / "rectifier.pt"
    save_checkpoint(ckpt_path, model, None, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)
    save_checkpoint(rectifier_path, model, None, None, CheckpointState(epoch=0, global_step=0, best_score=0.0, metrics={}), model_cfg)
    calls = {"count": 0}

    def fake_predict(*args, **kwargs):
        return [evaluator_module.FramePrediction(frame=2, prediction=Prediction(1, 4.0, 4.0, 1.0))]

    def fake_rectify(predictions, *, raw_width, raw_height, cfg):
        calls["count"] += 1
        return predictions

    monkeypatch.setattr(evaluator_module.PaperSpec, "aggregate_window_outputs", fake_predict)
    monkeypatch.setattr(evaluator_module, "rectify_predictions", fake_rectify)

    evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=ckpt_path,
            output_dir=tmp_path / "eval_v3_tracker_rectifier",
            dataset={"processed_root": str(synthetic_processed_root), "sequence_length": 3},
            model=model_cfg["model"],
            workers=0,
            device="cpu",
            rectifier_checkpoint_path=rectifier_path,
        )
    )

    assert calls["count"] == 1


def test_evaluation_streams_predictions_without_sequence_output_cache() -> None:
    import inspect

    source = inspect.getsource(evaluator_module.evaluate_checkpoint)

    assert "sequence_outputs" not in source
    assert "torch.cat(sequence_outputs" not in source


def test_evaluation_progress_bar_is_enabled_by_default() -> None:
    import inspect

    signature = inspect.signature(EvaluationConfig)
    source = inspect.getsource(evaluator_module.evaluate_checkpoint)

    assert signature.parameters["progress"].default is True
    assert "tqdm(" in source
    assert "Evaluating" in source


def test_evaluation_report_is_concise_and_links_artifacts(tmp_path: Path) -> None:
    summary_rows = [
        {
            "name": "tracknet_v1",
            "output_dir": str(tmp_path / "evaluation" / "tracknet_v1"),
            "checkpoint_path": "outputs/train/tracknet_v1_20260511_120015/checkpoints/model_best.pt",
            "coordinate_space": "model",
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.7466667,
            "tp": 7.0,
            "tn": 2.0,
            "fp1": 1.0,
            "fp2": 1.0,
            "fn": 3.0,
            "total": 14.0,
        }
    ]
    report_path = tmp_path / "EVALUATION_RESULTS.md"

    write_evaluation_report(report_path, summary_rows)

    text = report_path.read_text(encoding="utf-8")
    assert "# TrackNet Series Evaluation Results" in text
    assert "| tracknet_v1 | `outputs/train/tracknet_v1_20260511_120015/checkpoints/model_best.pt` | model | 0.9000 | 0.8000 | 0.7000 | 0.7467 |" in text
    assert "`metrics.json`" in text
    assert "TrackNetV2-sized protocol" in text
