# TrackNet Series Training Report

This report documents the tracked training artifacts for the TrackNet V1-V5 runs used by the evaluation configs in this repository.

Model checkpoint files are not versioned in Git. They remain in local `outputs/train/` run directories. This directory tracks lightweight training logs, evaluation artifacts, and Markdown/CSV/JSON summaries.

## Artifact Scope

```text
model_results/
  TRAINING_SUMMARY.md
  EVALUATION_RESULTS.md
  training_logs/
  evaluation/
```

- `training_logs/`: TensorBoard event logs copied from completed training runs.
- `evaluation/`: evaluation metrics, protocols, resolved configs, checkpoint metadata, and frame-level prediction files.
- `EVALUATION_RESULTS.md`: detailed evaluation report for the completed 30-epoch trained checkpoints.

## Run Policy

- Model checkpoint files are stored locally under `outputs/train/`.
- Git tracks TensorBoard log exports and evaluation artifacts, not `.pt` checkpoint files.
- Evaluation configs use the completed 30-epoch trained checkpoint from each run.
- TrackNet V3 is represented by two trained modules: tracker and rectifier.

## Training Environment

| Field | Value |
| --- | --- |
| Hardware | 8 x NVIDIA RTX 4090 GPUs, 24 GB each |
| Framework | PyTorch Distributed Data Parallel |
| Python environment | conda environment `tracknet`, Python 3.11 |
| Epochs | 30 for all released runs |
| Mixed precision | Disabled |
| Checkpoint policy | Completed 30-epoch trained checkpoint used for evaluation |
| TensorBoard | Enabled during training; copied logs are tracked here |

## Batch Size Convention

`train.batch_size` is per GPU / per DDP rank.

```text
global_batch_size = per_gpu_batch_size * number_of_gpus
```

Most TrackNet versions were trained on 2 GPUs with per-GPU batch size 1, giving global batch size 2.

TrackNet V3 uses two separately trained modules:

- `tracknet_v3_tracker`: heatmap tracking network.
- `tracknet_v3_rectifier`: trajectory rectification network.

The V3 tracker and rectifier use larger paper-matched global batch sizes.

## Run Summary

| Model | Config | Training Epochs | GPUs | Per-GPU Batch | Global Batch | Optimizer | Learning Rate | AMP |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| TrackNet V1 | `configs/train_v1.yaml` | 30 | 2 | 1 | 2 | Adadelta | 1.0 | false |
| TrackNet V2 | `configs/train_v2.yaml` | 30 | 2 | 1 | 2 | Adadelta | 1.0 | false |
| TrackNet V3 tracker | `configs/train_v3_tracker.yaml` | 30 | 2 | 5 | 10 | Adam | 0.001 | false |
| TrackNet V3 rectifier | `configs/train_v3_rectifier.yaml` | 30 | 2 | 16 | 32 | Adam | 0.001 | false |
| TrackNet V4 | `configs/train_v4.yaml` | 30 | 2 | 1 | 2 | Adadelta | 1.0 | false |
| TrackNet V5 | `configs/train_v5.yaml` | 30 | 2 | 1 | 2 | AdamW | 0.0001 | false |

## Checkpoints Used By Evaluation

The evaluation configs point to these local completed-training checkpoints:

| Model | Local checkpoint |
| --- | --- |
| TrackNet V1 | `outputs/train/tracknet_v1_20260511_120015/checkpoints/last.pt` |
| TrackNet V2 | `outputs/train/tracknet_v2_20260511_003832/checkpoints/last.pt` |
| TrackNet V3 tracker | `outputs/train/tracknet_v3_tracker_20260512_003925/checkpoints/last.pt` |
| TrackNet V3 rectifier | `outputs/train/tracknet_v3_rectifier_20260512_003951/checkpoints/last.pt` |
| TrackNet V4 | `outputs/train/tracknet_v4_20260511_002432/checkpoints/last.pt` |
| TrackNet V5 | `outputs/train/tracknet_v5_20260511_002455/checkpoints/last.pt` |

## TensorBoard Log Exports

```text
model_results/training_logs/
  tracknet_v1/tensorboard/
  tracknet_v2/tensorboard/
  tracknet_v3_tracker/tensorboard/
  tracknet_v3_rectifier/tensorboard/
  tracknet_v4/tensorboard/
  tracknet_v5/tensorboard/
```

The copied TensorBoard logs are intended for reviewing training curves and run metadata without versioning full checkpoint files.

## Evaluation Linkage

Each model is evaluated from its completed-training checkpoint and writes an independent result directory:

```text
model_results/evaluation/
  tracknet_v1/
  tracknet_v2/
  tracknet_v3_tracker/
  tracknet_v3_tracker_rectifier/
  tracknet_v4/
  tracknet_v5/
```

See `model_results/EVALUATION_RESULTS.md` for evaluation protocols, confusion counts, aggregate metrics, and reproduction commands.
