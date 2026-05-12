# TrackNet Series Training Results

This directory contains tracked training logs and evaluation artifacts for the TrackNet V1-V5 training runs. Model checkpoint files remain in the local `outputs/train/` run directories and are not versioned in Git.

## File Contents

- `training_logs/`: TensorBoard event logs copied from each completed training run.
- `evaluation/`: evaluation metrics, protocols, resolved configs, checkpoint metadata, and frame-level predictions.

TrackNet V3 is split into two trained modules:

- `tracknet_v3_tracker`: the V3 heatmap tracking network.
- `tracknet_v3_rectifier`: the V3 trajectory rectification network.

## Training Setup

- Hardware: 8 x NVIDIA RTX 4090 GPUs, 24 GB each.
- Training framework: PyTorch Distributed Data Parallel.
- GPUs per model run: 2.
- Epochs: 30 for all runs.
- Mixed precision: disabled.
- Batch size convention: `train.batch_size` is per GPU / per DDP rank.
- Global batch size: `per_gpu_batch_size x number_of_gpus`.

Most TrackNet versions were trained with per-GPU batch size 1 on 2 GPUs, giving global batch size 2.

TrackNet V3 used paper-matched global batch sizes:

- V3 tracker: per-GPU batch size 5 on 2 GPUs, global batch size 10.
- V3 rectifier: per-GPU batch size 16 on 2 GPUs, global batch size 32.

## Run Summary

| Model | Best Epoch | Final Epoch | Per-GPU Batch | Global Batch | Optimizer | Learning Rate |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| TrackNet V1 | 1 | 30 | 1 | 2 | Adadelta | 1.0 |
| TrackNet V2 | 1 | 30 | 1 | 2 | Adadelta | 1.0 |
| TrackNet V3 Tracker | 11 | 30 | 5 | 10 | Adam | 0.001 |
| TrackNet V3 Rectifier | 25 | 30 | 16 | 32 | Adam | 0.001 |
| TrackNet V4 | 1 | 30 | 1 | 2 | Adadelta | 1.0 |
| TrackNet V5 | 2 | 30 | 1 | 2 | AdamW | 0.0001 |

## Local Checkpoints

The evaluation configs use these local best-validation model checkpoints:

```text
outputs/train/tracknet_v1_20260511_120015/checkpoints/model_best.pt
outputs/train/tracknet_v2_20260511_003832/checkpoints/model_best.pt
outputs/train/tracknet_v3_tracker_20260512_003925/checkpoints/model_best.pt
outputs/train/tracknet_v3_rectifier_20260512_003951/checkpoints/model_best.pt
outputs/train/tracknet_v4_20260511_002432/checkpoints/model_best.pt
outputs/train/tracknet_v5_20260511_002455/checkpoints/model_best.pt
```

## Directory Layout

```text
model_results/
  TRAINING_SUMMARY.md
  training_logs/
    tracknet_v1/
      tensorboard/
    tracknet_v2/
      tensorboard/
    tracknet_v3_tracker/
      tensorboard/
    tracknet_v3_rectifier/
      tensorboard/
    tracknet_v4/
      tensorboard/
    tracknet_v5/
      tensorboard/
  evaluation/
    tracknet_v1/
    tracknet_v2/
    tracknet_v3_tracker/
    tracknet_v3_tracker_rectifier/
    tracknet_v4/
    tracknet_v5/
```
