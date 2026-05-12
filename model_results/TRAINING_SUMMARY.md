# TrackNet Series Pretrained Models

This directory contains pretrained TrackNet V1-V5 model artifacts and the complete TensorBoard logs from the corresponding training runs.

## File Contents

- `best_epoch_XX.pt`: model-only checkpoint saved from the best validation epoch.
- `last_epoch_30.pt`: full training checkpoint saved after the final epoch.
- `tensorboard/`: complete TensorBoard event logs for the training run.

TrackNet V3 is split into two trained modules:

- `tracknet_v3/tracker/`: the V3 heatmap tracking network.
- `tracknet_v3/rectifier/`: the V3 trajectory rectification network.

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

## Directory Layout

```text
pretrained_models/
  TRAINING_SUMMARY.md
  tracknet_v1/
    best_epoch_01.pt
    last_epoch_30.pt
    tensorboard/
  tracknet_v2/
    best_epoch_01.pt
    last_epoch_30.pt
    tensorboard/
  tracknet_v3/
    tracker/
      best_epoch_11.pt
      last_epoch_30.pt
      tensorboard/
    rectifier/
      best_epoch_25.pt
      last_epoch_30.pt
      tensorboard/
  tracknet_v4/
    best_epoch_01.pt
    last_epoch_30.pt
    tensorboard/
  tracknet_v5/
    best_epoch_02.pt
    last_epoch_30.pt
    tensorboard/
```
