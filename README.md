# TrackNet Series PyTorch

PyTorch implementation of the TrackNet shuttlecock trajectory tracking family, covering TrackNet V1, V2, V3, V4, and V5 in one reproducible pipeline.

The project provides source code for preprocessing, training, evaluation, video inference, visualization, and synthetic tests. Paper-specific behavior is centralized in version contracts, while the training, evaluation, and inference engines stay model-agnostic.

## Contents

- [Capabilities](#capabilities)
- [Model Coverage](#model-coverage)
- [Repository Layout](#repository-layout)
- [Environment](#environment)
- [Raw Dataset Layout](#raw-dataset-layout)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Video Inference](#video-inference)
- [Visualization](#visualization)
- [Tests](#tests)
- [Artifact Policy](#artifact-policy)

## Capabilities

- TrackNet V1-V5 model implementations.
- A stable processed dataset format built from raw badminton rally videos and trajectory CSV annotations.
- Paper-owned target generation, post-processing, window aggregation, and evaluation protocols.
- Single-GPU and PyTorch Distributed Data Parallel training.
- Checkpoint resume, TensorBoard logging, and evaluation-ready checkpoint export.
- Reproducible evaluation artifacts under `model_results/`.
- Frame-preserving video inference that exports `Frame,Visibility,X,Y` in raw video coordinates.
- Synthetic tests for data handling, model shapes, losses, checkpoints, training, evaluation, and inference.

## Model Coverage

| Version | Main module | Input contract | Target / loss | Evaluation behavior |
| --- | --- | --- | --- | --- |
| TrackNet V1 | `tracknet.models.tracknet_v1.TrackNetV1` | 3 RGB frames, 640x360 default | 256-class heatmap, cross entropy | Last-frame prediction, Hough-circle decoding, configured evaluation tolerance |
| TrackNet V2 | `tracknet.models.tracknet_v2.TrackNetV2` | 3 RGB frames, 512x288 default | 3 sigmoid heatmaps, WBCE | Weighted window aggregation, largest-blob centroid, 4 px tolerance |
| TrackNet V3 tracker | `tracknet.models.tracknet_v3.TrackNetV3Tracker` | 8 RGB frames plus match background | Binary-disk heatmaps, WBCE, video mixup | Center-weighted aggregation, optional rectifier support |
| TrackNet V3 rectifier | `tracknet.models.tracknet_v3.TrajectoryRectifier` | Trajectory windows `[x,y,visibility,mask]` | Masked trajectory MSE | Repairs raw-coordinate tracker trajectories |
| TrackNet V4 | `tracknet.models.tracknet_v4.TrackNetV4` | 3 RGB frames | Gaussian heatmaps, WBCE | Motion attention fusion, largest-blob centroid |
| TrackNet V5 | `tracknet.models.tracknet_v5.TrackNetV5` | 3 RGB frames with MDD motion channels | Binary-disk heatmaps, WBCE | Residual spatio-temporal refinement, largest-blob centroid |

The pipeline selects behavior through `tracknet.papers.PaperSpec`. Dataset configs describe sampling only; heatmap semantics, target frames, post-processing, and aggregation live in the paper contracts.

## Repository Layout

```text
tracknet/
  data/          # raw adapters, preprocessing, processed datasets, heatmaps
  models/        # TrackNet V1/V2/V3/V4/V5 and model registry
  papers/        # paper-specific contracts and defaults
  training/      # trainer, losses, metrics, checkpoints, TensorBoard
  evaluation/    # checkpoint evaluation and metric export
  inference/     # video prediction, aggregation, post-processing, rectification
  tools/         # CLI entry points
configs/         # runnable preprocessing, training, evaluation, inference configs
papers/          # source papers and extracted text
scripts/         # utility scripts for synthetic data
tests/           # synthetic unit and smoke tests
model_results/   # tracked training logs and evaluation reports
```

Generated data and heavyweight artifacts are intentionally separated from source code:

```text
dataset/raw/          # raw external dataset, ignored by Git
dataset/processed/    # processed dataset, ignored by Git
outputs/train/        # local training runs and checkpoints, ignored by Git
model_results/        # tracked logs, metrics, reports, and evaluation artifacts
```

## Environment

Use the `tracknet` conda environment unless you are intentionally building a different runtime.

Install and run from source:

```bash
conda activate tracknet
pip install -r requirements.txt
pip install -e .
```

Python 3.11 is the maintained environment for this workspace. Do not reinstall or replace PyTorch unless you explicitly need to rebuild the CUDA stack.

Hardware inspection:

```bash
python -m tracknet.tools.hardware
```

## Raw Dataset Layout

The default adapter is `tracknet_domain`, which expects the public TrackNet-style badminton dataset under `dataset/raw/`:

```text
dataset/raw/
  Professional/
    match1/
      video/
        rally1.mp4
      csv/
        rally1_ball.csv
  Amateur/
    match1/
      video/
      csv/
  Test/
    match1/
      video/
      csv/
```

CSV files are normalized to the semantic columns:

```text
Frame, Visibility, X, Y
```

`Visibility == 1` means the shuttlecock is visible and `X/Y` are valid raw-video coordinates. Invisible, missing, NaN, or negative coordinates are treated as invisible for training targets.

## Preprocessing

Preprocessing creates neutral frame and coordinate records. It does not write paper-specific heatmaps; those are generated on demand during training and evaluation.

Run default 512x288 preprocessing for V2-V5:

```bash
python -m tracknet.tools.preprocess --config configs/preprocess.yaml
```

Run V1 640x360 preprocessing:

```bash
python -m tracknet.tools.preprocess --config configs/preprocess_v1_640x360.yaml
```

Run high-worker variants for full-machine preprocessing:

```bash
python -m tracknet.tools.preprocess --config configs/preprocess_fast_512x288.yaml
python -m tracknet.tools.preprocess --config configs/preprocess_fast_v1_640x360.yaml
```

Processed output structure:

```text
dataset/processed/tracknet_dataset_512x288/
  manifest.json
  backgrounds/
  splits/
    train.txt
    val.txt
    test.txt
  sequences/
    <domain>__<match>__<rally>/
      frames/
      annotations.csv
      sequence_median.png
      meta.json
```

Each processed annotation row stores both raw and model-space coordinates:

```text
frame, visibility, x_raw, y_raw, x_model, y_model, frame_file
```

## Training

Training requires explicit split files produced by preprocessing. The default full-training configs use 30 epochs with AMP disabled.

Run single-process training:

```bash
python -m tracknet.tools.train --config configs/train_v1.yaml
python -m tracknet.tools.train --config configs/train_v2.yaml
python -m tracknet.tools.train --config configs/train_v3_tracker.yaml
python -m tracknet.tools.train --config configs/train_v3_rectifier.yaml
python -m tracknet.tools.train --config configs/train_v4.yaml
python -m tracknet.tools.train --config configs/train_v5.yaml
```

Run two-GPU DDP training:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 -m tracknet.tools.train --config configs/train_v4.yaml
```

`train.batch_size` is per GPU / per DDP rank:

```text
global_batch_size = train.batch_size * WORLD_SIZE
```

Training outputs:

```text
outputs/train/<experiment>_<timestamp>/
  config.resolved.json
  metrics.last.json
  tensorboard/
  checkpoints/
    ...
```

Evaluation configs point to completed 30-epoch trained checkpoints from these run directories.

Resume from a run directory:

```yaml
train:
  resume: outputs/train/tracknet_v2_20260511_003832
```

Resume from a specific checkpoint:

```yaml
train:
  resume_checkpoint: outputs/train/tracknet_v2_20260511_003832/checkpoints/last.pt
  output_root: outputs/train
```

## Evaluation

Each evaluation config evaluates one checkpoint and writes one independent result directory under `model_results/evaluation/`.

Run the full evaluation set:

```bash
python -m tracknet.tools.evaluate --config configs/evaluate_v1.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v2.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v3_tracker.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v3_tracker_rectifier.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v4.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v5.yaml
```

Evaluation output:

```text
model_results/evaluation/<model>/
  metrics.json
  metrics.by_sequence.json
  protocol.json
  evaluation.resolved.json
  checkpoint.json
  predictions.csv
```

Regenerate the aggregate report:

```bash
python -m tracknet.tools.collect_evaluations
```

### Completed Evaluation Results

The tracked results below were produced from completed 30-epoch trained checkpoints listed in `model_results/TRAINING_SUMMARY.md`. They are reproducible repository results, not a claim of paper-level reproduction.

| Model | Coordinate space | Accuracy | Precision | Recall | F1 | Total frames | Protocol summary |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| TrackNet V1 | model | 0.6729 | 0.9978 | 0.6066 | 0.7545 | 12,600 | 640x360, Hough threshold 128, 7.5 px tolerance |
| TrackNet V2 | model | 0.7299 | 0.9975 | 0.6749 | 0.8051 | 12,658 | 512x288, threshold 0.5, 4 px tolerance |
| TrackNet V3 tracker + rectifier | raw | 0.7071 | 0.8171 | 0.8019 | 0.8094 | 12,658 | Raw-coordinate rectified trajectory, 4 px tolerance |
| TrackNet V4 | model | 0.7259 | 0.9979 | 0.6698 | 0.8016 | 12,658 | 512x288, motion fusion, 4 px tolerance |
| TrackNet V5 | model | 0.6940 | 0.8788 | 0.6903 | 0.7732 | 12,658 | 512x288 public protocol, 4 px tolerance |

Full protocol, confusion counts, checkpoint paths, and artifact descriptions are documented in [`model_results/EVALUATION_RESULTS.md`](model_results/EVALUATION_RESULTS.md).

## Video Inference

Edit `configs/predict_video.yaml` with a real input video and checkpoint path:

```yaml
inference:
  video_path: dataset/raw/Test/match1/video/rally1.mp4
  checkpoint_path: model-v2.0.1-epoch30/tracknet_v2_epoch30.pt
  output_csv: outputs/predict/rally1_predictions.csv
  output_video: outputs/predict/rally1_overlay.mp4
  target_width: 512
  target_height: 288
  sequence_length: 3
  threshold: 0.5
  batch_size: 4
  device: auto
  progress: true
  verbose: true
```

Run:

```bash
python -m tracknet.tools.predict_video --config configs/predict_video.yaml
```

The CSV output is frame-preserving and uses the original video coordinate system:

```text
Frame,Visibility,X,Y
0,1,123,45
1,0,-1,-1
```

Inference prints a short run summary, a window-level prediction progress bar, and an overlay-writing progress bar when `output_video` is enabled. OpenCV writes the overlay as a video-only stream, so source audio is not preserved.

V3 rectification can be enabled by adding:

```yaml
inference:
  rectifier_checkpoint_path: model-v2.0.1-epoch30/tracknet_v3_rectifier_epoch30.pt
  rectifier_sequence_length: 16
  rectifier_delta_y_pixels: 30.0
```

## Visualization

Inspect processed samples, coordinate mappings, and heatmap overlays:

```bash
python -m tracknet.tools.visualize_dataset --config configs/visualize_dataset.yaml
```

## Tests

The test suite uses synthetic data and small temporary models; it does not require the real dataset or large checkpoints.

```bash
python -m pytest -q
```

Useful smoke commands:

```bash
python scripts/make_synthetic_raw.py --output test_results/dataset/raw
python -m tracknet.tools.preprocess --config configs/mac_smoke_preprocess.yaml
python -m tracknet.tools.train --config configs/mac_smoke_train_v2.yaml
```

Coverage includes raw discovery, CSV normalization, letterbox coordinate round trips, heatmap targets, model forward passes, losses, checkpoint save/load, DDP-safe training paths, evaluation aggregation, and frame-preserving video inference.

## Artifact Policy

Real datasets, processed frames, local training runs, checkpoints, virtual environments, caches, and temporary test outputs are excluded from source control. The repository tracks source code, configs, tests, papers, TensorBoard log exports, and evaluation summaries needed to understand and reproduce the current results.
