# TrackNet Series Industrial PyTorch

A clean PyTorch project for the TrackNet model series. The legacy repository is used only to identify the raw dataset layout and historical experiment intent; this project does not keep the legacy entry points or directory structure.

## Project Goals

The project separates the TrackNet workflow into independently runnable and testable stages:

1. Read the legacy raw video/CSV layout and preprocess it into a stable processed dataset.
2. Train V1/V2/V3/V4/V5 heatmap models or the V3 trajectory rectifier from the processed dataset.
3. Resume training from full checkpoints.
4. Evaluate checkpoints with frame-level aggregation and export metrics plus prediction CSV files.
5. Run video inference and export `Frame, Visibility, X, Y` in the original video coordinate system, optionally with overlay video.
6. Visualize processed samples to inspect heatmaps and coordinate mappings.
7. Test data handling, shapes, losses, checkpoints, evaluation and inference post-processing with synthetic data and small models.

## Legacy Raw Data Layout

The raw reader expects the layout below. Users do not need to reorganize the original raw data.

```text
raw_root/
  match1/
    video/
      rally1.mp4
      rally2.mp4
    csv/
      rally1_ball.csv
      rally2_ball.csv
  match2/
    video/
      ...
    csv/
      ...
```

CSV files must contain these semantic columns. A few historical case/name variants are accepted and normalized:

```text
Frame, Visibility, X, Y
```

Coordinate conventions:

- `Frame` is treated as a zero-based video frame index by default; filenames such as `000123.jpg` are parsed by extracting the numeric part.
- `Visibility == 1` means the object is visible and `X/Y` is valid.
- `Visibility != 1`, missing annotations, NaN values and negative coordinates are treated as invisible for heatmap supervision.
- `X/Y` always means `(x, y)` in the original video coordinate system; arrays and heatmaps are always `[H, W]`.

## Processed Data Layout

Preprocessing writes a stable project-native format:

```text
processed_root/
  manifest.json
  backgrounds/
    match1.png
  sequences/
    match1__rally1/
      frames/
        000000.png
        000001.png
      annotations.csv
      sequence_median.png
      meta.json
```

`manifest.json` records target size, sequence metadata and raw sources. Each `annotations.csv` contains:

```text
frame, visibility, x_raw, y_raw, x_model, y_model, frame_file
```

Fields:

- `x_raw/y_raw` are original video coordinates.
- `x_model/y_model` are coordinates after letterboxing into model resolution.
- Training, evaluation and inference share the same letterbox transform to avoid resize/coordinate drift.
- Heatmaps are generated on demand by the dataset, so all model versions can share one processed dataset while keeping model-specific target semantics.

## Paper-to-Code Mapping

| Paper | Module | Key implementation |
|---|---|---|
| V1 | `tracknet.models.tracknet_v1.TrackNetV1` | VGG/DeconvNet-style encoder-decoder without U-Net skips; three-frame input predicts the last frame by default; 256-bin grayscale heatmap logits; `CrossEntropyLoss`; Gaussian target scaled to `[0,255]`; post-processing uses threshold 128 followed by Hough circle detection, accepting exactly one circle. |
| V2 | `tracknet.models.tracknet_v2.TrackNetV2` | 512x288 3-in/3-out U-Net; three sigmoid heatmaps; WBCE; threshold plus largest-blob centroid; default tolerance 4 px. |
| V3 tracker | `tracknet.models.tracknet_v3.TrackNetV3Tracker` | Default eight-frame input plus match-level median background; binary-disk heatmaps; WBCE; overlapping inference uses center-weighted aggregation; preprocessing writes sequence and match median backgrounds. |
| V3 rectifier | `tracknet.models.tracknet_v3.TrajectoryRectifier` | 1D U-Net trajectory repair; input channels `[x,y,visibility,mask]`; coordinates normalized to `[0,1]`; training masks valid points by ratio; inference builds an inpainting mask with the `delta_y=30` missing-interval rule; masked MSE. |
| V4 | `tracknet.models.tracknet_v4.TrackNetV4` | Absolute grayscale differencing; learnable motion prompt layer; Eq. (4) fusion via `fusion_variant=eq4` plus the mean-attention ablation via `fusion_variant=mean`; sigmoid heatmaps and WBCE. |
| V5 | `tracknet.models.tracknet_v5.TrackNetV5` | MDD signed grayscale polarity decoupling; 13-channel input; V2-style coarse draft; R-STR residual refinement; training-only stochastic context dropout; AdamW + MultiStepLR config; binary-disk radius configurable as 30 or 40. |

### Engineering Judgments

- V1 uses 640x360, while later TrackNet variants commonly use 512x288. Target size is configured during preprocessing instead of hard-coded into model code.
- V1 states `sigma^2=10`; the config uses `sigma=sqrt(10)`.
- V1 does not fully specify OpenCV Hough parameters. The implementation preserves the paper threshold 128 and centralizes radius parameters in one helper.
- V3 binary radius is described as based on average shuttlecock size. Because that cannot be inferred reliably for arbitrary data, `radius` is a config value.
- V3 training background uses rally medians followed by a match median. Single-video inference lacks match context, so it uses that video median as a self-contained approximation.
- V4 describes Eq. (4) fusion and a mean-attention variant; both are selected through `fusion_variant`.
- V5 reports radius 40 for its high-resolution internal data and radius 30 for TrackNetV2-sized data; configs default to 30 and can be adjusted.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run directly from source:

```bash
export PYTHONPATH=.
```

Or install as an editable package:

```bash
pip install -e .
```

## Preprocessing

Edit `configs/preprocess.yaml`:

```yaml
preprocess:
  raw_root: data/raw
  output_root: data/processed/tracknet_512x288
  target_width: 512
  target_height: 288
```

Run:

```bash
python -m tracknet.tools.preprocess --config configs/preprocess.yaml
```

For V1 at 640x360, create a separate preprocess config with `target_width: 640` and `target_height: 360`, then point `configs/train_v1.yaml` at that processed root.

## Training

V2 example:

```bash
python -m tracknet.tools.train --config configs/train_v2.yaml
```

V1/V3/V4/V5:

```bash
python -m tracknet.tools.train --config configs/train_v1.yaml
python -m tracknet.tools.train --config configs/train_v3_tracker.yaml
python -m tracknet.tools.train --config configs/train_v3_rectifier.yaml
python -m tracknet.tools.train --config configs/train_v4.yaml
python -m tracknet.tools.train --config configs/train_v5.yaml
```

Output layout:

```text
outputs/train/<experiment>_<timestamp>/
  config.resolved.json
  metrics.last.json
  checkpoints/
    last.pt          # latest full training state for resume
    best.pt          # full training state with best validation loss
    model_best.pt    # model weights plus config for evaluation/inference
```


### Validation Splits and AMP

Training uses sequence-level validation splitting when multiple sequences are available. This prevents overlapping sliding windows from the same rally from appearing in both training and validation. Tiny one-sequence smoke datasets use a contiguous window split and record that limitation in checkpoint metadata.

Set `train.amp: true` to enable mixed precision on CUDA. The same config is safe on CPU; checkpoints record both `amp_requested` and `amp_enabled` so experiments remain auditable.

## Resume Training

Set this in the training config:

```yaml
train:
  resume: outputs/train/tracknet_v2_20260505_120000
```

By default this reads:

```text
<resume>/checkpoints/last.pt
```

A specific checkpoint can also be supplied:

```yaml
train:
  resume_checkpoint: outputs/train/tracknet_v2_20260505_120000/checkpoints/best.pt
  output_root: outputs/train
```

## Multi-GPU Training

The training entry point supports PyTorch DDP. Example for 8 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 -m tracknet.tools.train --config configs/train_v2.yaml
```

Distributed settings are inferred from `RANK/LOCAL_RANK/WORLD_SIZE`. CPU-only smoke tests can run with ordinary `python -m ...` commands.

## Evaluation

Edit `configs/evaluate.yaml` so checkpoint, dataset and model settings match training:

```bash
python -m tracknet.tools.evaluate --config configs/evaluate.yaml
```

Outputs:

```text
outputs/eval/<name>/
  metrics.json
  predictions.csv
```

Evaluation uses the common TrackNet confusion categories:

- `tp`: prediction visible, ground truth visible, distance within tolerance.
- `tn`: prediction invisible, ground truth invisible.
- `fn`: prediction invisible, ground truth visible.
- `fp2`: prediction visible, ground truth invisible.
- `fp1`: prediction visible, ground truth visible, but distance exceeds tolerance.

V2/V4/V5 default to a 4 px tolerance. V1 tennis/badminton tolerances can be configured according to the paper setting.

## Video Inference

Edit `configs/predict_video.yaml`:

```yaml
inference:
  video_path: data/raw/match1/video/rally1.mp4
  checkpoint_path: outputs/train/tracknet_v2/checkpoints/best.pt
  output_csv: outputs/predict/rally1_predictions.csv
  output_video: outputs/predict/rally1_overlay.mp4
  target_width: 512
  target_height: 288
  sequence_length: 3
  threshold: 0.5
```

Run:

```bash
python -m tracknet.tools.predict_video --config configs/predict_video.yaml
```

CSV output is in the original video coordinate system:

```text
Frame,Visibility,X,Y
0,1,123,45
1,0,-1,-1
```

Inference is frame-preserving: CSV row count matches the input frame count, and overlay video keeps input order, size and FPS. Overlapping MIMO window predictions are reduced with center-weighted heatmap aggregation; V1 uses prefix padding for early frames.

The V3 rectifier can be enabled during video inference:

```yaml
inference:
  rectifier_checkpoint_path: outputs/train/tracknet_v3_rectifier/checkpoints/best.pt
  rectifier_sequence_length: 16
  rectifier_delta_y_pixels: 30.0
```

## Processed Data Visualization

```bash
python -m tracknet.tools.visualize_dataset --config configs/visualize_dataset.yaml
```

This writes heatmap overlay PNGs for inspecting processed size, letterbox padding, coordinates and target semantics.

## Tests

The test suite does not require real datasets or large weights. It builds synthetic raw data, small models and temporary checkpoints:

```bash
PYTHONPATH=. pytest -q
```

Coverage includes:

- raw discovery and CSV normalization;
- letterbox coordinate round trips;
- Gaussian/binary heatmaps and largest-blob centroids;
- sliding-window dataset shapes;
- V1/V2/V3/V4/V5/rectifier forward passes;
- WBCE and trajectory masked MSE;
- checkpoint save/load;
- evaluation classification and metrics;
- frame-preserving video inference CSV output;
- minimal training smoke tests and best/last/model_best checkpoints;
- sequence-safe validation splitting;
- AMP metadata and CPU fallback;
- frame-level evaluation aggregation.

A tiny raw dataset can also be generated manually:

```bash
python scripts/make_synthetic_raw.py --output test_results/synthetic_raw
```

Then point `configs/preprocess.yaml` at that directory for CLI smoke runs.

## Directory Layout

```text
tracknet/
  data/          # raw reader, preprocess, heatmap, datasets
  models/        # V1/V2/V3/V4/V5 and registry
  training/      # losses, trainer, checkpoint, metrics
  evaluation/    # checkpoint evaluation
  inference/     # postprocess, video prediction, V3 rectification, visualization
  tools/         # CLI entry points
configs/         # runnable YAML examples
papers/          # v1.pdf ... v5.pdf
scripts/         # optional synthetic raw generator
tests/           # synthetic tests and smoke tests
```

## Excluded From Release Archives

Release archives exclude real datasets, training outputs, large model weights, `.git`, virtual environments, caches, `__pycache__`, `.DS_Store` and temporary test results.
