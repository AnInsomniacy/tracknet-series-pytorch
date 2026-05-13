# TrackNet Series Evaluation Report

This report documents the tracked evaluation artifacts for the trained TrackNet V1-V5 checkpoints in this repository.

The results are reproducible repository outputs from the current implementation and dataset processing pipeline. They should not be presented as paper-level reproduction results.

## Scope

- Evaluated models: TrackNet V1, V2, V3 tracker, V3 tracker plus rectifier, V4, and V5.
- Checkpoint policy: best-validation `model_best.pt` checkpoint for each trained model.
- Dataset split: processed `splits/test.txt`.
- Output root: `model_results/evaluation/`.
- Evaluation command: `python -m tracknet.tools.evaluate --config <config>`.
- Aggregate report command: `python -m tracknet.tools.collect_evaluations`.

## Result Files

```text
model_results/evaluation/
  summary.csv
  summary.json
  tracknet_v1/
  tracknet_v2/
  tracknet_v3_tracker/
  tracknet_v3_tracker_rectifier/
  tracknet_v4/
  tracknet_v5/
```

## Protocol Summary

| Model | Config | Dataset root | Sequence length | Coordinate space | Decoder / aggregation | Tolerance |
| --- | --- | --- | ---: | --- | --- | ---: |
| TrackNet V1 | `configs/evaluate_v1.yaml` | `dataset/processed/tracknet_dataset_640x360` | 3 | model | Last-frame V1 Hough-circle protocol, threshold 128 | 5 px |
| TrackNet V2 | `configs/evaluate_v2.yaml` | `dataset/processed/tracknet_dataset_512x288` | 3 | model | Weighted heatmap aggregation, threshold 0.5, largest-blob centroid | 4 px |
| TrackNet V3 tracker | `configs/evaluate_v3_tracker.yaml` | `dataset/processed/tracknet_dataset_512x288` | 8 | model | Center-weighted aggregation, threshold 0.5, largest-blob centroid | 4 px |
| TrackNet V3 tracker + rectifier | `configs/evaluate_v3_tracker_rectifier.yaml` | `dataset/processed/tracknet_dataset_512x288` | 8 | raw | Tracker output mapped to raw coordinates and repaired by trajectory rectifier | 4 px |
| TrackNet V4 | `configs/evaluate_v4.yaml` | `dataset/processed/tracknet_dataset_512x288` | 3 | model | Motion-fusion heatmap output, threshold 0.5, largest-blob centroid | 4 px |
| TrackNet V5 | `configs/evaluate_v5.yaml` | `dataset/processed/tracknet_dataset_512x288` | 3 | model | V5 full model output, threshold 0.5, largest-blob centroid | 4 px |

## Metric Definitions

Frame-level outcomes use the common TrackNet confusion categories:

- `tp`: prediction visible, ground truth visible, distance within tolerance.
- `tn`: prediction invisible, ground truth invisible.
- `fn`: prediction invisible, ground truth visible.
- `fp1`: prediction visible, ground truth visible, distance above tolerance.
- `fp2`: prediction visible, ground truth invisible.

Derived metrics:

```text
accuracy  = (tp + tn) / total
precision = tp / (tp + fp1 + fp2)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)
```

## Results

| Model | Coordinate Space | Accuracy | Precision | Recall | F1 | TP | TN | FP1 | FP2 | FN | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TrackNet V1 | model | 0.6390 | 0.9939 | 0.5669 | 0.7220 | 5,906 | 2,146 | 31 | 5 | 4,512 | 12,600 |
| TrackNet V2 | model | 0.7091 | 0.9939 | 0.6513 | 0.7869 | 6,799 | 2,177 | 29 | 13 | 3,640 | 12,658 |
| TrackNet V3 tracker | model | 0.7664 | 0.9913 | 0.7235 | 0.8365 | 7,564 | 2,137 | 13 | 53 | 2,891 | 12,658 |
| TrackNet V3 tracker + rectifier | raw | 0.6875 | 0.8655 | 0.7154 | 0.7833 | 7,148 | 1,555 | 476 | 635 | 2,844 | 12,658 |
| TrackNet V4 | model | 0.7022 | 0.9910 | 0.6443 | 0.7809 | 6,717 | 2,172 | 43 | 18 | 3,708 | 12,658 |
| TrackNet V5 | model | 0.6928 | 0.8175 | 0.7347 | 0.7739 | 6,654 | 2,116 | 1,411 | 74 | 2,403 | 12,658 |

## Checkpoints

| Model | Best-validation checkpoint used for evaluation |
| --- | --- |
| TrackNet V1 | `outputs/train/tracknet_v1_20260511_120015/checkpoints/model_best.pt` |
| TrackNet V2 | `outputs/train/tracknet_v2_20260511_003832/checkpoints/model_best.pt` |
| TrackNet V3 tracker | `outputs/train/tracknet_v3_tracker_20260512_003925/checkpoints/model_best.pt` |
| TrackNet V3 rectifier | `outputs/train/tracknet_v3_rectifier_20260512_003951/checkpoints/model_best.pt` |
| TrackNet V4 | `outputs/train/tracknet_v4_20260511_002432/checkpoints/model_best.pt` |
| TrackNet V5 | `outputs/train/tracknet_v5_20260511_002455/checkpoints/model_best.pt` |

## Interpretation

TrackNet V3 tracker is the strongest tracked model in this run by F1 score and accuracy. V1, V2, V3 tracker, and V4 all show very high precision with materially lower recall, which means these checkpoints are conservative: they rarely emit incorrect visible detections, but miss many visible shuttlecock frames.

The V3 rectifier evaluation is reported in raw coordinate space. In this run, rectification increases raw-coordinate trajectory coverage only modestly and lowers precision relative to tracker-only model-space evaluation. This is a separate protocol and should not be compared as a direct model-space replacement without considering the coordinate-space change.

TrackNet V5 has the highest recall among the non-rectified 3-frame variants, but its precision is substantially lower because of a high `fp1` count. The current V5 result therefore does not match the strong paper-reported profile for the full model.

Overall, these results validate that the end-to-end preprocessing, checkpoint loading, paper-specific decoding, metric export, and report collection paths are operational. They do not establish paper-level reproduction accuracy.

Each per-model directory contains:

- `metrics.json`: aggregate metrics and confusion counts.
- `metrics.by_sequence.json`: per-sequence metric breakdown.
- `protocol.json`: resolved threshold, tolerance, coordinate space, and rectifier support.
- `evaluation.resolved.json`: checkpoint path, dataset config, runtime config, model config, and rectifier config.
- `checkpoint.json`: checkpoint metadata available at evaluation time.
- `predictions.csv`: frame-level predictions, ground truth, scores, coordinate fields, and outcome labels.

## Reproduction Commands

Run evaluations independently:

```bash
python -m tracknet.tools.evaluate --config configs/evaluate_v1.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v2.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v3_tracker.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v3_tracker_rectifier.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v4.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v5.yaml
```

Regenerate aggregate files:

```bash
python -m tracknet.tools.collect_evaluations
```
