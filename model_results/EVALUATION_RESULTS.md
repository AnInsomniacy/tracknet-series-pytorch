# TrackNet Series Evaluation Results

This file summarizes the tracked evaluation results for the trained TrackNet V1-V5 checkpoints.

## Evaluation Protocol

- All entries use the best-validation checkpoint for the corresponding trained model.
- All entries evaluate the processed `splits/test.txt` split.
- TrackNet V1 uses the V1 protocol: 640x360 inputs, Hough threshold 128, and 5 px tolerance.
- TrackNet V2, V4, and V5 use the TrackNetV2-sized protocol: 512x288 inputs, heatmap threshold 0.5, largest-blob centroid decoding, and 4 px tolerance.
- TrackNet V3 is reported both as tracker-only and as tracker plus trajectory rectifier.
- TrackNet V5 is evaluated with the public TrackNetV2-sized protocol used by this repository.

## Summary

| Model | Checkpoint | Coordinate Space | Accuracy | Precision | Recall | F1 | TP | TN | FP1 | FP2 | FN | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tracknet_v1 | `outputs/train/tracknet_v1_20260511_120015/checkpoints/model_best.pt` | model | 0.6390 | 0.9939 | 0.5669 | 0.7220 | 5906 | 2146 | 31 | 5 | 4512 | 12600 |
| tracknet_v2 | `outputs/train/tracknet_v2_20260511_003832/checkpoints/model_best.pt` | model | 0.7091 | 0.9939 | 0.6513 | 0.7869 | 6799 | 2177 | 29 | 13 | 3640 | 12658 |
| tracknet_v3_tracker | `outputs/train/tracknet_v3_tracker_20260512_003925/checkpoints/model_best.pt` | model | 0.7664 | 0.9913 | 0.7235 | 0.8365 | 7564 | 2137 | 13 | 53 | 2891 | 12658 |
| tracknet_v3_tracker_rectifier | `outputs/train/tracknet_v3_tracker_20260512_003925/checkpoints/model_best.pt` | raw | 0.6875 | 0.8655 | 0.7154 | 0.7833 | 7148 | 1555 | 476 | 635 | 2844 | 12658 |
| tracknet_v4 | `outputs/train/tracknet_v4_20260511_002432/checkpoints/model_best.pt` | model | 0.7022 | 0.9910 | 0.6443 | 0.7809 | 6717 | 2172 | 43 | 18 | 3708 | 12658 |
| tracknet_v5 | `outputs/train/tracknet_v5_20260511_002455/checkpoints/model_best.pt` | model | 0.6928 | 0.8175 | 0.7347 | 0.7739 | 6654 | 2116 | 1411 | 74 | 2403 | 12658 |

## Artifacts

Each evaluation directory contains:

- `metrics.json`: aggregate metrics.
- `metrics.by_sequence.json`: per-sequence metrics.
- `protocol.json`: resolved evaluation protocol.
- `evaluation.resolved.json`: resolved model, dataset, runtime, and rectifier settings.
- `checkpoint.json`: checkpoint metadata used for evaluation.
- `predictions.csv`: frame-level predictions and outcomes.
