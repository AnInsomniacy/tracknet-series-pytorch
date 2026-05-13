# TrackNet Series Evaluation Report

This report summarizes the tracked evaluation results for the completed 30-epoch TrackNet V1-V5 training runs.

The results are reproducible repository outputs from the current implementation and dataset processing pipeline. They should not be presented as paper-level reproduction results.

## Scope

- Evaluated models: TrackNet V1, V2, V3 tracker plus rectifier, V4, and V5.
- Checkpoint policy: completed 30-epoch trained checkpoint for each model.
- Dataset split: processed `splits/test.txt`.
- Output root: `model_results/evaluation/`.
- Evaluation command: `python -m tracknet.tools.evaluate --config <config>`.
- Aggregate report command: `python -m tracknet.tools.collect_evaluations`.

## Protocol Summary

- TrackNet V1 uses the badminton protocol: 640x360 inputs, Hough threshold 128, and 7.5 px tolerance.
- TrackNet V2, V4, and V5 use 512x288 inputs, heatmap threshold 0.5, largest-blob centroid decoding, and 4 px tolerance.
- TrackNet V3 is reported as tracker plus trajectory rectifier in raw coordinate space.
- TrackNet V5 is evaluated with the public TrackNetV2-sized protocol used by this repository.

## Summary

| Model | Checkpoint | Coordinate Space | Accuracy | Precision | Recall | F1 | TP | TN | FP1 | FP2 | FN | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tracknet_v1 | `outputs/train/tracknet_v1_20260511_120015/checkpoints/last.pt` | model | 0.6729 | 0.9978 | 0.6066 | 0.7545 | 6334 | 2145 | 8 | 6 | 4107 | 12600 |
| tracknet_v2 | `outputs/train/tracknet_v2_20260511_003832/checkpoints/last.pt` | model | 0.7299 | 0.9975 | 0.6749 | 0.8051 | 7060 | 2179 | 7 | 11 | 3401 | 12658 |
| tracknet_v3_tracker_rectifier | `outputs/train/tracknet_v3_tracker_20260512_003925/checkpoints/last.pt` | raw | 0.7071 | 0.8171 | 0.8019 | 0.8094 | 7872 | 1079 | 651 | 1111 | 1945 | 12658 |
| tracknet_v4 | `outputs/train/tracknet_v4_20260511_002432/checkpoints/last.pt` | model | 0.7259 | 0.9979 | 0.6698 | 0.8016 | 7009 | 2179 | 4 | 11 | 3455 | 12658 |
| tracknet_v5 | `outputs/train/tracknet_v5_20260511_002455/checkpoints/last.pt` | model | 0.6940 | 0.8788 | 0.6903 | 0.7732 | 6603 | 2182 | 903 | 8 | 2962 | 12658 |

## Interpretation

TrackNet V3 tracker plus rectifier has the highest F1 score in this evaluation set. It also has the highest recall, but it is evaluated in raw coordinate space and has substantially lower precision than V1, V2, and V4.

TrackNet V1, V2, and V4 are highly conservative under their current decoding protocols: precision is near 1.0, while recall is materially lower. This indicates that these models rarely emit incorrect visible detections, but still miss many visible shuttlecock frames.

TrackNet V5 improves recall relative to the most conservative 3-frame models, but its precision is lower because of a larger `fp1` count. This result does not establish paper-level performance for the V5 full model.

Overall, the evaluation confirms that preprocessing, checkpoint loading, paper-specific decoding, metric export, and report collection are operational for the five reported TrackNet versions. The results remain repository reproduction metrics and should be interpreted together with the exact protocols above.

## Artifacts

Each evaluation directory contains:

- `metrics.json`: aggregate metrics.
- `metrics.by_sequence.json`: per-sequence metrics.
- `protocol.json`: resolved evaluation protocol.
- `evaluation.resolved.json`: resolved model, dataset, runtime, and rectifier settings.
- `checkpoint.json`: checkpoint metadata used for evaluation.
- `predictions.csv`: frame-level predictions and outcomes.

## Reproduction Commands

Run evaluations independently:

```bash
python -m tracknet.tools.evaluate --config configs/evaluate_v1.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v2.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v3_tracker_rectifier.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v4.yaml
python -m tracknet.tools.evaluate --config configs/evaluate_v5.yaml
```

Regenerate aggregate files:

```bash
python -m tracknet.tools.collect_evaluations
```
