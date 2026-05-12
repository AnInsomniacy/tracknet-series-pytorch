"""Collect independent evaluation outputs into release summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from tracknet.utils.io import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TrackNet evaluation result directories")
    parser.add_argument("--root", type=Path, default=Path("model_results/evaluation"), help="Directory containing per-model evaluation folders")
    parser.add_argument("--report", type=Path, default=Path("model_results/EVALUATION_RESULTS.md"), help="Markdown report path")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional summary CSV path")
    return parser.parse_args()


def _fmt_metric(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def _fmt_count(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(int(float(value)))


def collect_evaluation_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        raise FileNotFoundError(f"Evaluation root not found: {root}")
    for result_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metrics_path = result_dir / "metrics.json"
        protocol_path = result_dir / "protocol.json"
        resolved_path = result_dir / "evaluation.resolved.json"
        if not metrics_path.exists() or not protocol_path.exists() or not resolved_path.exists():
            continue
        metrics = read_json(metrics_path)
        protocol = read_json(protocol_path)
        resolved = read_json(resolved_path)
        row = {
            "name": result_dir.name,
            "output_dir": str(result_dir),
            "checkpoint_path": str(resolved.get("checkpoint_path", "")),
            "coordinate_space": str(protocol.get("coordinate_space", "")),
            "predictions_csv": str(result_dir / "predictions.csv"),
            "metrics_json": str(metrics_path),
        }
        row.update(metrics)
        rows.append(row)
    if not rows:
        raise ValueError(f"No complete evaluation results found in {root}")
    return rows


def write_evaluation_report(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a concise report for tracked evaluation artifacts."""

    ensure_dir(path.parent)
    lines = [
        "# TrackNet Series Evaluation Results",
        "",
        "This file summarizes the tracked evaluation results for the trained TrackNet V1-V5 checkpoints.",
        "",
        "## Evaluation Protocol",
        "",
        "- All entries use the best-validation checkpoint for the corresponding trained model.",
        "- All entries evaluate the processed `splits/test.txt` split.",
        "- TrackNet V1 uses the V1 protocol: 640x360 inputs, Hough threshold 128, and 5 px tolerance.",
        "- TrackNet V2, V4, and V5 use the TrackNetV2-sized protocol: 512x288 inputs, heatmap threshold 0.5, largest-blob centroid decoding, and 4 px tolerance.",
        "- TrackNet V3 is reported both as tracker-only and as tracker plus trajectory rectifier.",
        "- TrackNet V5 is evaluated with the public TrackNetV2-sized protocol used by this repository.",
        "",
        "## Summary",
        "",
        "| Model | Checkpoint | Coordinate Space | Accuracy | Precision | Recall | F1 | TP | TN | FP1 | FP2 | FN | Total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    f"`{row.get('checkpoint_path', '')}`",
                    str(row.get("coordinate_space", "")),
                    _fmt_metric(row.get("accuracy")),
                    _fmt_metric(row.get("precision")),
                    _fmt_metric(row.get("recall")),
                    _fmt_metric(row.get("f1")),
                    _fmt_count(row.get("tp")),
                    _fmt_count(row.get("tn")),
                    _fmt_count(row.get("fp1")),
                    _fmt_count(row.get("fp2")),
                    _fmt_count(row.get("fn")),
                    _fmt_count(row.get("total")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "Each evaluation directory contains:",
            "",
            "- `metrics.json`: aggregate metrics.",
            "- `metrics.by_sequence.json`: per-sequence metrics.",
            "- `protocol.json`: resolved evaluation protocol.",
            "- `evaluation.resolved.json`: resolved model, dataset, runtime, and rectifier settings.",
            "- `checkpoint.json`: checkpoint metadata used for evaluation.",
            "- `predictions.csv`: frame-level predictions and outcomes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_evaluation_rows(args.root)
    summary_csv = args.summary_csv or (args.root / "summary.csv")
    ensure_dir(summary_csv.parent)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    write_json(summary_csv.with_suffix(".json"), rows)
    write_evaluation_report(args.report, rows)
    print(f"Summary: {summary_csv}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
