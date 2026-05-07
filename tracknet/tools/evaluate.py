"""Evaluate a checkpoint on a processed split."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracknet.config import load_yaml
from tracknet.evaluation import EvaluationConfig, evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrackNet checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="YAML config with evaluation section")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    section = cfg.get("evaluation")
    if not isinstance(section, dict):
        raise KeyError("Missing config section: evaluation")
    ecfg = EvaluationConfig(
        checkpoint_path=Path(section["checkpoint_path"]),
        output_dir=Path(section["output_dir"]),
        dataset=dict(section["dataset"]),
        model=section.get("model"),
        batch_size=int(section.get("batch_size", 4)),
        workers=int(section.get("workers", 2)),
        device=str(section.get("device", "auto")),
        threshold=float(section.get("threshold", 0.5)),
        hough_threshold=int(section.get("hough_threshold", 128)),
        tolerance_pixels=float(section.get("tolerance_pixels", 4.0)),
        num_threads=int(section["num_threads"]) if section.get("num_threads") is not None else None,
        interop_threads=int(section["interop_threads"]) if section.get("interop_threads") is not None else None,
    )
    result = evaluate_checkpoint(ecfg)
    print(f"Metrics: {result.metrics}")
    print(f"Predictions: {result.predictions_csv}")


if __name__ == "__main__":
    main()
