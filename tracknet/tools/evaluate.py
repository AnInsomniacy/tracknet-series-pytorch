"""Evaluate a checkpoint on a processed split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from tracknet.config import load_yaml
from tracknet.evaluation import EvaluationConfig, evaluate_checkpoint
from tracknet.papers.base import CoordinateSpace


def evaluation_config_from_section(section: dict) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint_path=Path(section["checkpoint_path"]),
        output_dir=Path(section["output_dir"]),
        dataset=dict(section["dataset"]),
        model=section.get("model"),
        batch_size=int(section.get("batch_size", 4)),
        workers=int(section.get("workers", 2)),
        device=str(section.get("device", "auto")),
        threshold=float(section["threshold"]) if section.get("threshold") is not None else None,
        hough_threshold=int(section["hough_threshold"]) if section.get("hough_threshold") is not None else None,
        tolerance_pixels=float(section["tolerance_pixels"]) if section.get("tolerance_pixels") is not None else None,
        coordinate_space=cast(CoordinateSpace, str(section["coordinate_space"])) if section.get("coordinate_space") is not None else None,
        rectifier_checkpoint_path=Path(section["rectifier_checkpoint_path"]) if section.get("rectifier_checkpoint_path") else None,
        rectifier_model=section.get("rectifier_model"),
        rectifier_sequence_length=int(section.get("rectifier_sequence_length", 16)),
        rectifier_delta_y_pixels=float(section.get("rectifier_delta_y_pixels", 30.0)),
        num_threads=int(section["num_threads"]) if section.get("num_threads") is not None else None,
        interop_threads=int(section["interop_threads"]) if section.get("interop_threads") is not None else None,
    )


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
    ecfg = evaluation_config_from_section(section)
    result = evaluate_checkpoint(ecfg)
    print(f"Metrics: {result.metrics}")
    print(f"Predictions: {result.predictions_csv}")


if __name__ == "__main__":
    main()
