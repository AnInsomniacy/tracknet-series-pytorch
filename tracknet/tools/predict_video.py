"""Run frame-accurate video inference and write CSV/video outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracknet.config import load_yaml
from tracknet.inference import VideoPredictionConfig, run_video_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict shuttle/ball locations for a video")
    parser.add_argument("--config", type=Path, required=True, help="YAML config with inference section")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    section = cfg.get("inference")
    if not isinstance(section, dict):
        raise KeyError("Missing config section: inference")
    icfg = VideoPredictionConfig(
        video_path=Path(section["video_path"]),
        checkpoint_path=Path(section["checkpoint_path"]),
        output_csv=Path(section["output_csv"]),
        output_video=Path(section["output_video"]) if section.get("output_video") else None,
        model=section.get("model"),
        target_width=int(section.get("target_width", 512)),
        target_height=int(section.get("target_height", 288)),
        sequence_length=int(section.get("sequence_length", 3)),
        threshold=float(section.get("threshold", 0.5)),
        hough_threshold=int(section.get("hough_threshold", 128)),
        batch_size=int(section.get("batch_size", 4)),
        device=str(section.get("device", "auto")),
        frame_index_base=int(section.get("frame_index_base", 0)),
        trail_length=int(section.get("trail_length", 12)),
        rectifier_checkpoint_path=Path(section["rectifier_checkpoint_path"]) if section.get("rectifier_checkpoint_path") else None,
        rectifier_model=section.get("rectifier_model"),
        rectifier_sequence_length=int(section.get("rectifier_sequence_length", 16)),
        rectifier_delta_y_pixels=float(section.get("rectifier_delta_y_pixels", 30.0)),
        background_sample_stride=int(section.get("background_sample_stride", 8)),
        max_background_samples=int(section.get("max_background_samples", 512)),
        num_threads=int(section["num_threads"]) if section.get("num_threads") is not None else None,
        interop_threads=int(section["interop_threads"]) if section.get("interop_threads") is not None else None,
    )
    df = run_video_prediction(icfg)
    print(f"Wrote {len(df)} frame predictions to {icfg.output_csv}")


if __name__ == "__main__":
    main()
