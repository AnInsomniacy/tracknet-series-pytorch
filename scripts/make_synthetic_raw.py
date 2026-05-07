"""Create a tiny legacy-layout raw dataset for local smoke tests.

This script is optional and does not download data. It writes:
  <output>/match1/video/rally1.mp4
  <output>/match1/csv/rally1_ball.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("test_results/synthetic_raw"))
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    video_dir = args.output / "match1" / "video"
    csv_dir = args.output / "match1" / "csv"
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "rally1.mp4"
    w, h = 32, 24
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")
    rows = []
    for i in range(args.frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x, y = min(w - 3, 4 + i * 3), min(h - 3, 6 + i)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
        writer.write(frame)
        rows.append({"Frame": i, "Visibility": 1, "X": x, "Y": y})
    writer.release()
    pd.DataFrame(rows).to_csv(csv_dir / "rally1_ball.csv", index=False)
    print(f"Wrote synthetic raw dataset to {args.output}")


if __name__ == "__main__":
    main()
