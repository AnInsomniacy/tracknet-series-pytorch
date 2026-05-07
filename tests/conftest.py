from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from tracknet.data.preprocessing import PreprocessConfig, preprocess_dataset


@pytest.fixture()
def synthetic_raw_root(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    video_dir = raw / "match1" / "video"
    csv_dir = raw / "match1" / "csv"
    video_dir.mkdir(parents=True)
    csv_dir.mkdir(parents=True)
    video_path = video_dir / "rally1.mp4"
    width, height = 32, 24
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    assert writer.isOpened()
    rows = []
    for i in range(8):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x, y = 4 + i * 3, 6 + i
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
        writer.write(frame)
        rows.append({"Frame": i, "Visibility": 1 if i != 3 else 0, "X": x if i != 3 else -1, "Y": y if i != 3 else -1})
    writer.release()
    pd.DataFrame(rows).to_csv(csv_dir / "rally1_ball.csv", index=False)
    return raw


@pytest.fixture()
def synthetic_processed_root(tmp_path: Path, synthetic_raw_root: Path) -> Path:
    processed = tmp_path / "processed"
    preprocess_dataset(PreprocessConfig(raw_root=synthetic_raw_root, output_root=processed, target_width=32, target_height=32, overwrite=True))
    return processed


@pytest.fixture()
def synthetic_processed_two_sequences_root(tmp_path: Path) -> Path:
    raw = tmp_path / "raw_two_sequences"
    for sequence_idx, sequence_name in enumerate(["rally1", "rally2"]):
        video_dir = raw / "match1" / "video"
        csv_dir = raw / "match1" / "csv"
        video_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{sequence_name}.mp4"
        width, height = 32, 24
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
        assert writer.isOpened()
        rows = []
        for frame_idx in range(8):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 4 + frame_idx * 2 + sequence_idx
            y = 6 + frame_idx
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
            writer.write(frame)
            rows.append({"Frame": frame_idx, "Visibility": 1, "X": x, "Y": y})
        writer.release()
        pd.DataFrame(rows).to_csv(csv_dir / f"{sequence_name}_ball.csv", index=False)
    processed = tmp_path / "processed_two_sequences"
    preprocess_dataset(PreprocessConfig(raw_root=raw, output_root=processed, target_width=32, target_height=32, overwrite=True))
    return processed
