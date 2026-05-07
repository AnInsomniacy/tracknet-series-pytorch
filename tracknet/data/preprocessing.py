"""Build a deterministic processed dataset from legacy raw videos.

Processed data intentionally does not store old heatmap JPEGs. Instead it stores
RGB frames, transformed annotations and geometry metadata. Heatmaps are generated
on demand by model-specific datasets so TrackNetV1/V2/V3/V4/V5 can keep their
paper-specific target semantics without rebuilding raw data for every loss.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tracknet.constants import (
    PROCESSED_FRAME_COL,
    PROCESSED_VISIBILITY_COL,
    PROCESSED_X_MODEL_COL,
    PROCESSED_X_RAW_COL,
    PROCESSED_Y_MODEL_COL,
    PROCESSED_Y_RAW_COL,
    RAW_FRAME_COL,
    RAW_VISIBILITY_COL,
    RAW_X_COL,
    RAW_Y_COL,
)
from tracknet.data.raw_reader import RawSequence, discover_raw_sequences, load_raw_annotations
from tracknet.data.transforms import letterbox_image
from tracknet.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class PreprocessConfig:
    raw_root: Path
    output_root: Path
    target_width: int = 512
    target_height: int = 288
    overwrite: bool = False
    image_extension: str = ".png"
    write_frames: bool = True
    missing_annotation_policy: str = "invisible"  # invisible|skip
    background_sample_stride: int = 1

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "PreprocessConfig":
        return cls(
            raw_root=Path(cfg["raw_root"]),
            output_root=Path(cfg["output_root"]),
            target_width=int(cfg.get("target_width", 512)),
            target_height=int(cfg.get("target_height", 288)),
            overwrite=bool(cfg.get("overwrite", False)),
            image_extension=str(cfg.get("image_extension", ".png")),
            write_frames=bool(cfg.get("write_frames", True)),
            missing_annotation_policy=str(cfg.get("missing_annotation_policy", "invisible")),
            background_sample_stride=max(1, int(cfg.get("background_sample_stride", 1))),
        )


def _sequence_id(match_name: str, sequence_name: str) -> str:
    safe = f"{match_name}__{sequence_name}".replace("/", "_").replace("\\", "_")
    return safe


def _annotation_lookup(df: pd.DataFrame) -> dict[int, pd.Series]:
    return {int(row[RAW_FRAME_COL]): row for _, row in df.iterrows()}


def _safe_visibility(row: pd.Series | None) -> int:
    if row is None:
        return 0
    return 1 if int(row[RAW_VISIBILITY_COL]) == 1 else 0


def _safe_xy(row: pd.Series | None) -> tuple[float, float]:
    if row is None:
        return -1.0, -1.0
    x = row[RAW_X_COL]
    y = row[RAW_Y_COL]
    if pd.isna(x) or pd.isna(y):
        return -1.0, -1.0
    return float(x), float(y)


def process_raw_sequence(seq: RawSequence, cfg: PreprocessConfig, sequence_root: Path) -> dict[str, Any]:
    annotations = load_raw_annotations(seq.annotation_path)
    lookup = _annotation_lookup(annotations)
    frames_dir = ensure_dir(sequence_root / "frames")

    cap = cv2.VideoCapture(str(seq.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {seq.video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    rows: list[dict[str, Any]] = []
    median_samples: list[np.ndarray] = []
    first_transform = None
    frame_index = 0
    written = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        row = lookup.get(frame_index)
        if row is None and cfg.missing_annotation_policy == "skip":
            frame_index += 1
            continue
        image_rgb, transform = letterbox_image(frame_bgr, cfg.target_width, cfg.target_height)
        if first_transform is None:
            first_transform = transform
        if cfg.write_frames:
            frame_path = frames_dir / f"{frame_index:06d}{cfg.image_extension}"
            # cv2.imwrite expects BGR. Store PNG in RGB semantic by converting back
            # only for the encoder; all readers convert with cv2.COLOR_BGR2RGB.
            cv2.imwrite(str(frame_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if frame_index % cfg.background_sample_stride == 0:
            median_samples.append(image_rgb)
        visibility = _safe_visibility(row)
        x_raw, y_raw = _safe_xy(row)
        if visibility == 1 and x_raw >= 0 and y_raw >= 0:
            x_model, y_model = transform.raw_to_model(x_raw, y_raw)
            x_model, y_model = transform.clamp_model(x_model, y_model)
        else:
            x_model, y_model = -1.0, -1.0
        rows.append(
            {
                PROCESSED_FRAME_COL: frame_index,
                PROCESSED_VISIBILITY_COL: visibility,
                PROCESSED_X_RAW_COL: x_raw,
                PROCESSED_Y_RAW_COL: y_raw,
                PROCESSED_X_MODEL_COL: x_model,
                PROCESSED_Y_MODEL_COL: y_model,
                "frame_file": f"frames/{frame_index:06d}{cfg.image_extension}",
            }
        )
        written += 1
        frame_index += 1
    cap.release()
    if not rows:
        raise RuntimeError(f"No frames processed for sequence {seq.match_name}/{seq.sequence_name}")
    ann_df = pd.DataFrame(rows)
    ann_df.to_csv(sequence_root / "annotations.csv", index=False)

    if median_samples:
        seq_median = np.median(np.stack(median_samples, axis=0), axis=0).astype(np.uint8)
        cv2.imwrite(str(sequence_root / "sequence_median.png"), cv2.cvtColor(seq_median, cv2.COLOR_RGB2BGR))
    else:
        seq_median = None

    meta = {
        "sequence_id": sequence_root.name,
        "match_name": seq.match_name,
        "sequence_name": seq.sequence_name,
        "video_path": str(seq.video_path),
        "annotation_path": str(seq.annotation_path),
        "fps": fps,
        "raw_width": raw_width,
        "raw_height": raw_height,
        "reported_frame_count": frame_count_meta,
        "processed_frame_count": written,
        "target_width": cfg.target_width,
        "target_height": cfg.target_height,
        "transform": first_transform.as_dict() if first_transform else None,
    }
    write_json(sequence_root / "meta.json", meta)
    return meta


def _write_match_backgrounds(output_root: Path, sequence_metas: list[dict[str, Any]]) -> None:
    by_match: dict[str, list[Path]] = defaultdict(list)
    for meta in sequence_metas:
        median_path = output_root / "sequences" / meta["sequence_id"] / "sequence_median.png"
        if median_path.exists():
            by_match[meta["match_name"]].append(median_path)
    bg_dir = ensure_dir(output_root / "backgrounds")
    for match_name, paths in by_match.items():
        imgs = []
        for path in paths:
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is not None:
                imgs.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not imgs:
            continue
        bg = np.median(np.stack(imgs, axis=0), axis=0).astype(np.uint8)
        cv2.imwrite(str(bg_dir / f"{match_name}.png"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))


def preprocess_dataset(cfg: PreprocessConfig) -> dict[str, Any]:
    if cfg.target_width <= 0 or cfg.target_height <= 0:
        raise ValueError("target_width and target_height must be positive")
    if cfg.output_root.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Processed output already exists: {cfg.output_root}. Set overwrite=true to rebuild.")
        import shutil

        shutil.rmtree(cfg.output_root)
    ensure_dir(cfg.output_root / "sequences")
    sequences = discover_raw_sequences(cfg.raw_root)
    metas: list[dict[str, Any]] = []
    for seq in tqdm(sequences, desc="Preprocessing raw videos", unit="sequence"):
        sid = _sequence_id(seq.match_name, seq.sequence_name)
        sequence_root = ensure_dir(cfg.output_root / "sequences" / sid)
        metas.append(process_raw_sequence(seq, cfg, sequence_root))
    _write_match_backgrounds(cfg.output_root, metas)
    manifest = {
        "format_version": 1,
        "raw_root": str(cfg.raw_root),
        "target_width": cfg.target_width,
        "target_height": cfg.target_height,
        "image_extension": cfg.image_extension,
        "sequences": metas,
    }
    write_json(cfg.output_root / "manifest.json", manifest)
    return manifest
