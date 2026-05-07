"""Reader for the legacy raw TrackNet data layout.

The old project is not reused, but it revealed the raw layout this project must
support:

raw_root/
  match1/
    video/<rally>.mp4
    csv/<rally>_ball.csv
  match2/...

CSV files are expected to contain TrackNet columns `Frame`, `Visibility`, `X`,
`Y`. The reader accepts a few common case variants but normalizes everything to
stable internal names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from tracknet.constants import RAW_FRAME_COL, RAW_VISIBILITY_COL, RAW_X_COL, RAW_Y_COL, SUPPORTED_VIDEO_EXTENSIONS


@dataclass(frozen=True)
class RawSequence:
    match_name: str
    sequence_name: str
    video_path: Path
    annotation_path: Path


def _visible_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"


def discover_raw_sequences(raw_root: str | Path) -> list[RawSequence]:
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset root not found: {root}")
    sequences: list[RawSequence] = []
    for match_dir in sorted(p for p in root.iterdir() if _visible_dir(p) and p.name.startswith("match")):
        video_dir = match_dir / "video"
        csv_dir = match_dir / "csv"
        if not video_dir.exists() or not csv_dir.exists():
            continue
        for video_path in sorted(video_dir.iterdir()):
            if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                continue
            sequence_name = video_path.stem
            annotation_path = csv_dir / f"{sequence_name}_ball.csv"
            if annotation_path.exists():
                sequences.append(RawSequence(match_dir.name, sequence_name, video_path, annotation_path))
    if not sequences:
        raise ValueError(
            f"No raw sequences found under {root}. Expected match*/video/*.mp4 and match*/csv/*_ball.csv"
        )
    return sequences


def normalize_annotation_columns(df: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    candidates = {
        RAW_FRAME_COL: ["Frame", "frame", "FrameIndex", "frame_index", "frame_id", "Frame Name", "FrameName"],
        RAW_VISIBILITY_COL: ["Visibility", "visibility", "Visibility Class", "VisibilityClass", "visible"],
        RAW_X_COL: ["X", "x", "cx", "CenterX"],
        RAW_Y_COL: ["Y", "y", "cy", "CenterY"],
    }
    rename: dict[str, str] = {}
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    for target, names in candidates.items():
        found = None
        for name in names:
            col = lower_to_col.get(name.lower())
            if col is not None:
                found = col
                break
        if found is None:
            raise ValueError(f"Annotation file {source} is missing required column '{target}'. Columns={list(df.columns)}")
        rename[found] = target
    out = df.rename(columns=rename)[[RAW_FRAME_COL, RAW_VISIBILITY_COL, RAW_X_COL, RAW_Y_COL]].copy()
    # Some historical label files use frame names like 0008.jpg. Convert them
    # to integer frame indices without assuming 1-based numbering.
    if not pd.api.types.is_numeric_dtype(out[RAW_FRAME_COL]):
        out[RAW_FRAME_COL] = out[RAW_FRAME_COL].astype(str).str.extract(r"(\d+)")[0]
    out[RAW_FRAME_COL] = pd.to_numeric(out[RAW_FRAME_COL], errors="coerce").astype("Int64")
    out[RAW_VISIBILITY_COL] = pd.to_numeric(out[RAW_VISIBILITY_COL], errors="coerce").fillna(0).astype(int)
    out[RAW_X_COL] = pd.to_numeric(out[RAW_X_COL], errors="coerce")
    out[RAW_Y_COL] = pd.to_numeric(out[RAW_Y_COL], errors="coerce")
    out = out.dropna(subset=[RAW_FRAME_COL]).copy()
    out[RAW_FRAME_COL] = out[RAW_FRAME_COL].astype(int)
    out = out.sort_values(RAW_FRAME_COL).drop_duplicates(RAW_FRAME_COL, keep="last")
    return out.reset_index(drop=True)


def load_raw_annotations(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")
    df = pd.read_csv(path)
    return normalize_annotation_columns(df, path)
