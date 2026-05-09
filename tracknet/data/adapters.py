"""Raw dataset adapters for canonical preprocessing.

Adapters are the only layer that understands external dataset layouts. They
emit neutral `RawSequence` records so preprocessing, training, and paper specs
can stay independent from raw directory conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

from tracknet.constants import SUPPORTED_VIDEO_EXTENSIONS
from tracknet.data.raw_reader import RawSequence, discover_raw_sequences

DatasetAdapterName = Literal["legacy", "tracknet_domain"]


def _visible_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"


def _iter_match_sequences(match_dir: Path, *, domain: str | None = None) -> Iterable[RawSequence]:
    video_dir = match_dir / "video"
    csv_dir = match_dir / "csv"
    if not video_dir.exists() or not csv_dir.exists():
        return
    match_name = match_dir.name if domain is None else f"{domain}__{match_dir.name}"
    for video_path in sorted(video_dir.iterdir()):
        if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        sequence_name = video_path.stem
        annotation_path = csv_dir / f"{sequence_name}_ball.csv"
        if annotation_path.exists():
            yield RawSequence(match_name, sequence_name, video_path, annotation_path)


def discover_tracknet_domain_sequences(raw_root: str | Path) -> list[RawSequence]:
    """Discover `Professional|Amateur|Test/match*/video,csv` TrackNet data."""
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset root not found: {root}")
    sequences: list[RawSequence] = []
    for domain_dir in sorted(p for p in root.iterdir() if _visible_dir(p)):
        for match_dir in sorted(p for p in domain_dir.iterdir() if _visible_dir(p) and p.name.startswith("match")):
            sequences.extend(_iter_match_sequences(match_dir, domain=domain_dir.name))
    if not sequences:
        raise ValueError(
            f"No TrackNet domain sequences found under {root}. Expected <domain>/match*/video/*.mp4 and <domain>/match*/csv/*_ball.csv"
        )
    return sequences


def discover_raw_sequences_with_adapter(raw_root: str | Path, adapter: str = "legacy") -> list[RawSequence]:
    if adapter == "legacy":
        return discover_raw_sequences(raw_root)
    if adapter == "tracknet_domain":
        return discover_tracknet_domain_sequences(raw_root)
    raise ValueError(f"Unknown raw dataset adapter: {adapter}")


def sequence_domain(seq: RawSequence) -> str:
    if "__" in seq.match_name:
        return seq.match_name.split("__", 1)[0]
    return "default"


def sequence_match_key(seq: RawSequence) -> str:
    return seq.match_name
