from __future__ import annotations

from pathlib import Path

from tracknet.data.dataset import ProcessedTrackNetDataset, TrackNetDatasetConfig
from tracknet.data.raw_reader import discover_raw_sequences, load_raw_annotations
from tracknet.papers.base import HeatmapTargetPolicy


def test_legacy_raw_reader_and_preprocess(synthetic_raw_root: Path, synthetic_processed_root: Path) -> None:
    sequences = discover_raw_sequences(synthetic_raw_root)
    assert len(sequences) == 1
    ann = load_raw_annotations(sequences[0].annotation_path)
    assert {"Frame", "Visibility", "X", "Y"}.issubset(ann.columns)
    assert (synthetic_processed_root / "manifest.json").exists()
    assert (synthetic_processed_root / "sequences" / "match1__rally1" / "annotations.csv").exists()
    assert (synthetic_processed_root / "backgrounds" / "match1.png").exists()


def test_processed_dataset_window_shapes(synthetic_processed_root: Path) -> None:
    ds = ProcessedTrackNetDataset(
        TrackNetDatasetConfig(
            processed_root=synthetic_processed_root,
            sequence_length=3,
        ),
        target_policy=HeatmapTargetPolicy(target_frame_mode="all", heatmap_mode="gaussian", sigma=2.0),
    )
    sample = ds[0]
    assert sample["input"].shape == (9, 32, 32)
    assert sample["frames"].shape == (3, 3, 32, 32)
    assert sample["target"].shape == (3, 32, 32)
    assert len(sample["frame_indices"]) == 3


def test_v3_background_input_and_binary_targets(synthetic_processed_root: Path) -> None:
    ds = ProcessedTrackNetDataset(
        TrackNetDatasetConfig(
            processed_root=synthetic_processed_root,
            sequence_length=4,
        ),
        target_policy=HeatmapTargetPolicy(
            target_frame_mode="all",
            heatmap_mode="binary_disk",
            radius=2,
            include_background=True,
        ),
    )
    sample = ds[0]
    assert sample["input"].shape == (15, 32, 32)
    assert sample["target"].shape == (4, 32, 32)
