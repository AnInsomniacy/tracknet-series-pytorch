from __future__ import annotations

from pathlib import Path

from tracknet.data.preprocessing import PreprocessConfig, preprocess_dataset
from tracknet.data.dataset import ProcessedTrackNetDataset, TrackNetDatasetConfig
from tracknet.data.raw_reader import discover_raw_sequences, load_raw_annotations
from tracknet.data.targets import HeatmapTargetPolicy


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


def test_tracknet_domain_dataset_preprocess_writes_unique_manifest_splits_and_backgrounds(tmp_path: Path, synthetic_tracknet_domain_raw_root: Path) -> None:
    processed = tmp_path / "processed"

    manifest = preprocess_dataset(
        PreprocessConfig(
            raw_root=synthetic_tracknet_domain_raw_root,
            output_root=processed,
            target_width=32,
            target_height=32,
            overwrite=True,
            adapter="tracknet_domain",
            val_fraction=0.5,
        )
    )

    sequence_ids = [item["sequence_id"] for item in manifest["sequences"]]
    assert sequence_ids == [
        "Amateur__match1__rally3",
        "Professional__match1__rally1",
        "Professional__match2__rally2",
        "Test__match1__rally4",
    ]
    assert {item["domain"] for item in manifest["sequences"]} == {"Professional", "Amateur", "Test"}
    assert (processed / "backgrounds" / "Professional__match1.png").exists()
    assert (processed / "backgrounds" / "Amateur__match1.png").exists()
    assert (processed / "splits" / "train.txt").read_text().strip()
    assert (processed / "splits" / "val.txt").read_text().strip()
    assert (processed / "splits" / "test.txt").read_text().strip() == "Test__match1__rally4"


def test_tracknet_domain_preprocess_splits_are_sorted_and_deterministic(tmp_path: Path, synthetic_tracknet_domain_raw_root: Path) -> None:
    processed = tmp_path / "processed"

    manifest = preprocess_dataset(
        PreprocessConfig(
            raw_root=synthetic_tracknet_domain_raw_root,
            output_root=processed,
            target_width=32,
            target_height=32,
            overwrite=True,
            adapter="tracknet_domain",
            val_fraction=0.5,
        )
    )

    assert manifest["splits"]["train"] == ["Professional__match2__rally2"]
    assert manifest["splits"]["val"] == ["Amateur__match1__rally3", "Professional__match1__rally1"]
    assert manifest["splits"]["test"] == ["Test__match1__rally4"]


def test_processed_dataset_uses_sequence_background_key_and_split_file(tmp_path: Path, synthetic_tracknet_domain_raw_root: Path) -> None:
    processed = tmp_path / "processed"
    preprocess_dataset(
        PreprocessConfig(
            raw_root=synthetic_tracknet_domain_raw_root,
            output_root=processed,
            target_width=32,
            target_height=32,
            overwrite=True,
            adapter="tracknet_domain",
            val_fraction=0.5,
        )
    )

    ds = ProcessedTrackNetDataset(
        TrackNetDatasetConfig(
            processed_root=processed,
            sequence_length=3,
            split_file=processed / "splits" / "test.txt",
        ),
        target_policy=HeatmapTargetPolicy(
            target_frame_mode="all",
            heatmap_mode="binary_disk",
            radius=2,
            include_background=True,
        ),
    )

    assert {seq.sequence_id for seq in ds.sequences} == {"Test__match1__rally4"}
    assert ds.sequences[0].background_path == processed / "backgrounds" / "Test__match1.png"
    assert ds[0]["input"].shape == (12, 32, 32)
