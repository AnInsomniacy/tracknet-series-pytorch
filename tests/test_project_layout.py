from __future__ import annotations

from pathlib import Path

from tracknet.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_keeps_source_data_package_trackable() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/dataset/" in gitignore
    assert "data/" not in gitignore
    assert "/data/" not in gitignore


def test_default_configs_use_dataset_raw_and_processed_roots() -> None:
    preprocess = load_yaml(PROJECT_ROOT / "configs" / "preprocess.yaml")
    preprocess_v1 = load_yaml(PROJECT_ROOT / "configs" / "preprocess_v1_640x360.yaml")

    assert preprocess["preprocess"]["raw_root"] == "dataset/raw"
    assert preprocess["preprocess"]["output_root"] == "dataset/processed/tracknet_dataset_512x288"
    assert preprocess_v1["preprocess"]["raw_root"] == "dataset/raw"
    assert preprocess_v1["preprocess"]["output_root"] == "dataset/processed/tracknet_dataset_640x360"

    for config_path in sorted((PROJECT_ROOT / "configs").glob("train*.yaml")):
        cfg = load_yaml(config_path)
        dataset = cfg["dataset"]
        assert str(dataset["processed_root"]).startswith("dataset/processed/")
        assert str(dataset["train_split_file"]).startswith("dataset/processed/")
        assert str(dataset["val_split_file"]).startswith("dataset/processed/")


def test_source_data_helpers_are_importable() -> None:
    from tracknet.data.adapters import discover_raw_sequences_with_adapter
    from tracknet.data.targets import HeatmapTargetPolicy

    assert callable(discover_raw_sequences_with_adapter)
    assert HeatmapTargetPolicy().target_frame_mode == "all"
