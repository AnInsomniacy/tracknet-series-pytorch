"""Preprocess legacy raw TrackNet data into the stable project format."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracknet.config import load_yaml
from tracknet.data.preprocessing import PreprocessConfig, preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess legacy raw TrackNet data")
    parser.add_argument("--config", type=Path, required=True, help="YAML config with a preprocess section")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    section = cfg.get("preprocess")
    if not isinstance(section, dict):
        raise KeyError("Missing config section: preprocess")
    pcfg = PreprocessConfig.from_mapping(section)
    manifest = preprocess_dataset(pcfg)
    print(f"Processed {len(manifest['sequences'])} sequences into {pcfg.output_root}")


if __name__ == "__main__":
    main()
