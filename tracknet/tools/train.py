"""Train or resume a TrackNet model."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracknet.config import load_yaml
from tracknet.training.trainer import TrackNetTrainer, cleanup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TrackNet")
    parser.add_argument("--config", type=Path, required=True, help="Training YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    trainer = TrackNetTrainer(cfg)
    try:
        result = trainer.fit()
    finally:
        cleanup_distributed()
    print(f"Training finished. output_dir={result.output_dir} best_loss={result.best_loss:.6f} last_epoch={result.last_epoch}")


if __name__ == "__main__":
    main()
