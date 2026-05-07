"""Create a small visualization gallery from processed samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tracknet.config import load_yaml
from tracknet.papers import DEFAULT_PAPER_ID, get_paper_spec, paper_id_from_model_config
from tracknet.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize processed TrackNet windows")
    parser.add_argument("--config", type=Path, required=True, help="YAML config with visualize_dataset section")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    section = cfg.get("visualize_dataset")
    if not isinstance(section, dict):
        raise KeyError("Missing config section: visualize_dataset")
    model_cfg = section.get("model") or cfg.get("model") or {"version": DEFAULT_PAPER_ID}
    ds = get_paper_spec(paper_id_from_model_config(model_cfg)).build_heatmap_dataset(section["dataset"])
    out_dir = ensure_dir(Path(section.get("output_dir", "outputs/visualize_dataset")))
    max_samples = min(int(section.get("max_samples", 8)), len(ds))
    for i in range(max_samples):
        sample = ds[i]
        # Show the target frame and overlay the heatmap. Tensors are RGB [C,H,W].
        local_idx = int(sample["target_info"][0]["local_index"])
        frame = sample["frames"][local_idx].numpy().transpose(1, 2, 0)
        frame_u8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
        target = sample["target"].numpy()
        if target.ndim == 3:
            target = target[0]
        target_norm = target.astype(np.float32)
        if target_norm.max() > 0:
            target_norm = target_norm / target_norm.max()
        heat_color = cv2.applyColorMap(np.clip(target_norm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame_bgr, 0.75, heat_color, 0.25, 0.0)
        cv2.imwrite(str(out_dir / f"sample_{i:04d}.png"), overlay)
    print(f"Wrote {max_samples} visualization images to {out_dir}")


if __name__ == "__main__":
    main()
