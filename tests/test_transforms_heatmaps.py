from __future__ import annotations

import numpy as np

from tracknet.data.heatmaps import binary_disk_heatmap, gaussian_heatmap, heatmap_largest_blob_centroid
from tracknet.data.transforms import build_letterbox_transform


def test_letterbox_round_trip() -> None:
    t = build_letterbox_transform(1920, 1080, 512, 288)
    x, y = 1234.5, 456.25
    mx, my = t.raw_to_model(x, y)
    rx, ry = t.model_to_raw(mx, my)
    assert abs(rx - x) < 1e-4
    assert abs(ry - y) < 1e-4


def test_heatmap_peak_and_blob_centroid() -> None:
    heat = gaussian_heatmap(32, 32, 15, 12, sigma=2.0)
    yy, xx = np.unravel_index(int(np.argmax(heat)), heat.shape)
    assert (xx, yy) == (15, 12)
    disk = binary_disk_heatmap(32, 32, 10, 9, radius=3)
    pred = heatmap_largest_blob_centroid(disk, threshold=0.5)
    assert pred == (10, 9)
