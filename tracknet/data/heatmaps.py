"""Heatmap generation and coordinate extraction.

TrackNet papers use two target semantics:
- V1/V2/V4: probability-like Gaussian heatmaps.
- V3/V5: binary masks around the labeled coordinate.
The functions below keep the channel shape and coordinate convention explicit:
coordinates are always (x, y), arrays are always [H, W].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch

HeatmapMode = Literal["gaussian", "binary_disk", "v1_uint8"]


def gaussian_heatmap(width: int, height: int, x: float, y: float, sigma: float, peak: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    yy, xx = np.mgrid[0:height, 0:width]
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    return heat * float(peak)


def binary_disk_heatmap(width: int, height: int, x: float, y: float, radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    yy, xx = np.mgrid[0:height, 0:width]
    mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius * radius
    return mask.astype(np.float32)


def empty_heatmap(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.float32)


def make_heatmap(width: int, height: int, visibility: int, x: float, y: float, mode: HeatmapMode, sigma: float, radius: float) -> np.ndarray:
    if int(visibility) != 1 or not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0:
        return empty_heatmap(width, height)
    x = float(np.clip(x, 0, width - 1))
    y = float(np.clip(y, 0, height - 1))
    if mode == "gaussian":
        return gaussian_heatmap(width, height, x, y, sigma=sigma, peak=1.0)
    if mode == "v1_uint8":
        # TrackNetV1 trains a 256-way per-pixel softmax over grayscale heatmap
        # values. The paper scales the Gaussian target to [0, 255].
        return np.rint(gaussian_heatmap(width, height, x, y, sigma=sigma, peak=255.0)).astype(np.int64)
    if mode == "binary_disk":
        return binary_disk_heatmap(width, height, x, y, radius=radius)
    raise ValueError(f"Unsupported heatmap mode: {mode}")


def heatmap_argmax(heatmap: np.ndarray, threshold: float) -> tuple[int, int] | None:
    if heatmap.size == 0 or float(np.max(heatmap)) < threshold:
        return None
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return int(x), int(y)


def heatmap_largest_blob_centroid(heatmap: np.ndarray, threshold: float) -> tuple[int, int] | None:
    """TrackNetV2/V3 post-processing: threshold then centroid of largest blob."""
    if heatmap.size == 0 or float(np.max(heatmap)) < threshold:
        return None
    binary = (heatmap >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return heatmap_argmax(heatmap, threshold)
    return int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))


def v1_hough_coordinate(heatmap_uint8: np.ndarray, threshold: int = 128, min_radius: int = 2, max_radius: int = 12) -> tuple[int, int] | None:
    """TrackNetV1 post-processing from the paper.

    V1 first binarizes the grayscale heatmap at t=128 and then applies the
    Hough Gradient circle detector. The paper returns a coordinate only when
    exactly one circle is found; otherwise the frame is treated as no detection.
    """
    heat_u8 = np.asarray(np.clip(heatmap_uint8, 0, 255), dtype=np.uint8)
    _, binary = cv2.threshold(heat_u8, threshold, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(1, min_radius * 2),
        param1=50,
        param2=4,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    if len(circles) != 1:
        return None
    x, y, _ = circles[0]
    return int(x), int(y)


def coordinate_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return math.inf
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))
