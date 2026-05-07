"""Image resizing and coordinate transforms.

All stages use the same letterbox transform. This matters because a predicted
heatmap coordinate is only meaningful when the inverse transform maps it back to
exactly the same raw video coordinate system used by the labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale: float
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int

    def raw_to_model(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.pad_x, y * self.scale + self.pad_y

    def model_to_raw(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale

    def clamp_model(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, 0.0), float(self.target_width - 1)),
            min(max(y, 0.0), float(self.target_height - 1)),
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "scale": self.scale,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int | float]) -> "LetterboxTransform":
        return cls(
            original_width=int(data["original_width"]),
            original_height=int(data["original_height"]),
            target_width=int(data["target_width"]),
            target_height=int(data["target_height"]),
            scale=float(data["scale"]),
            pad_x=int(data["pad_x"]),
            pad_y=int(data["pad_y"]),
            resized_width=int(data["resized_width"]),
            resized_height=int(data["resized_height"]),
        )


def build_letterbox_transform(original_width: int, original_height: int, target_width: int, target_height: int) -> LetterboxTransform:
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid original size: {original_width}x{original_height}")
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    return LetterboxTransform(
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def letterbox_image(image_bgr: np.ndarray, target_width: int, target_height: int) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize BGR frame with padding and return an RGB image.

    OpenCV reads frames as BGR, but TrackNet training uses RGB tensors. The
    conversion is centralized here to prevent BGR/RGB drift between training and
    inference.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape [H,W,3], got {image_bgr.shape}")
    original_height, original_width = image_bgr.shape[:2]
    t = build_letterbox_transform(original_width, original_height, target_width, target_height)
    resized = cv2.resize(image_bgr, (t.resized_width, t.resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[t.pad_y : t.pad_y + t.resized_height, t.pad_x : t.pad_x + t.resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, t


def image_to_tensor_chw_uint8_rgb(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")
    return np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1))
