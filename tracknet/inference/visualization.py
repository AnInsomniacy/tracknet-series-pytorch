"""Visualization helpers for prediction outputs."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import cv2
import numpy as np

from tracknet.inference.postprocess import Prediction


def draw_prediction(
    frame_bgr: np.ndarray,
    prediction: Prediction,
    *,
    frame_index: int | None = None,
    trail: Iterable[tuple[int, int]] | None = None,
    radius: int = 6,
) -> np.ndarray:
    """Draw one prediction on a raw BGR frame without changing its dimensions."""
    out = frame_bgr.copy()
    if trail is not None:
        pts = list(trail)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, a, b, (0, 255, 255), 2, cv2.LINE_AA)
    if prediction.visibility == 1 and prediction.x >= 0 and prediction.y >= 0:
        center = (int(round(prediction.x)), int(round(prediction.y)))
        cv2.circle(out, center, int(radius), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(out, center, 2, (0, 255, 0), -1, cv2.LINE_AA)
    if frame_index is not None:
        cv2.putText(out, f"Frame {frame_index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def update_trail(trail: deque[tuple[int, int]], prediction: Prediction, max_length: int) -> deque[tuple[int, int]]:
    if prediction.visibility == 1 and prediction.x >= 0 and prediction.y >= 0:
        trail.append((int(round(prediction.x)), int(round(prediction.y))))
    while len(trail) > max_length:
        trail.popleft()
    return trail
