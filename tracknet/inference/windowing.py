"""Window scheduling primitives for video and dataset inference."""

from __future__ import annotations


def window_indices_for_frame(frame_index: int, frame_count: int, sequence_length: int) -> list[int]:
    """Return a prefix-padded window ending at `frame_index`.

    This is used by single-output papers such as TrackNetV1. Prefix padding
    preserves early frames instead of dropping them before enough temporal
    context is available.
    """

    start = frame_index - sequence_length + 1
    return [min(max(i, 0), frame_count - 1) for i in range(start, frame_index + 1)]


def last_frame_windows(frame_count: int, sequence_length: int) -> list[list[int]]:
    if frame_count <= 0:
        return []
    return [window_indices_for_frame(i, frame_count, sequence_length) for i in range(frame_count)]


def sliding_windows(frame_count: int, sequence_length: int) -> list[list[int]]:
    if frame_count <= 0:
        return []
    if frame_count < sequence_length:
        return last_frame_windows(frame_count, sequence_length)
    return [list(range(start, start + sequence_length)) for start in range(0, frame_count - sequence_length + 1)]
