"""Shared constants and column names.

The project keeps data, training, evaluation and inference aligned by reusing
these names everywhere instead of hard-coding strings in scripts.
"""

from __future__ import annotations

RAW_FRAME_COL = "Frame"
RAW_VISIBILITY_COL = "Visibility"
RAW_X_COL = "X"
RAW_Y_COL = "Y"

PROCESSED_FRAME_COL = "frame"
PROCESSED_VISIBILITY_COL = "visibility"
PROCESSED_X_RAW_COL = "x_raw"
PROCESSED_Y_RAW_COL = "y_raw"
PROCESSED_X_MODEL_COL = "x_model"
PROCESSED_Y_MODEL_COL = "y_model"

CSV_OUTPUT_COLUMNS = [RAW_FRAME_COL, RAW_VISIBILITY_COL, RAW_X_COL, RAW_Y_COL]
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
