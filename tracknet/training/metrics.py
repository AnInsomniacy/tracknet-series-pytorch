"""TrackNet evaluation classification and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tracknet.data.heatmaps import coordinate_distance

PredictionClass = Literal["tp", "tn", "fp1", "fp2", "fn"]


@dataclass
class ConfusionCounts:
    tp: int = 0
    tn: int = 0
    fp1: int = 0
    fp2: int = 0
    fn: int = 0

    def add(self, cls: PredictionClass) -> None:
        setattr(self, cls, getattr(self, cls) + 1)

    @property
    def fp(self) -> int:
        return self.fp1 + self.fp2

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def metrics(self) -> dict[str, float]:
        total = self.total
        precision_den = self.tp + self.fp
        recall_den = self.tp + self.fn
        accuracy = (self.tp + self.tn) / total if total else 0.0
        precision = self.tp / precision_den if precision_den else 0.0
        recall = self.tp / recall_den if recall_den else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": float(self.tp),
            "tn": float(self.tn),
            "fp1": float(self.fp1),
            "fp2": float(self.fp2),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "total": float(total),
        }


def classify_prediction(pred: tuple[float, float] | None, gt: tuple[float, float] | None, tolerance: float) -> PredictionClass:
    has_pred = pred is not None
    has_gt = gt is not None
    if not has_pred and not has_gt:
        return "tn"
    if not has_pred and has_gt:
        return "fn"
    if has_pred and not has_gt:
        return "fp2"
    return "tp" if coordinate_distance(pred, gt) <= tolerance else "fp1"
