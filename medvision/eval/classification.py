"""Per-label classification metrics (PLAN.md T31).

Deliberately independent of any specific dataset: callers hand this raw probability
arrays and binary ground-truth labels for one finding at a time (however they got
them — see scripts/run_eval.py for the real labeled data we actually have). This
module only knows how to score, not where the data comes from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from medvision.vision.calibrate import expected_calibration_error


@dataclass
class ClassificationMetrics:
    node_id: str
    n: int
    n_positive: int
    auroc: float | None  # None if the label has only one class present (AUROC undefined)
    precision: float
    recall: float
    f1: float
    ece: float
    threshold: float


def _binary_counts(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    predicted_positive = probs >= threshold
    actual_positive = labels >= 0.5
    tp = int(np.sum(predicted_positive & actual_positive))
    fp = int(np.sum(predicted_positive & ~actual_positive))
    fn = int(np.sum(~predicted_positive & actual_positive))
    tn = int(np.sum(~predicted_positive & ~actual_positive))
    return tp, fp, fn, tn


def evaluate_classification(
    node_id: str, probs: np.ndarray, labels: np.ndarray, threshold: float
) -> ClassificationMetrics:
    """Score one finding's predicted probabilities against binary ground truth.

    threshold is the decision cutoff used for precision/recall/F1 (AUROC and ECE
    don't need one — they evaluate the full probability, not a hard decision).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    n_positive = int(labels.sum())
    auroc = None
    if 0 < n_positive < len(labels):
        auroc = float(roc_auc_score(labels, probs))

    tp, fp, fn, _tn = _binary_counts(probs, labels, threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ClassificationMetrics(
        node_id=node_id,
        n=len(labels),
        n_positive=n_positive,
        auroc=auroc,
        precision=precision,
        recall=recall,
        f1=f1,
        ece=expected_calibration_error(probs, labels),
        threshold=threshold,
    )
