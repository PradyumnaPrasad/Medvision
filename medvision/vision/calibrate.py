"""Temperature scaling for the classifier's per-label probabilities (PLAN.md T14).

torchxrayvision's forward pass already applies its own operating-point normalization
(op_norm), so we don't get raw pre-sigmoid logits back — only a probability-like
score per label. Standard temperature scaling divides a *logit* by a learned T before
the sigmoid; here we approximate that by treating the given probability as if it came
from a sigmoid, inverting it to a pseudo-logit, scaling by T, and re-applying sigmoid:

    calibrated_p = sigmoid(logit(raw_p) / T)

T > 1 softens (pulls toward 0.5, i.e. less confident); T < 1 sharpens. T is fit per
label by minimizing binary cross-entropy against real ground-truth labels — see
scripts/fit_calibration.py for how those labels are obtained.

Calibrating all 18 torchxrayvision labels needs per-label ground truth (e.g. the full
NIH ChestX-ray14 set, PLAN.md T31). Right now we only have a real, labeled proxy
dataset for pneumonia (a public Normal-vs-Pneumonia dataset), so only the pneumonia-
correlated labels get a fitted temperature; everything else defaults to T=1.0
(identity — unchanged raw probability) rather than a fabricated number. This module
does not know or care which labels were fit; that's a fact about the saved
CalibrationData, recorded honestly in its `notes` field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EPS = 1e-6


def _clip(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, _EPS, 1 - _EPS)


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = _clip(p)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(prob: float, temperature: float) -> float:
    """Rescale one probability by a fitted temperature. temperature=1.0 is a no-op."""
    if temperature == 1.0:
        return prob
    return float(sigmoid(logit(prob) / temperature))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bin predictions by confidence, weight each bin's |accuracy -
    confidence| gap by how many predictions fall in it, sum. Lower is better; a
    perfectly calibrated model scores 0."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    if n == 0:
        return 0.0

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_confidence = probs[in_bin].mean()
        avg_accuracy = labels[in_bin].mean()
        ece += (count / n) * abs(avg_accuracy - avg_confidence)
    return float(ece)


def _bce(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = _clip(probs)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def fit_temperature(raw_probs: np.ndarray, labels: np.ndarray) -> float:
    """Find the temperature T minimizing binary cross-entropy on (raw_probs, labels).

    Uses a bounded 1-D scalar search (scipy) rather than gradient descent — there's
    exactly one parameter and the loss is smooth and unimodal in practice, so this is
    simpler and just as reliable.
    """
    from scipy.optimize import minimize_scalar

    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pseudo_logits = logit(raw_probs)

    def loss(t: float) -> float:
        if t <= 0:
            return float("inf")
        calibrated = sigmoid(pseudo_logits / t)
        return _bce(calibrated, labels)

    result = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


@dataclass
class CalibrationData:
    temperatures: dict[str, float] = field(default_factory=dict)
    default_temperature: float = 1.0
    fitted_at: str | None = None
    dataset: str | None = None
    notes: str | None = None
    metrics: dict[str, dict] = field(default_factory=dict)

    def temperature_for(self, node_id: str) -> float:
        return self.temperatures.get(node_id, self.default_temperature)

    def apply(self, node_id: str, raw_prob: float) -> float:
        return apply_temperature(raw_prob, self.temperature_for(node_id))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(
                {
                    "temperatures": self.temperatures,
                    "default_temperature": self.default_temperature,
                    "fitted_at": self.fitted_at,
                    "dataset": self.dataset,
                    "notes": self.notes,
                    "metrics": self.metrics,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> CalibrationData:
        with path.open() as f:
            raw = json.load(f)
        return cls(**raw)

    @classmethod
    def identity(cls) -> CalibrationData:
        """No calibration data available/loaded: every label passes through unchanged."""
        return cls(notes="No calibration.json found; all probabilities are raw/uncalibrated.")


def load_calibration(path: Path) -> CalibrationData:
    """Load calibration.json if it exists, else return the identity (no-op) calibration."""
    if not path.exists():
        return CalibrationData.identity()
    return CalibrationData.load(path)
