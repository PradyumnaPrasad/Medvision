#!/usr/bin/env python3
"""Sweep medvision/kg/reasoning.py's activation_threshold to see its effect on
Pneumonia's Recall@1 vs. False-alarm@1, using the real held-out eval set.

This exists because scripts/run_eval.py found a 48.4% false-alarm rate for
D_PNEUMONIA at the current default threshold (0.3) — this script checks whether a
different threshold gets a better trade-off, using real data instead of guessing.
Classifier inference is run once and cached; only the (cheap) KG scoring step is
re-run per threshold value.

Usage: uv run python scripts/tune_activation_threshold.py
"""

from __future__ import annotations

import time

import numpy as np

from medvision.config import get_settings
from medvision.eval.data import load_pneumonia_split
from medvision.kg.reasoning import score_differentials
from medvision.pipeline import get_models

DISEASE_NODE = "D_PNEUMONIA"
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def main() -> None:
    settings = get_settings()
    eval_path = settings.paths.resolve("raw_dir") / "calibration" / "pneumonia_eval.parquet"
    if not eval_path.exists():
        raise SystemExit("Missing eval data. Run: bash scripts/download_calibration_set.sh")

    print(f"Loading eval split from {eval_path} ...")
    images, labels = load_pneumonia_split(eval_path)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"{len(images)} images ({n_pos} pneumonia, {n_neg} normal)")

    models = get_models()
    print("\nRunning classifier once over all images (cached for every threshold below)...")
    t0 = time.perf_counter()
    cached_finding_probs: list[dict[str, float]] = []
    for i, image in enumerate(images):
        preds = models.classifier.predict(image)
        cached_finding_probs.append(
            {p.node_id: models.calibration.apply(p.node_id, p.prob) for p in preds}
        )
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(images)} ({time.perf_counter() - t0:.1f}s)")

    is_positive = labels.astype(bool)
    print(
        f"\n{'Threshold':>9}  {'Recall@1':>9}  {'False-alarm@1':>14}  {'Recall - FalseAlarm':>20}"
    )
    best = None
    for threshold in THRESHOLDS:
        top1_hit_positive = []
        top1_hit_negative = []
        for finding_probs, positive in zip(cached_finding_probs, is_positive, strict=True):
            diffs = score_differentials(models.graph, finding_probs, activation_threshold=threshold)
            top1 = diffs[0].node_id if diffs else None
            if positive:
                top1_hit_positive.append(top1 == DISEASE_NODE)
            else:
                top1_hit_negative.append(top1 == DISEASE_NODE)

        recall_at_1 = float(np.mean(top1_hit_positive)) if top1_hit_positive else 0.0
        false_alarm_at_1 = float(np.mean(top1_hit_negative)) if top1_hit_negative else 0.0
        margin = recall_at_1 - false_alarm_at_1
        marker = ""
        if best is None or margin > best[1]:
            best = (threshold, margin)
            marker = "  <-- best margin so far"
        row = f"{threshold:>9.2f}  {recall_at_1:>9.1%}  {false_alarm_at_1:>14.1%}  {margin:>20.1%}"
        print(row + marker)

    print("\nCurrent default (configs/default.yaml kg.reasoning.activation_threshold): 0.3")
    if best:
        print(f"Best recall-minus-false-alarm margin found at threshold={best[0]:.2f}")


if __name__ == "__main__":
    main()
