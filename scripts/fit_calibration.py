#!/usr/bin/env python3
"""Fit per-label temperature scaling for the classifier against real labeled data.

Only labels with a defensible ground-truth proxy in the available dataset are
calibrated — everything else is left at the identity temperature (1.0) rather than
faking a number (see medvision/vision/calibrate.py's module docstring). Right now
that means the pneumonia-correlated labels, calibrated against a real Normal-vs-
Pneumonia chest X-ray dataset (see scripts/download_calibration_set.sh).

Fits on `pneumonia_fit.parquet`, evaluates on the *separate* `pneumonia_eval.parquet`
so the reported before/after ECE numbers aren't just overfit to the fitting data.

Usage: uv run python scripts/fit_calibration.py
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np
from PIL import Image

from medvision.config import get_settings
from medvision.eval.data import load_pneumonia_split
from medvision.vision.calibrate import CalibrationData, expected_calibration_error, fit_temperature
from medvision.vision.classifier import ChestXrayClassifier

# Findings with a defensible ground-truth proxy in a binary Normal-vs-Pneumonia
# dataset: these are the radiographic correlates a Normal-vs-Pneumonia label can
# actually stand in for. All other torchxrayvision labels are left uncalibrated.
CALIBRATABLE_NODES = ["F_PNEUMONIA_OPACITY", "F_CONSOLIDATION", "F_INFILTRATION", "F_LUNG_OPACITY"]

DATASET_DESCRIPTION = (
    "keremberke/chest-xray-classification (HF parquet mirror of the Kermany/Mooney "
    "'Chest X-Ray Images (Pneumonia)' dataset): full/test split (N={n_fit}) used to fit "
    "temperature, full/validation split (N={n_eval}) held out for evaluation. Binary "
    "Normal(0)/Pneumonia(1) label used as ground truth for pneumonia-correlated findings only."
)


def _predict_all(
    classifier: ChestXrayClassifier, images: list[Image.Image]
) -> dict[str, np.ndarray]:
    """Returns {node_id: array of raw probs, one per image} for the calibratable nodes."""
    per_node: dict[str, list[float]] = {node_id: [] for node_id in CALIBRATABLE_NODES}
    t0 = time.perf_counter()
    for i, image in enumerate(images):
        preds = {p.node_id: p.prob for p in classifier.predict(image)}
        for node_id in CALIBRATABLE_NODES:
            per_node[node_id].append(preds.get(node_id, 0.0))
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(images)} images ({elapsed:.1f}s elapsed)")
    return {node_id: np.array(vals) for node_id, vals in per_node.items()}


def main() -> None:
    settings = get_settings()
    fit_path = settings.paths.resolve("raw_dir") / "calibration" / "pneumonia_fit.parquet"
    eval_path = settings.paths.resolve("raw_dir") / "calibration" / "pneumonia_eval.parquet"
    out_path = settings.paths.resolve("models_dir") / "calibration.json"

    if not fit_path.exists() or not eval_path.exists():
        raise SystemExit(
            f"Missing calibration data. Run: bash scripts/download_calibration_set.sh\n"
            f"Expected: {fit_path} and {eval_path}"
        )

    print(f"Loading fit split from {fit_path} ...")
    fit_images, fit_labels = load_pneumonia_split(fit_path)
    print(f"Loading eval split from {eval_path} ...")
    eval_images, eval_labels = load_pneumonia_split(eval_path)
    print(
        f"Fit: {len(fit_images)} images ({int(fit_labels.sum())} pneumonia). "
        f"Eval: {len(eval_images)} images ({int(eval_labels.sum())} pneumonia)."
    )

    classifier = ChestXrayClassifier()

    print("\nRunning classifier over the fit split...")
    fit_probs = _predict_all(classifier, fit_images)
    print("Running classifier over the eval split...")
    eval_probs = _predict_all(classifier, eval_images)

    # A fitted temperature is only trusted if it (a) doesn't sit at the search bound —
    # which we found happens when the raw probability has no usable absolute-value
    # signal and the fit degenerates toward "just predict the base rate" — and (b)
    # actually improves held-out ECE by a non-trivial margin, not just on the data it
    # was fit on. Anything that fails either check reverts to the identity (1.0)
    # rather than shipping a "calibration" that doesn't generalize.
    max_temperature = 20.0
    min_relative_improvement = 0.10

    temperatures: dict[str, float] = {}
    metrics: dict[str, dict] = {}

    print(f"\n{'Label':<22} {'T':>6}  {'ECE before':>11}  {'ECE after':>10}  Accepted?")
    for node_id in CALIBRATABLE_NODES:
        t = fit_temperature(fit_probs[node_id], fit_labels)

        raw_eval = eval_probs[node_id]
        calibrated_eval = np.array([_apply(p, t) for p in raw_eval])
        ece_before = expected_calibration_error(raw_eval, eval_labels)
        ece_after = expected_calibration_error(calibrated_eval, eval_labels)

        degenerate = t >= 0.95 * max_temperature
        relative_improvement = 1 - (ece_after / ece_before) if ece_before > 0 else 0.0
        accepted = (not degenerate) and relative_improvement >= min_relative_improvement

        if accepted:
            temperatures[node_id] = t
            reason = "kept fitted temperature"
        else:
            reason = (
                "fit diverged toward the search bound (degenerate — raw probability has "
                "no usable absolute-value signal here)"
                if degenerate
                else f"held-out ECE improvement ({relative_improvement:.0%}) below the "
                f"{min_relative_improvement:.0%} acceptance threshold"
            )
            # temperatures[node_id] intentionally left unset -> CalibrationData.apply()
            # falls back to default_temperature=1.0 (identity) for this node

        metrics[node_id] = {
            "fitted_temperature": t,
            "accepted": accepted,
            "reason": reason,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "n_fit": len(fit_labels),
            "n_eval": len(eval_labels),
        }
        print(f"{node_id:<22} {t:>6.2f}  {ece_before:>11.4f}  {ece_after:>10.4f}  {accepted}")
        if not accepted:
            print(f"    -> reverted to identity (1.0): {reason}")

    accepted_nodes = sorted(temperatures.keys())
    rejected_nodes = sorted(n for n in CALIBRATABLE_NODES if n not in temperatures)
    calibration = CalibrationData(
        temperatures=temperatures,
        default_temperature=1.0,
        fitted_at=datetime.now(UTC).isoformat(),
        dataset=DATASET_DESCRIPTION.format(n_fit=len(fit_labels), n_eval=len(eval_labels)),
        notes=(
            f"Candidates considered (using Normal-vs-Pneumonia labels as a ground-truth "
            f"proxy for these pneumonia-correlated findings): {', '.join(CALIBRATABLE_NODES)}. "
            f"Accepted (fitted temperature genuinely improved held-out ECE, see metrics): "
            f"{', '.join(accepted_nodes) or 'none'}. "
            f"Rejected (reverted to identity — see metrics[node].reason): "
            f"{', '.join(rejected_nodes) or 'none'}. "
            "All other labels were never candidates and also default to 1.0 — calibrating "
            "them needs per-label ground truth (e.g. full NIH ChestX-ray14), future work "
            "(PLAN.md T31)."
        ),
        metrics=metrics,
    )
    calibration.save(out_path)
    print(f"\nSaved {out_path}")


def _apply(prob: float, temperature: float) -> float:
    from medvision.vision.calibrate import apply_temperature

    return apply_temperature(prob, temperature)


if __name__ == "__main__":
    main()
