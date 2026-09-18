#!/usr/bin/env python3
"""End-to-end evaluation against the real labeled chest X-ray set (PLAN.md T31).

Two things get measured, both on `pneumonia_eval.parquet` — the split held out from
calibration fitting, so nothing here is graded on data it has already seen:

1. Per-label classification quality (AUROC, F1, ECE) for the classifier outputs we
   have real ground truth for, comparing raw vs. calibrated probabilities so T14's
   effect on real decisions (not just ECE) is visible.
2. Differential-diagnosis ranking quality: when a case truly has pneumonia, does
   medvision/kg/reasoning.py actually rank Pneumonia highly? When it doesn't, does
   the system avoid falsely putting Pneumonia at #1? This is the evaluation that
   speaks to MEDVISION's own contribution (the KG reasoning), not just the
   pretrained classifier.

Usage: uv run python scripts/run_eval.py
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import numpy as np

from medvision.config import get_settings
from medvision.eval.classification import ClassificationMetrics, evaluate_classification
from medvision.eval.data import load_pneumonia_split
from medvision.eval.differential import DifferentialEvalResult, evaluate_differential_ranking
from medvision.pipeline import get_models

CANDIDATE_NODES = ["F_PNEUMONIA_OPACITY", "F_CONSOLIDATION", "F_INFILTRATION", "F_LUNG_OPACITY"]
DISEASE_NODE = "D_PNEUMONIA"

DATASET_NOTE = (
    "keremberke/chest-xray-classification (HF parquet mirror of the Kermany/Mooney "
    "'Chest X-Ray Images (Pneumonia)' dataset), full/validation split, N={n} images "
    "({n_pos} pneumonia, {n_neg} normal). This is the split HELD OUT from calibration "
    "fitting (scripts/fit_calibration.py fits on full/test) — nothing reported here was "
    "seen during calibration."
)


def _classification_report(images, labels, threshold: float) -> dict[str, dict]:
    models = get_models()
    raw_by_node: dict[str, list[float]] = {n: [] for n in CANDIDATE_NODES}
    calibrated_by_node: dict[str, list[float]] = {n: [] for n in CANDIDATE_NODES}

    t0 = time.perf_counter()
    for i, image in enumerate(images):
        preds = {p.node_id: p.prob for p in models.classifier.predict(image)}
        for node_id in CANDIDATE_NODES:
            raw = preds.get(node_id, 0.0)
            raw_by_node[node_id].append(raw)
            calibrated_by_node[node_id].append(models.calibration.apply(node_id, raw))
        if (i + 1) % 200 == 0:
            print(f"  classification pass: {i + 1}/{len(images)} ({time.perf_counter() - t0:.1f}s)")

    report: dict[str, dict] = {}
    for node_id in CANDIDATE_NODES:
        raw_metrics = evaluate_classification(
            node_id, np.array(raw_by_node[node_id]), labels, threshold
        )
        cal_metrics = evaluate_classification(
            node_id, np.array(calibrated_by_node[node_id]), labels, threshold
        )
        report[node_id] = {
            "raw": _metrics_dict(raw_metrics),
            "calibrated": _metrics_dict(cal_metrics),
        }
    return report


def _metrics_dict(m: ClassificationMetrics) -> dict:
    return {
        "n": m.n,
        "n_positive": m.n_positive,
        "auroc": m.auroc,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "ece": m.ece,
        "threshold": m.threshold,
    }


def _differential_dict(r: DifferentialEvalResult) -> dict:
    return {
        "disease_node_id": r.disease_node_id,
        "n_positive": r.n_positive,
        "n_negative": r.n_negative,
        "recall_at_1": r.recall_at_1,
        "recall_at_3": r.recall_at_3,
        "recall_at_any": r.recall_at_any,
        "false_alarm_at_1": r.false_alarm_at_1,
    }


def _write_markdown(path, classification: dict, differential: dict, dataset_note: str) -> None:
    lines = [
        "# MEDVISION evaluation report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        f"Dataset: {dataset_note}",
        "",
        "## Classification metrics (raw vs. calibrated)",
        "",
        "| Label | AUROC | F1 (raw) | F1 (calibrated) | ECE (raw) | ECE (calibrated) |",
        "|---|---|---|---|---|---|",
    ]
    for node_id, m in classification.items():
        auroc = m["raw"]["auroc"]
        lines.append(
            f"| {node_id} | {auroc:.3f} | {m['raw']['f1']:.3f} | {m['calibrated']['f1']:.3f} "
            f"| {m['raw']['ece']:.3f} | {m['calibrated']['ece']:.3f} |"
        )

    d = differential
    lines += [
        "",
        "## Differential-diagnosis ranking (knowledge-graph reasoning)",
        "",
        f"Disease evaluated: **{d['disease_node_id']}** "
        f"({d['n_positive']} true positive cases, {d['n_negative']} true negative cases)",
        "",
        f"- Recall@1 (ranked #1 when truly present): **{d['recall_at_1']:.1%}**",
        f"- Recall@3 (ranked in top 3 when truly present): **{d['recall_at_3']:.1%}**",
        f"- Recall@any (appears anywhere in the ranking when truly present): "
        f"**{d['recall_at_any']:.1%}**",
        f"- False-alarm@1 (wrongly ranked #1 on a true negative case): "
        f"**{d['false_alarm_at_1']:.1%}**",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    settings = get_settings()
    eval_path = settings.paths.resolve("raw_dir") / "calibration" / "pneumonia_eval.parquet"
    if not eval_path.exists():
        raise SystemExit(
            "Missing eval data. Run: bash scripts/download_calibration_set.sh\n"
            f"Expected: {eval_path}"
        )

    reports_dir = settings.paths.resolve("data_dir").parent / "reports"

    print(f"Loading eval split from {eval_path} ...")
    images, labels = load_pneumonia_split(eval_path)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"{len(images)} images ({n_pos} pneumonia, {n_neg} normal)")

    threshold = settings.vision.default_pos_threshold
    print(f"\nRunning classification evaluation (threshold={threshold}) ...")
    classification = _classification_report(images, labels, threshold)

    print("\nRunning differential-ranking evaluation ...")
    images_and_labels = list(zip(images, labels.astype(bool), strict=True))
    differential = _differential_dict(
        evaluate_differential_ranking(images_and_labels, DISEASE_NODE)
    )

    dataset_note = DATASET_NOTE.format(n=len(images), n_pos=n_pos, n_neg=n_neg)

    print("\n=== Classification metrics (raw -> calibrated: AUROC / F1 / ECE) ===")
    for node_id, m in classification.items():
        print(
            f"  {node_id:<22} AUROC={m['raw']['auroc']:.3f}  "
            f"F1 {m['raw']['f1']:.3f}->{m['calibrated']['f1']:.3f}  "
            f"ECE {m['raw']['ece']:.3f}->{m['calibrated']['ece']:.3f}"
        )
    print("\n=== Differential ranking (D_PNEUMONIA) ===")
    print(
        f"  Recall@1={differential['recall_at_1']:.1%}  "
        f"Recall@3={differential['recall_at_3']:.1%}  "
        f"Recall@any={differential['recall_at_any']:.1%}  "
        f"False-alarm@1={differential['false_alarm_at_1']:.1%}"
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "eval_report.json"
    md_path = reports_dir / "eval_report.md"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset": dataset_note,
                "classification": classification,
                "differential": differential,
            },
            indent=2,
        )
    )
    _write_markdown(md_path, classification, differential, dataset_note)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
