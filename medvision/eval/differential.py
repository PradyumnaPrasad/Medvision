"""Does the knowledge-graph differential-diagnosis ranking actually work? (PLAN.md T31)

This is the evaluation that matters most for MEDVISION specifically: the classifier
metrics in classification.py score the underlying vision model, which we didn't
train — but medvision/kg/reasoning.py's differential ranking is our own contribution,
and until now nothing had ever measured whether it puts the right disease near the
top. This module runs classifier -> calibration -> KG reasoning (skipping Grad-CAM,
localization and LLM narration, which are irrelevant to ranking and would only slow
down evaluation over many images) and checks the ranked list against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from medvision.kg.reasoning import DiagnosisScore, score_differentials
from medvision.pipeline import get_models


@dataclass
class DifferentialEvalResult:
    disease_node_id: str
    n_positive: int  # true cases of this disease
    n_negative: int  # true cases known NOT to be this disease (e.g. Normal)
    recall_at_1: float  # disease is the #1 differential, among positive cases
    recall_at_3: float  # disease is in the top 3, among positive cases
    recall_at_any: float  # disease appears anywhere in the differential list
    false_alarm_at_1: float  # disease is #1 differential among NEGATIVE cases (lower is better)


def predict_differentials(image: Image.Image) -> list[DiagnosisScore]:
    """classifier -> calibration -> KG reasoning only; see module docstring for why
    Grad-CAM/localization/LLM narration are skipped here."""
    models = get_models()
    predictions = models.classifier.predict(image)
    finding_probs = {p.node_id: models.calibration.apply(p.node_id, p.prob) for p in predictions}
    return score_differentials(models.graph, finding_probs)


def evaluate_differential_ranking(
    images_and_labels: list[tuple[Image.Image, bool]], disease_node_id: str
) -> DifferentialEvalResult:
    """images_and_labels: (image, is_true_case_of_disease_node_id) pairs."""
    positive_ranks: list[int | None] = []
    negative_top1_hits: list[bool] = []

    for image, is_positive in images_and_labels:
        diffs = predict_differentials(image)
        ranked_ids = [d.node_id for d in diffs]
        rank = ranked_ids.index(disease_node_id) + 1 if disease_node_id in ranked_ids else None

        if is_positive:
            positive_ranks.append(rank)
        else:
            negative_top1_hits.append(bool(ranked_ids and ranked_ids[0] == disease_node_id))

    n_positive = len(positive_ranks)
    n_negative = len(negative_top1_hits)

    def _fraction(predicate) -> float:
        return sum(1 for r in positive_ranks if predicate(r)) / n_positive if n_positive else 0.0

    return DifferentialEvalResult(
        disease_node_id=disease_node_id,
        n_positive=n_positive,
        n_negative=n_negative,
        recall_at_1=_fraction(lambda r: r == 1),
        recall_at_3=_fraction(lambda r: r is not None and r <= 3),
        recall_at_any=_fraction(lambda r: r is not None),
        false_alarm_at_1=(sum(negative_top1_hits) / n_negative if n_negative else 0.0),
    )
