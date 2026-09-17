"""Deterministic differential-diagnosis scoring over the knowledge graph (PLAN.md 4.6/§1 step 4).

This is the source of truth for "what does the model think might be wrong" — not the LLM.
The LLM (medvision/reasoning/) only narrates the paths this module returns, and must cite
them. Keeping this deterministic and inspectable is what makes MEDVISION's explanations
auditable: every differential comes with the exact finding->disease edges that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from medvision.config import get_settings
from medvision.kg.graph import KGEdge, KnowledgeGraph


@dataclass(frozen=True)
class PathStep:
    """One edge on the way from an activated finding to a scored diagnosis."""

    src: str
    rel: str
    dst: str
    weight: float
    contribution: float  # probability(src) * weight, signed (negative for `contradicts`)


@dataclass(frozen=True)
class DiagnosisScore:
    node_id: str
    name: str
    score: float  # in [0, 1] after normalization across all candidates
    raw_score: float  # pre-normalization signed sum of contributions
    paths: list[PathStep]


def _activated_findings(
    finding_probs: dict[str, float], activation_threshold: float
) -> dict[str, float]:
    return {node_id: p for node_id, p in finding_probs.items() if p >= activation_threshold}


def score_differentials(
    graph: KnowledgeGraph,
    finding_probs: dict[str, float],
    top_k: int | None = None,
    activation_threshold: float | None = None,
    contradiction_penalty_weight: float | None = None,
) -> list[DiagnosisScore]:
    """Score candidate diseases from a set of (calibrated) finding probabilities.

    score(dx) = sum_f p(f) * w(f -> dx)               over `suggestive_of` edges
              - penalty * sum_f p(f) * w(f -> dx)      over `contradicts` edges
    normalized to [0, 1] by dividing by the top raw score (so the best candidate is 1.0,
    unless every raw score is <= 0, in which case all normalized scores are 0).

    Returns the top_k DiagnosisScore objects, sorted by score descending, each carrying
    the exact finding->disease edges (PathStep) that contributed to it, for citation by
    the reasoning/narration layer.
    """
    settings = get_settings()
    top_k = top_k or settings.kg.reasoning.top_k_differentials
    activation_threshold = (
        activation_threshold
        if activation_threshold is not None
        else settings.kg.reasoning.activation_threshold
    )
    penalty = (
        contradiction_penalty_weight
        if contradiction_penalty_weight is not None
        else settings.kg.reasoning.contradiction_penalty_weight
    )

    activated = _activated_findings(finding_probs, activation_threshold)

    raw_scores: dict[str, float] = {}
    paths_by_disease: dict[str, list[PathStep]] = {}

    for finding_id, prob in activated.items():
        edges: list[KGEdge] = graph.out_edges(finding_id, rel="suggestive_of") + graph.out_edges(
            finding_id, rel="contradicts"
        )
        for edge in edges:
            disease = graph.get_node(edge.dst)
            if disease is None or disease.type != "Disease":
                continue
            sign = -penalty if edge.rel == "contradicts" else 1.0
            contribution = sign * prob * edge.weight
            raw_scores[edge.dst] = raw_scores.get(edge.dst, 0.0) + contribution
            paths_by_disease.setdefault(edge.dst, []).append(
                PathStep(
                    src=finding_id,
                    rel=edge.rel,
                    dst=edge.dst,
                    weight=edge.weight,
                    contribution=contribution,
                )
            )

    if not raw_scores:
        return []

    max_raw = max(raw_scores.values())
    results: list[DiagnosisScore] = []
    for disease_id, raw in raw_scores.items():
        node = graph.get_node(disease_id)
        normalized = max(0.0, raw) / max_raw if max_raw > 0 else 0.0
        paths = sorted(paths_by_disease[disease_id], key=lambda p: -abs(p.contribution))
        results.append(
            DiagnosisScore(
                node_id=disease_id,
                name=node.name if node else disease_id,
                score=normalized,
                raw_score=raw,
                paths=paths,
            )
        )

    results.sort(key=lambda d: -d.score)
    return results[:top_k]
