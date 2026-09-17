"""End-to-end MEDVISION inference pipeline: image in, explainable AnalysisResult out.

Current scope (demo milestone): classifier -> Grad-CAM -> zone localization -> KG
differential-diagnosis scoring -> a template-based report. This intentionally
does NOT yet include calibration (T14), retrieval (T19-20), or LLM narration with
a grounding verifier (T22-24) — those replace `_build_report` in a later slice
without changing this module's public `analyze()` contract.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel

from medvision.config import get_settings
from medvision.explain.gradcam import GradCamExplainer
from medvision.explain.localize import localize
from medvision.kg.build import build_graph
from medvision.kg.graph import KnowledgeGraph
from medvision.kg.reasoning import DiagnosisScore, score_differentials
from medvision.vision.classifier import ChestXrayClassifier

FindingStatus = Literal["present", "possible", "absent"]


class PathStepOut(BaseModel):
    finding_node_id: str
    finding_name: str
    finding_prob: float
    relation: str
    weight: float
    contribution: float


class DifferentialOut(BaseModel):
    node_id: str
    name: str
    score: float
    paths: list[PathStepOut]


class FindingOut(BaseModel):
    node_id: str
    label: str
    name: str
    prob: float
    status: FindingStatus
    region: str | None = None
    region_share: float | None = None
    heatmap_path: str | None = None  # filesystem-relative path under uploads_dir


class AnalysisResult(BaseModel):
    id: str
    created_at: float
    findings: list[FindingOut]
    differentials: list[DifferentialOut]
    report_text: str
    timings_ms: dict[str, float]
    disclaimer: str


@dataclass
class _Models:
    classifier: ChestXrayClassifier
    explainer: GradCamExplainer
    graph: KnowledgeGraph


@lru_cache(maxsize=1)
def get_models() -> _Models:
    """Load all heavy models once (classifier weights, Grad-CAM hooks, KG). Cache process-wide."""
    settings = get_settings()
    classifier = ChestXrayClassifier()
    explainer = GradCamExplainer(classifier)
    from medvision.config import get_labels_map

    graph = build_graph(settings.paths.resolve("kg_ontology_dir"), get_labels_map())
    return _Models(classifier=classifier, explainer=explainer, graph=graph)


def _classify_status(
    prob: float, pos_threshold: float, uncertain_threshold: float
) -> FindingStatus:
    if prob >= pos_threshold:
        return "present"
    if prob >= uncertain_threshold:
        return "possible"
    return "absent"


def _build_report(
    findings: list[FindingOut], differentials: list[DiagnosisScore], disclaimer: str
) -> str:
    """Template-based draft report. To be replaced by LLM narration + verifier (PLAN.md T22-T24)."""
    shown = [f for f in findings if f.status != "absent"]
    lines: list[str] = []

    lines.append("FINDINGS:")
    if not shown:
        lines.append("  No findings met the reporting threshold.")
    else:
        for f in shown:
            region = (
                f" most prominent in the {graph_node_name(f.region).lower()}" if f.region else ""
            )
            qualifier = "is present" if f.status == "present" else "is possibly present"
            lines.append(f"  - {f.name} {qualifier} (p={f.prob:.2f}){region}.")

    lines.append("")
    lines.append("IMPRESSION:")
    if not differentials:
        lines.append("  No differential diagnosis reached the activation threshold.")
    else:
        top = differentials[0]
        evidence = ", ".join(sorted({graph_finding_name(p.src) for p in top.paths}))
        lines.append(
            f"  Findings are most consistent with {top.name} (score={top.score:.2f}), "
            f"supported by: {evidence}."
        )
        if len(differentials) > 1:
            others = ", ".join(d.name for d in differentials[1:])
            lines.append(f"  Differential diagnoses to also consider: {others}.")

    lines.append("")
    lines.append(f"NOTE: {disclaimer}")
    return "\n".join(lines)


def graph_node_name(node_id: str) -> str:
    """Resolve any KG node id (finding, anatomy or disease) to its display name."""
    node = get_models().graph.get_node(node_id)
    return node.name if node else node_id


# kept as an alias for readability at call sites that specifically resolve findings
graph_finding_name = graph_node_name


def analyze(
    image: Image.Image,
    save_dir: Path | None = None,
    result_id: str | None = None,
) -> AnalysisResult:
    """Run the full pipeline on a single PIL image and return an explainable AnalysisResult.

    If save_dir is given, Grad-CAM overlays for each present/possible finding are written
    there as heatmap_<node_id>.png, and FindingOut.heatmap_path is set to that filename.
    Callers that need to know the id up front (e.g. to name save_dir after it) can pass
    result_id explicitly instead of letting analyze() generate one.
    """
    settings = get_settings()
    models = get_models()
    timings: dict[str, float] = {}
    result_id = result_id or str(uuid.uuid4())

    t0 = time.perf_counter()
    predictions = models.classifier.predict(image)
    timings["vision_ms"] = (time.perf_counter() - t0) * 1000

    pos_th = settings.vision.default_pos_threshold
    unc_th = settings.vision.default_uncertain_threshold

    t0 = time.perf_counter()
    findings: list[FindingOut] = []
    finding_probs: dict[str, float] = {}
    for p in predictions:
        status = _classify_status(p.prob, pos_th, unc_th)
        finding_probs[p.node_id] = p.prob
        node = models.graph.get_node(p.node_id)
        region = None
        region_share = None
        heatmap_path = None
        if status != "absent":
            heatmap = models.explainer.heatmap(image, p.node_id)
            zone = localize(heatmap)
            region = zone.zone_id
            region_share = zone.share
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                overlay = models.explainer.overlay(image, p.node_id)
                filename = f"heatmap_{p.node_id}.png"
                overlay.save(save_dir / filename)
                heatmap_path = filename
        findings.append(
            FindingOut(
                node_id=p.node_id,
                label=p.label,
                name=node.name if node else p.label,
                prob=p.prob,
                status=status,
                region=region,
                region_share=region_share,
                heatmap_path=heatmap_path,
            )
        )
    timings["explain_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    diagnosis_scores = score_differentials(models.graph, finding_probs)
    differentials = [
        DifferentialOut(
            node_id=d.node_id,
            name=d.name,
            score=d.score,
            paths=[
                PathStepOut(
                    finding_node_id=step.src,
                    finding_name=graph_finding_name(step.src),
                    finding_prob=finding_probs.get(step.src, 0.0),
                    relation=step.rel,
                    weight=step.weight,
                    contribution=step.contribution,
                )
                for step in d.paths
            ],
        )
        for d in diagnosis_scores
    ]
    timings["kg_reasoning_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    report_text = _build_report(findings, diagnosis_scores, settings.disclaimer)
    timings["report_ms"] = (time.perf_counter() - t0) * 1000

    findings.sort(key=lambda f: -f.prob)
    timings["total_ms"] = sum(timings.values())

    return AnalysisResult(
        id=result_id,
        created_at=time.time(),
        findings=findings,
        differentials=differentials,
        report_text=report_text,
        timings_ms=timings,
        disclaimer=settings.disclaimer,
    )
