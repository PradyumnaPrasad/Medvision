"""End-to-end MEDVISION inference pipeline: image in, explainable AnalysisResult out.

Pipeline: classifier -> temperature-scaled calibration -> Grad-CAM -> zone
localization -> KG differential-diagnosis scoring -> LLM narration with a citation
verifier (falling back to a deterministic template report if the LLM is unavailable
or fails verification). This intentionally does NOT yet include retrieval (T19-20)
— that slots in without changing this module's public `analyze()` contract.
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
from medvision.kg.reasoning import score_differentials
from medvision.reasoning.llm import OllamaClient
from medvision.reasoning.narrate import narrate
from medvision.reasoning.schema import EvidenceBlock, GroundingResult, LLMReport
from medvision.reasoning.template import flatten_report
from medvision.vision.calibrate import CalibrationData, load_calibration
from medvision.vision.classifier import ChestXrayClassifier

ReportSource = Literal["llm", "template"]

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
    prob: float  # calibrated probability (identity/unchanged for labels with no fitted temperature)
    raw_prob: float  # the classifier's original, uncalibrated output — kept for transparency
    status: FindingStatus
    region: str | None = None
    region_share: float | None = None
    heatmap_path: str | None = None  # filesystem-relative path under uploads_dir


class AnalysisResult(BaseModel):
    id: str
    created_at: float
    findings: list[FindingOut]
    differentials: list[DifferentialOut]
    report: LLMReport
    report_source: ReportSource
    grounding: GroundingResult
    evidence: EvidenceBlock  # id-tagged evidence (F1, D1, K1, ...) that report cites
    report_text: str  # flattened plain-text rendering of `report`, for PDF/back-compat
    timings_ms: dict[str, float]
    disclaimer: str


@dataclass
class _Models:
    classifier: ChestXrayClassifier
    explainer: GradCamExplainer
    graph: KnowledgeGraph
    llm_client: OllamaClient
    calibration: CalibrationData


@lru_cache(maxsize=1)
def get_models() -> _Models:
    """Load all heavy models once (classifier weights, Grad-CAM hooks, KG). Cache process-wide."""
    settings = get_settings()
    classifier = ChestXrayClassifier()
    explainer = GradCamExplainer(classifier)
    from medvision.config import get_labels_map

    graph = build_graph(settings.paths.resolve("kg_ontology_dir"), get_labels_map())
    llm_client = OllamaClient()
    calibration = load_calibration(settings.paths.resolve("models_dir") / "calibration.json")
    return _Models(
        classifier=classifier,
        explainer=explainer,
        graph=graph,
        llm_client=llm_client,
        calibration=calibration,
    )


def _classify_status(
    prob: float, pos_threshold: float, uncertain_threshold: float
) -> FindingStatus:
    if prob >= pos_threshold:
        return "present"
    if prob >= uncertain_threshold:
        return "possible"
    return "absent"


@dataclass(frozen=True)
class _EvidenceFindingView:
    """Adapts a FindingOut for medvision.reasoning (region resolved to a clinical name,
    e.g. 'right lower zone' instead of the raw KG id 'A_RIGHT_LOWER_ZONE') without
    changing FindingOut itself, which the API/UI still key by the raw region id."""

    node_id: str
    name: str
    prob: float
    status: str
    region: str | None


def _evidence_view(finding: FindingOut, graph: KnowledgeGraph) -> _EvidenceFindingView:
    region_name = None
    if finding.region:
        node = graph.get_node(finding.region)
        region_name = node.name.lower() if node else finding.region.lower()
    return _EvidenceFindingView(
        node_id=finding.node_id,
        name=finding.name,
        prob=finding.prob,
        status=finding.status,
        region=region_name,
    )


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
        calibrated_prob = models.calibration.apply(p.node_id, p.prob)
        status = _classify_status(calibrated_prob, pos_th, unc_th)
        finding_probs[p.node_id] = calibrated_prob
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
                prob=calibrated_prob,
                raw_prob=p.prob,
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

    # sort before building evidence so evidence ids (F1, F2, ...) follow the same
    # descending-probability order the API/UI display the findings in
    findings.sort(key=lambda f: -f.prob)

    t0 = time.perf_counter()
    evidence_findings = [_evidence_view(f, models.graph) for f in findings]
    narration = narrate(evidence_findings, diagnosis_scores, models.graph, client=models.llm_client)
    report_text = flatten_report(narration.report, settings.disclaimer)
    timings["narration_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = sum(timings.values())

    return AnalysisResult(
        id=result_id,
        created_at=time.time(),
        findings=findings,
        differentials=differentials,
        report=narration.report,
        report_source=narration.source,
        grounding=narration.grounding,
        evidence=narration.evidence,
        report_text=report_text,
        timings_ms=timings,
        disclaimer=settings.disclaimer,
    )
