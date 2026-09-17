"""Orchestrates LLM narration with verification, regeneration, and a template fallback.

This is the single entry point medvision/pipeline.py calls; it hides the decision of
whether the LLM was used at all, so the pipeline's contract (an AnalysisResult always
has a report) never breaks just because Ollama is down.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from medvision.config import get_settings
from medvision.kg.graph import KnowledgeGraph
from medvision.kg.reasoning import DiagnosisScore
from medvision.reasoning.llm import LlmInvalidResponseError, LlmUnavailableError, OllamaClient
from medvision.reasoning.prompt import FindingLike, build_evidence_block, build_prompt
from medvision.reasoning.schema import EvidenceBlock, GroundingResult, LLMReport
from medvision.reasoning.template import build_template_report
from medvision.reasoning.verifier import feedback_text, verify

logger = logging.getLogger(__name__)

_TEMPLATE_GROUNDING = GroundingResult(score=1.0, unsupported=[], contradictions=[])


class NarrationResult(NamedTuple):
    report: LLMReport
    grounding: GroundingResult
    source: str  # "llm" | "template"
    evidence: EvidenceBlock


def narrate(
    findings: list[FindingLike],
    differentials: list[DiagnosisScore],
    graph: KnowledgeGraph,
    client: OllamaClient | None = None,
) -> NarrationResult:
    evidence = build_evidence_block(findings, differentials, graph)
    client = client or OllamaClient()

    if not client.is_available():
        logger.info("Ollama unavailable; using template report.")
        return NarrationResult(
            build_template_report(evidence), _TEMPLATE_GROUNDING, "template", evidence
        )

    threshold = get_settings().llm.grounding_regenerate_threshold
    try:
        report = client.generate_report(build_prompt(evidence))
        grounding = verify(report, evidence, graph)

        if grounding.score < threshold:
            logger.info("Grounding score %.2f below threshold; regenerating once.", grounding.score)
            retry_prompt = build_prompt(evidence, feedback=feedback_text(grounding))
            report2 = client.generate_report(retry_prompt)
            grounding2 = verify(report2, evidence, graph)
            if grounding2.score > grounding.score:
                report, grounding = report2, grounding2

        return NarrationResult(report, grounding, "llm", evidence)
    except (LlmUnavailableError, LlmInvalidResponseError) as exc:
        logger.warning("LLM narration failed (%s); falling back to template report.", exc)
        return NarrationResult(
            build_template_report(evidence), _TEMPLATE_GROUNDING, "template", evidence
        )
