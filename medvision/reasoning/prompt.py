"""Builds the EvidenceBlock the LLM is allowed to narrate, and the prompt around it.

Deliberately decoupled from medvision.pipeline's pydantic types (FindingLike is a
structural Protocol, not an import) so medvision.reasoning never imports
medvision.pipeline, which imports medvision.reasoning — no import cycle.
"""

from __future__ import annotations

from typing import Protocol

from medvision.kg.graph import KnowledgeGraph
from medvision.kg.reasoning import DiagnosisScore
from medvision.reasoning.schema import (
    EvidenceBlock,
    EvidenceDifferential,
    EvidenceFinding,
    EvidenceKgFact,
)


class FindingLike(Protocol):
    node_id: str
    name: str
    prob: float
    status: str
    region: str | None


SYSTEM_INSTRUCTIONS = """You are a radiology report drafting assistant. You are NOT a doctor \
and this draft will always be reviewed by a qualified clinician before it is used.

Rules you must follow exactly:
1. Use ONLY the facts given in the EVIDENCE block below. Never introduce a finding, \
measurement, or diagnosis that is not present in EVIDENCE.
2. Every item you write (each entry in findings, impression, and differentials) MUST \
include the evidence_ids of every evidence item it is based on, using the exact ids \
given (e.g. "F1", "D2", "K3").
3. Do not state that a finding is absent unless its status in EVIDENCE is "absent". Do \
not state a finding is present/possible unless EVIDENCE says so.
4. Rank differentials in the same order EVIDENCE gives them; do not invent a differential \
that is not listed.
5. Keep language concise and clinical. Output must match the required JSON schema exactly.
"""


def build_evidence_block(
    findings: list[FindingLike],
    differentials: list[DiagnosisScore],
    graph: KnowledgeGraph,
) -> EvidenceBlock:
    """Assemble the id-tagged evidence the LLM (and later the verifier) will work from."""
    shown = [f for f in findings if f.status != "absent"]

    evidence_findings = [
        EvidenceFinding(
            id=f"F{i + 1}",
            node_id=f.node_id,
            name=f.name,
            prob=f.prob,
            status=f.status,
            region=f.region,
        )
        for i, f in enumerate(shown)
    ]

    kg_facts: list[EvidenceKgFact] = []
    fact_id_by_edge: dict[tuple[str, str, str], str] = {}

    def _fact_id(src: str, rel: str, dst: str, weight: float) -> str:
        key = (src, rel, dst)
        if key not in fact_id_by_edge:
            src_name = graph.get_node(src).name if graph.get_node(src) else src
            dst_name = graph.get_node(dst).name if graph.get_node(dst) else dst
            verb = "contradicts" if rel == "contradicts" else "is suggestive of"
            text = f"{src_name} {verb} {dst_name} (weight={weight:.2f})"
            fact_id = f"K{len(kg_facts) + 1}"
            kg_facts.append(EvidenceKgFact(id=fact_id, text=text))
            fact_id_by_edge[key] = fact_id
        return fact_id_by_edge[key]

    evidence_differentials: list[EvidenceDifferential] = []
    for i, d in enumerate(differentials):
        supporting_ids = [_fact_id(p.src, p.rel, p.dst, p.weight) for p in d.paths]
        evidence_differentials.append(
            EvidenceDifferential(
                id=f"D{i + 1}",
                node_id=d.node_id,
                name=d.name,
                score=d.score,
                supporting_fact_ids=supporting_ids,
            )
        )

    return EvidenceBlock(
        findings=evidence_findings, differentials=evidence_differentials, kg_facts=kg_facts
    )


def build_user_prompt(evidence: EvidenceBlock, feedback: str | None = None) -> str:
    """The user-turn prompt: the evidence as JSON plus the task instructions."""
    parts = [
        "EVIDENCE:",
        evidence.model_dump_json(indent=2),
        "",
        "TASK: Using only the EVIDENCE above, write a structured draft radiology report. "
        "For 'findings', write one item per evidence finding with status present or possible. "
        "For 'impression', summarize which differential(s) the findings support. "
        "For 'differentials', cover every evidence differential in the same order, citing "
        "its own id and its supporting_fact_ids. Include 2-4 short 'limitations' relevant to "
        "a single-view chest X-ray read by an automated system. Keep 'recommendation' to one "
        "sentence recommending clinical correlation and radiologist review.",
    ]
    if feedback:
        parts += [
            "",
            "Your previous attempt had problems that MUST be fixed this time:",
            feedback,
        ]
    return "\n".join(parts)


def build_prompt(evidence: EvidenceBlock, feedback: str | None = None) -> str:
    return f"{SYSTEM_INSTRUCTIONS}\n\n{build_user_prompt(evidence, feedback)}"
