"""Checks an LLMReport against the EvidenceBlock it was supposed to be grounded in.

This is what keeps MEDVISION's narration honest: the LLM's job is only to phrase the
evidence in prose, and this module is what actually enforces that it did. Nothing
downstream (the API response, the UI) should show an LLM report that hasn't been
run through verify() first.
"""

from __future__ import annotations

import re

from medvision.kg.graph import KnowledgeGraph
from medvision.reasoning.schema import EvidenceBlock, GroundingResult, LLMReport

_NEGATION_PATTERN = re.compile(
    r"\b(no |not |without|absent|absence of|rules? out|ruled out|excludes?|negative for)\b"
)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _mentions_finding_as_absent(text: str, names: list[str]) -> bool:
    """True if any sentence in `text` both names the finding and negates it."""
    for sentence in _split_sentences(text.lower()):
        if any(name in sentence for name in names if name) and _NEGATION_PATTERN.search(sentence):
            return True
    return False


class _Scorer:
    def __init__(self, evidence: EvidenceBlock, graph: KnowledgeGraph) -> None:
        self.evidence = evidence
        self.graph = graph
        self.valid_ids = evidence.all_ids()
        self.total = 0
        self.supported = 0
        self.unsupported: list[str] = []
        self.contradictions: list[str] = []

    def _finding_aliases(self, node_id: str) -> list[str]:
        node = self.graph.get_node(node_id)
        return [node.name.lower(), *[a.lower() for a in node.aliases]] if node else []

    def check(self, text: str, evidence_ids: list[str], label: str, extra_ok: bool = True) -> None:
        self.total += 1
        ok = extra_ok

        if not evidence_ids:
            self.unsupported.append(f"{label}: has no evidence_ids")
            ok = False
        missing = [i for i in evidence_ids if i not in self.valid_ids]
        if missing:
            self.unsupported.append(f"{label}: cites unknown evidence id(s) {missing}")
            ok = False

        for finding in self.evidence.findings:
            names = self._finding_aliases(finding.node_id) or [finding.name.lower()]
            if _mentions_finding_as_absent(text, names):
                self.contradictions.append(
                    f"{label}: implies '{finding.name}' is absent, but evidence marks it "
                    f"{finding.status} (p={finding.prob:.2f})"
                )
                ok = False

        if ok:
            self.supported += 1

    def result(self) -> GroundingResult:
        score = self.supported / self.total if self.total else 0.0
        return GroundingResult(
            score=score, unsupported=self.unsupported, contradictions=self.contradictions
        )


def verify(report: LLMReport, evidence: EvidenceBlock, graph: KnowledgeGraph) -> GroundingResult:
    scorer = _Scorer(evidence, graph)
    known_differential_names = {d.name.lower() for d in evidence.differentials}

    for item in report.findings:
        scorer.check(item.text, item.evidence_ids, "finding statement")
    for item in report.impression:
        scorer.check(item.text, item.evidence_ids, "impression statement")
    for item in report.differentials:
        name_ok = item.diagnosis.lower() in known_differential_names
        label = f"differential '{item.diagnosis}'"
        if not name_ok:
            scorer.unsupported.append(f"{label} is not one of the evidence differentials")
        scorer.check(item.rationale, item.evidence_ids, label, extra_ok=name_ok)

    return scorer.result()


def feedback_text(result: GroundingResult) -> str:
    """Turn a failed GroundingResult into instructions for a regeneration attempt."""
    lines: list[str] = []
    for msg in result.unsupported:
        lines.append(f"- Fix: {msg}")
    for msg in result.contradictions:
        lines.append(f"- Fix: {msg}")
    return "\n".join(lines)
