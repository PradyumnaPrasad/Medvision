"""Evidence + LLM-report schemas for grounded narration (PLAN.md section 4.6, T22-T24).

The LLM never decides *what* is wrong — medvision/kg/reasoning.py already did that
deterministically. The LLM only narrates the evidence handed to it in EvidenceBlock,
and every sentence it writes must cite the evidence item(s) it came from by id
(F1, D1, K1, ...). medvision/reasoning/verifier.py checks those citations; nothing
here trusts the LLM's output until it's been verified.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceFinding(BaseModel):
    id: str  # "F1", "F2", ...
    node_id: str
    name: str
    prob: float
    status: str
    region: str | None = None


class EvidenceDifferential(BaseModel):
    id: str  # "D1", "D2", ...
    node_id: str
    name: str
    score: float
    supporting_fact_ids: list[str]  # ids into EvidenceBlock.kg_facts


class EvidenceKgFact(BaseModel):
    id: str  # "K1", "K2", ...
    text: str  # human-readable edge, e.g. "Consolidation suggestive_of Pneumonia (w=0.85)"


class EvidenceCase(BaseModel):
    id: str  # "C1", "C2", ...
    case_id: str
    similarity: float
    labels: list[str]
    text: str  # e.g. "Similar reference case (similarity 0.83) was labelled: Effusion"


class EvidenceBlock(BaseModel):
    findings: list[EvidenceFinding]
    differentials: list[EvidenceDifferential]
    kg_facts: list[EvidenceKgFact]
    cases: list[EvidenceCase] = Field(default_factory=list)

    def all_ids(self) -> set[str]:
        ids = {f.id for f in self.findings}
        ids |= {d.id for d in self.differentials}
        ids |= {k.id for k in self.kg_facts}
        ids |= {c.id for c in self.cases}
        return ids


class ReportItem(BaseModel):
    """One sentence/statement plus the evidence ids it must be traceable to.

    evidence_ids requires at least one entry: this is enforced in the JSON schema
    handed to Ollama (as minItems=1), so grammar-constrained decoding structurally
    cannot produce an uncited item — we don't have to rely on the model reading and
    following the "always cite" instruction in the prompt.
    """

    text: str
    evidence_ids: list[str] = Field(min_length=1)


class DifferentialItem(BaseModel):
    diagnosis: str
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)


class LLMReport(BaseModel):
    """The structured output an LLM (or the template fallback) must produce.

    Passed to Ollama as a JSON schema via LLMReport.model_json_schema() so the model
    is grammar-constrained to this exact shape.
    """

    findings: list[ReportItem]
    impression: list[ReportItem]
    differentials: list[DifferentialItem]
    limitations: list[str]
    recommendation: str


class GroundingResult(BaseModel):
    score: float  # fraction of citable items whose citations & claims check out
    unsupported: list[str]  # human-readable descriptions of items that failed
    contradictions: list[str]
