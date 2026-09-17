from dataclasses import dataclass

import httpx
import pytest
from pydantic import ValidationError

from medvision.kg.graph import KGEdge, KGNode, KnowledgeGraph
from medvision.kg.reasoning import score_differentials
from medvision.reasoning.llm import LlmInvalidResponseError, LlmUnavailableError, OllamaClient
from medvision.reasoning.narrate import narrate
from medvision.reasoning.prompt import build_evidence_block, build_prompt
from medvision.reasoning.schema import (
    DifferentialItem,
    EvidenceBlock,
    EvidenceDifferential,
    EvidenceFinding,
    EvidenceKgFact,
    LLMReport,
    ReportItem,
)
from medvision.reasoning.template import build_template_report
from medvision.reasoning.verifier import verify


@dataclass
class FakeFinding:
    node_id: str
    name: str
    prob: float
    status: str
    region: str | None = None


def _toy_graph() -> KnowledgeGraph:
    nodes = [
        KGNode(id="F_CONSOLIDATION", type="Finding", name="Consolidation", definition="d"),
        KGNode(
            id="F_EFFUSION",
            type="Finding",
            name="Pleural Effusion",
            definition="d",
            aliases=["fluid"],
        ),
        KGNode(id="D_PNEUMONIA", type="Disease", name="Pneumonia", definition="d"),
        KGNode(id="D_CHF", type="Disease", name="Congestive Heart Failure", definition="d"),
    ]
    edges = [
        KGEdge(src="F_CONSOLIDATION", rel="suggestive_of", dst="D_PNEUMONIA", weight=0.85),
        KGEdge(src="F_EFFUSION", rel="suggestive_of", dst="D_CHF", weight=0.6),
    ]
    return KnowledgeGraph(nodes, edges)


def _sample_findings() -> list[FakeFinding]:
    return [
        FakeFinding("F_CONSOLIDATION", "Consolidation", 0.9, "present", "A_RIGHT_LOWER_ZONE"),
        FakeFinding("F_EFFUSION", "Pleural Effusion", 0.4, "possible", None),
        FakeFinding("F_NODULE", "Pulmonary Nodule", 0.05, "absent", None),
    ]


# ---------- evidence building ----------


def test_build_evidence_block_only_includes_non_absent_findings():
    graph = _toy_graph()
    findings = _sample_findings()
    finding_probs = {f.node_id: f.prob for f in findings}
    diffs = score_differentials(graph, finding_probs, activation_threshold=0.3)

    evidence = build_evidence_block(findings, diffs, graph)

    assert [f.name for f in evidence.findings] == ["Consolidation", "Pleural Effusion"]
    assert evidence.findings[0].id == "F1"
    assert evidence.findings[1].id == "F2"
    assert len(evidence.differentials) == len(diffs)
    assert evidence.kg_facts  # at least one fact behind the differentials


def test_build_evidence_block_dedupes_kg_facts():
    graph = _toy_graph()
    findings = _sample_findings()
    diffs = score_differentials(
        graph, {f.node_id: f.prob for f in findings}, activation_threshold=0.3
    )
    evidence = build_evidence_block(findings, diffs, graph)
    # each finding->disease edge should appear as exactly one kg_fact, referenced by id
    all_fact_ids = {k.id for k in evidence.kg_facts}
    for d in evidence.differentials:
        assert set(d.supporting_fact_ids) <= all_fact_ids


def test_build_prompt_includes_instructions_and_evidence():
    graph = _toy_graph()
    findings = _sample_findings()
    diffs = score_differentials(
        graph, {f.node_id: f.prob for f in findings}, activation_threshold=0.3
    )
    evidence = build_evidence_block(findings, diffs, graph)
    prompt = build_prompt(evidence)
    assert "EVIDENCE" in prompt
    assert "Consolidation" in prompt
    assert "evidence_ids" in prompt


# ---------- template fallback ----------


def test_template_report_is_fully_grounded():
    graph = _toy_graph()
    findings = _sample_findings()
    diffs = score_differentials(
        graph, {f.node_id: f.prob for f in findings}, activation_threshold=0.3
    )
    evidence = build_evidence_block(findings, diffs, graph)

    report = build_template_report(evidence)
    grounding = verify(report, evidence, graph)

    assert grounding.score == 1.0
    assert not grounding.unsupported
    assert not grounding.contradictions


# ---------- verifier ----------


def _base_evidence() -> EvidenceBlock:
    return EvidenceBlock(
        findings=[
            EvidenceFinding(
                id="F1", node_id="F_CONSOLIDATION", name="Consolidation", prob=0.9, status="present"
            ),
        ],
        differentials=[
            EvidenceDifferential(
                id="D1",
                node_id="D_PNEUMONIA",
                name="Pneumonia",
                score=1.0,
                supporting_fact_ids=["K1"],
            ),
        ],
        kg_facts=[
            EvidenceKgFact(id="K1", text="Consolidation is suggestive of Pneumonia (weight=0.85)")
        ],
    )


def test_verify_flags_unknown_evidence_id():
    graph = _toy_graph()
    evidence = _base_evidence()
    report = LLMReport(
        findings=[ReportItem(text="Consolidation is present.", evidence_ids=["F99"])],
        impression=[ReportItem(text="Consistent with pneumonia.", evidence_ids=["D1"])],
        differentials=[
            DifferentialItem(diagnosis="Pneumonia", rationale="see K1", evidence_ids=["D1", "K1"])
        ],
        limitations=["single view"],
        recommendation="follow up",
    )
    result = verify(report, evidence, graph)
    assert result.score < 1.0
    assert any("F99" in msg for msg in result.unsupported)


def test_report_item_rejects_empty_evidence_ids_at_construction():
    """The 'at least one citation' rule is enforced by the schema itself (minItems=1 in
    the JSON schema handed to Ollama), not just by the verifier after the fact — so an
    LLM (or template) can't produce an uncited item in the first place."""
    with pytest.raises(ValidationError):
        ReportItem(text="Consolidation is present.", evidence_ids=[])
    with pytest.raises(ValidationError):
        DifferentialItem(diagnosis="Pneumonia", rationale="x", evidence_ids=[])


def test_verify_flags_contradiction_when_present_finding_denied():
    graph = _toy_graph()
    evidence = _base_evidence()
    report = LLMReport(
        findings=[ReportItem(text="There is no consolidation seen.", evidence_ids=["F1"])],
        impression=[ReportItem(text="Consistent with pneumonia.", evidence_ids=["D1"])],
        differentials=[
            DifferentialItem(diagnosis="Pneumonia", rationale="see K1", evidence_ids=["D1", "K1"])
        ],
        limitations=[],
        recommendation="follow up",
    )
    result = verify(report, evidence, graph)
    assert result.contradictions
    assert "Consolidation" in result.contradictions[0]


def test_verify_flags_unknown_differential_name():
    graph = _toy_graph()
    evidence = _base_evidence()
    report = LLMReport(
        findings=[ReportItem(text="Consolidation is present.", evidence_ids=["F1"])],
        impression=[ReportItem(text="Consistent with something.", evidence_ids=["D1"])],
        differentials=[
            DifferentialItem(diagnosis="Tuberculosis", rationale="see K1", evidence_ids=["K1"])
        ],
        limitations=[],
        recommendation="follow up",
    )
    result = verify(report, evidence, graph)
    assert any("not one of the evidence differentials" in msg for msg in result.unsupported)


def test_verify_perfect_report_scores_one():
    graph = _toy_graph()
    evidence = _base_evidence()
    report = LLMReport(
        findings=[
            ReportItem(
                text="Consolidation is present in the right lower zone.", evidence_ids=["F1"]
            )
        ],
        impression=[
            ReportItem(text="Findings are consistent with pneumonia.", evidence_ids=["D1", "K1"])
        ],
        differentials=[
            DifferentialItem(
                diagnosis="Pneumonia", rationale="Supported by K1", evidence_ids=["D1", "K1"]
            )
        ],
        limitations=["Single view."],
        recommendation="Clinical correlation recommended.",
    )
    result = verify(report, evidence, graph)
    assert result.score == 1.0
    assert not result.unsupported
    assert not result.contradictions


# ---------- OllamaClient (network mocked) ----------


class _FakeResponse:
    def __init__(self, json_body: dict, status_code: int = 200):
        self._json = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


def _valid_llm_report_json() -> str:
    return LLMReport(
        findings=[ReportItem(text="Consolidation is present.", evidence_ids=["F1"])],
        impression=[ReportItem(text="Consistent with pneumonia.", evidence_ids=["D1"])],
        differentials=[
            DifferentialItem(diagnosis="Pneumonia", rationale="K1", evidence_ids=["D1", "K1"])
        ],
        limitations=["single view"],
        recommendation="follow up",
    ).model_dump_json()


def test_ollama_client_parses_valid_response(monkeypatch):
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakeResponse({"response": _valid_llm_report_json()})
    )
    client = OllamaClient(base_url="http://fake", model="test-model", max_retries=0)
    report = client.generate_report("some prompt")
    assert report.differentials[0].diagnosis == "Pneumonia"


def test_ollama_client_raises_on_connection_error(monkeypatch):
    def _raise(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _raise)
    client = OllamaClient(base_url="http://fake", model="test-model", max_retries=0)
    with pytest.raises(LlmUnavailableError):
        client.generate_report("prompt")


def test_ollama_client_raises_after_retries_on_bad_json(monkeypatch):
    calls = {"n": 0}

    def _post(*a, **k):
        calls["n"] += 1
        return _FakeResponse({"response": "not json"})

    monkeypatch.setattr(httpx, "post", _post)
    client = OllamaClient(base_url="http://fake", model="test-model", max_retries=1)
    with pytest.raises(LlmInvalidResponseError):
        client.generate_report("prompt")
    assert calls["n"] == 2  # first try + 1 retry


def test_ollama_client_is_available(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse({}, status_code=200))
    assert OllamaClient(base_url="http://fake").is_available() is True

    def _raise(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _raise)
    assert OllamaClient(base_url="http://fake").is_available() is False


# ---------- narrate() orchestration ----------


class _StubClient:
    def __init__(self, available: bool, reports: list):
        self._available = available
        self._reports = list(reports)

    def is_available(self):
        return self._available

    def generate_report(self, prompt: str) -> LLMReport:
        if not self._reports:
            raise LlmInvalidResponseError("no more stub reports")
        return self._reports.pop(0)


def test_narrate_uses_template_when_llm_unavailable():
    graph = _toy_graph()
    findings = _sample_findings()
    diffs = score_differentials(
        graph, {f.node_id: f.prob for f in findings}, activation_threshold=0.3
    )
    result = narrate(findings, diffs, graph, client=_StubClient(available=False, reports=[]))
    assert result.source == "template"
    assert result.grounding.score == 1.0


def test_narrate_accepts_good_first_llm_report():
    graph = _toy_graph()
    findings = _sample_findings()
    diffs = score_differentials(
        graph, {f.node_id: f.prob for f in findings}, activation_threshold=0.3
    )
    evidence = build_evidence_block(findings, diffs, graph)
    good_report = build_template_report(evidence)  # already fully grounded
    result = narrate(
        findings, diffs, graph, client=_StubClient(available=True, reports=[good_report])
    )
    assert result.source == "llm"
    assert result.grounding.score == 1.0


def test_narrate_regenerates_once_when_first_attempt_is_poorly_grounded():
    graph = _toy_graph()
    findings = _sample_findings()
    diffs = score_differentials(
        graph, {f.node_id: f.prob for f in findings}, activation_threshold=0.3
    )
    evidence = build_evidence_block(findings, diffs, graph)

    bad_report = LLMReport(
        findings=[ReportItem(text="Consolidation is present.", evidence_ids=["F99"])],  # bad id
        impression=[ReportItem(text="x", evidence_ids=["D1"])],
        differentials=[
            DifferentialItem(diagnosis="Pneumonia", rationale="x", evidence_ids=["D1", "K1"])
        ],
        limitations=[],
        recommendation="x",
    )
    good_report = build_template_report(evidence)

    result = narrate(
        findings,
        diffs,
        graph,
        client=_StubClient(available=True, reports=[bad_report, good_report]),
    )
    assert result.source == "llm"
    assert result.grounding.score == 1.0  # took the better second attempt


def test_narrate_falls_back_to_template_when_llm_keeps_failing():
    graph = _toy_graph()
    findings = _sample_findings()
    diffs = score_differentials(
        graph, {f.node_id: f.prob for f in findings}, activation_threshold=0.3
    )

    class _AlwaysFailsClient:
        def is_available(self):
            return True

        def generate_report(self, prompt: str):
            raise LlmInvalidResponseError("nope")

    result = narrate(findings, diffs, graph, client=_AlwaysFailsClient())
    assert result.source == "template"
