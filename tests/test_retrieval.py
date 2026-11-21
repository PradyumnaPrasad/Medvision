from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from medvision.align.graph_encoder import build_graph_tensors
from medvision.align.model import AlignmentModel, finding_node_ids
from medvision.config import AlignConfig, get_labels_map, get_settings
from medvision.kg.build import build_graph
from medvision.pipeline import analyze, get_models
from medvision.reasoning.prompt import CASE_RULE, build_evidence_block, build_prompt
from medvision.reasoning.schema import LLMReport, ReportItem
from medvision.reasoning.verifier import verify
from medvision.retrieval import search as search_module
from medvision.retrieval.index import (
    get_collection,
    rebuild_case_index,
    rebuild_node_index,
)
from medvision.retrieval.search import RetrievalResult, RetrievalService, SimilarCase

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cxr.png"
FEATURE_DIM = 24


@pytest.fixture(scope="module")
def kg():
    return build_graph(get_settings().paths.resolve("kg_ontology_dir"), get_labels_map())


@pytest.fixture(scope="module")
def model(kg):
    cfg = AlignConfig(embed_dim=16, hidden_dim=32, gnn_heads=2, feature_dim=FEATURE_DIM)
    torch.manual_seed(0)
    m = AlignmentModel(build_graph_tensors(kg), cfg).eval()
    m.set_findings(finding_node_ids(kg))
    return m


def build_index(model, kg, chroma_dir, n_cases=30, seed=0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n_cases, FEATURE_DIM)).astype(np.float32)
    with torch.no_grad():
        emb = model.image_embeddings(torch.tensor(feats)).numpy()
        node_emb = model.finding_embeddings().numpy()
    cfg = get_settings().align
    rebuild_node_index(
        chroma_dir,
        cfg.chroma_collection_nodes,
        model.finding_ids,
        [kg.get_node(n).name for n in model.finding_ids],
        node_emb,
        "v1",
    )
    labels = [["Effusion"] if i % 2 else ["No Finding"] for i in range(n_cases)]
    nodes = [["F_EFFUSION"] if i % 2 else ["F_NO_FINDING"] for i in range(n_cases)]
    count = rebuild_case_index(
        chroma_dir,
        cfg.chroma_collection_cases,
        [f"case-{i}" for i in range(n_cases)],
        emb,
        labels,
        nodes,
        ["train"] * n_cases,
        "v1",
    )
    return feats, count


def make_service(model, kg, chroma_dir):
    return RetrievalService(model, kg, chroma_dir)


def test_index_is_idempotent_and_counts_match(model, kg, tmp_path):
    _, first = build_index(model, kg, tmp_path)
    _, second = build_index(model, kg, tmp_path)
    cfg = get_settings().align
    assert first == second == 30
    assert get_collection(tmp_path, cfg.chroma_collection_cases).count() == 30
    assert get_collection(tmp_path, cfg.chroma_collection_nodes).count() == len(model.finding_ids)


def test_rebuild_replaces_old_vectors_instead_of_accumulating(model, kg, tmp_path):
    build_index(model, kg, tmp_path, n_cases=30)
    _, count = build_index(model, kg, tmp_path, n_cases=10)
    assert count == 10


def test_search_returns_ids_scores_and_metadata(model, kg, tmp_path):
    feats, _ = build_index(model, kg, tmp_path)
    service = make_service(model, kg, tmp_path)
    result = service.search(feats[3])

    cfg = get_settings()
    assert len(result.similar_cases) == cfg.retrieval.top_k_cases
    assert len(result.node_hints) == cfg.retrieval.top_k_nodes
    best = result.similar_cases[0]
    assert best.case_id == "case-3" and best.similarity > 0.999  # its own vector
    assert best.labels == ["Effusion"] and best.finding_node_ids == ["F_EFFUSION"]
    assert best.split == "train"
    sims = [c.similarity for c in result.similar_cases]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0001 <= h.score <= 1.0001 and h.name for h in result.node_hints)
    assert {h.node_id for h in result.node_hints} <= set(model.finding_ids)


def test_service_is_none_when_artifacts_are_missing(kg, monkeypatch):
    monkeypatch.setattr(search_module, "artifacts_present", lambda *_: False)
    assert RetrievalService.load_if_available(kg) is None


def test_service_is_none_instead_of_crashing_on_a_corrupt_checkpoint(kg, monkeypatch, tmp_path):
    bad = tmp_path / "alignment.pt"
    bad.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(search_module, "artifacts_present", lambda *_: True)
    monkeypatch.setattr(
        search_module.get_settings().paths.__class__,
        "resolve",
        lambda self, key: tmp_path if key != "models_dir" else tmp_path,
    )
    assert RetrievalService.load_if_available(kg) is None


def _cases():
    return [
        SimpleNamespace(case_id="a", similarity=0.9, labels=["Effusion", "Atelectasis"]),
        SimpleNamespace(case_id="b", similarity=0.8, labels=["No Finding"]),
    ]


def test_evidence_block_gets_citable_case_ids(kg):
    evidence = build_evidence_block([], [], kg, _cases())
    assert [c.id for c in evidence.cases] == ["C1", "C2"]
    assert {"C1", "C2"} <= evidence.all_ids()
    assert "Effusion, Atelectasis" in evidence.cases[0].text


def test_prompt_only_mentions_cases_when_there_are_some(kg):
    without = build_prompt(build_evidence_block([], [], kg))
    with_cases = build_prompt(build_evidence_block([], [], kg, _cases()))
    assert CASE_RULE not in without and '"cases": []' in without
    assert CASE_RULE in with_cases and "Similar reference case" in with_cases


def test_verifier_accepts_real_case_ids_and_rejects_invented_ones(kg):
    evidence = build_evidence_block([], [], kg, _cases())
    ok = LLMReport(
        findings=[],
        impression=[ReportItem(text="Similar reference cases agree.", evidence_ids=["C1"])],
        differentials=[],
        limitations=[],
        recommendation="",
    )
    assert verify(ok, evidence, kg).score == 1.0
    bad = ok.model_copy(update={"impression": [ReportItem(text="Made up.", evidence_ids=["C9"])]})
    result = verify(bad, evidence, kg)
    assert result.score == 0.0 and any("C9" in u for u in result.unsupported)


class _StubRetrieval:
    def search(self, features):
        assert features.shape == (1024,)
        return RetrievalResult(
            node_hints=[],
            similar_cases=[
                SimilarCase(f"c{i}", 0.9 - i / 10, ["Effusion"], ["F_EFFUSION"], "train")
                for i in range(5)
            ],
        )


@pytest.mark.slow
def test_pipeline_includes_retrieval_and_feeds_only_a_few_cases_to_the_report(monkeypatch):
    models = get_models()
    monkeypatch.setattr(models.llm_client, "is_available", lambda: False)
    monkeypatch.setattr(models, "retrieval", _StubRetrieval())
    result = analyze(Image.open(FIXTURE))

    assert [c.case_id for c in result.retrieval.similar_cases] == [f"c{i}" for i in range(5)]
    assert len(result.evidence.cases) == get_settings().retrieval.cases_in_prompt
    assert "retrieval_ms" in result.timings_ms


@pytest.mark.slow
def test_pipeline_works_without_retrieval_artifacts(monkeypatch):
    models = get_models()
    monkeypatch.setattr(models.llm_client, "is_available", lambda: False)
    monkeypatch.setattr(models, "retrieval", None)
    result = analyze(Image.open(FIXTURE))
    assert result.retrieval is None and result.evidence.cases == []
    assert "retrieval_ms" not in result.timings_ms and result.findings


def test_results_stored_before_retrieval_existed_still_load():
    from medvision.pipeline import AnalysisResult
    from tests.test_report_pdf import make_result

    legacy = make_result().model_dump()
    legacy.pop("retrieval")
    legacy["evidence"].pop("cases")
    loaded = AnalysisResult.model_validate(legacy)
    assert loaded.retrieval is None and loaded.evidence.cases == []
