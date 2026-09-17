import pytest

from medvision.config import get_labels_map, get_settings
from medvision.kg.build import build_graph
from medvision.kg.graph import KGEdge, KGNode, KnowledgeGraph
from medvision.kg.reasoning import score_differentials

ONTOLOGY_DIR = get_settings().paths.resolve("kg_ontology_dir")


def _toy_graph() -> KnowledgeGraph:
    nodes = [
        KGNode(id="F_A", type="Finding", name="Finding A", definition="d"),
        KGNode(id="F_B", type="Finding", name="Finding B", definition="d"),
        KGNode(id="D_X", type="Disease", name="Disease X", definition="d"),
        KGNode(id="D_Y", type="Disease", name="Disease Y", definition="d"),
    ]
    edges = [
        KGEdge(src="F_A", rel="suggestive_of", dst="D_X", weight=0.8),
        KGEdge(src="F_A", rel="suggestive_of", dst="D_Y", weight=0.2),
        KGEdge(src="F_B", rel="suggestive_of", dst="D_Y", weight=0.9),
        KGEdge(src="F_B", rel="contradicts", dst="D_X", weight=0.5),
    ]
    return KnowledgeGraph(nodes, edges)


def test_known_toy_scores():
    graph = _toy_graph()
    # F_A activated at 1.0, F_B at 1.0
    # raw(D_X) = 1.0*0.8 (suggestive) - 1.0*1.0*0.5 (contradicts, penalty=1.0) = 0.3
    # raw(D_Y) = 1.0*0.2 + 1.0*0.9 = 1.1
    results = score_differentials(
        graph,
        {"F_A": 1.0, "F_B": 1.0},
        top_k=2,
        activation_threshold=0.3,
        contradiction_penalty_weight=1.0,
    )
    by_id = {r.node_id: r for r in results}
    assert by_id["D_Y"].raw_score == pytest.approx(1.1)
    assert by_id["D_X"].raw_score == pytest.approx(0.3)
    # D_Y should be top scored (normalized to 1.0), D_X below it
    assert results[0].node_id == "D_Y"
    assert results[0].score == 1.0
    assert 0.0 < results[1].score < 1.0


def test_activation_threshold_excludes_low_probability_findings():
    graph = _toy_graph()
    results = score_differentials(graph, {"F_A": 0.1, "F_B": 1.0}, activation_threshold=0.3)
    # F_A (0.1) is below threshold, so only F_B's edges contribute
    by_id = {r.node_id: r for r in results}
    assert "D_Y" in by_id
    assert by_id["D_Y"].raw_score == pytest.approx(0.9)
    # D_X only had F_A(suggestive)+F_B(contradicts); F_A excluded -> raw = -0.5, normalized 0
    assert by_id["D_X"].score == 0.0


def test_no_activated_findings_returns_empty():
    graph = _toy_graph()
    assert score_differentials(graph, {"F_A": 0.01, "F_B": 0.01}, activation_threshold=0.3) == []


def test_paths_are_sorted_by_contribution_magnitude():
    graph = _toy_graph()
    results = score_differentials(
        graph, {"F_A": 1.0, "F_B": 1.0}, activation_threshold=0.3, top_k=2
    )
    by_id = {r.node_id: r for r in results}
    dx_paths = by_id["D_X"].paths
    assert len(dx_paths) == 2
    assert abs(dx_paths[0].contribution) >= abs(dx_paths[1].contribution)


def test_real_ontology_produces_plausible_differentials():
    graph = build_graph(ONTOLOGY_DIR, get_labels_map())
    # simulate a classifier that strongly suspects consolidation + pneumonia-pattern opacity
    finding_probs = {
        "F_CONSOLIDATION": 0.9,
        "F_PNEUMONIA_OPACITY": 0.85,
        "F_EFFUSION": 0.4,
        "F_CARDIOMEGALY": 0.1,
    }
    results = score_differentials(graph, finding_probs)
    assert results, "expected at least one differential"
    assert results[0].node_id == "D_PNEUMONIA"
    assert results[0].score == 1.0
    assert len(results[0].paths) >= 2  # both consolidation and pneumonia-opacity contribute
