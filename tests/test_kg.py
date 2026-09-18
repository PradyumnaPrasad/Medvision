from pathlib import Path

import pytest

from medvision.config import get_labels_map, get_settings
from medvision.kg.build import KnowledgeGraphBuildError, build_graph, load_json, save_json
from medvision.kg.graph import KGEdge, KGNode, KnowledgeGraph, load_ontology
from medvision.kg.validate import validate

ONTOLOGY_DIR = get_settings().paths.resolve("kg_ontology_dir")


def test_real_ontology_is_valid():
    graph = build_graph(ONTOLOGY_DIR, get_labels_map())
    summary = graph.summary()
    assert summary["Finding"] >= 18
    assert summary["Disease"] >= 10
    assert summary["Anatomy"] >= 10
    assert summary["edges"] >= 50


def test_infiltration_pneumonia_weight_stays_at_evidence_based_value():
    """Regression guard for reports/false_alarm_investigation.md: this edge was
    lowered from 0.6 to 0.4 after real held-out data showed F_INFILTRATION alone
    was responsible for 100% of D_PNEUMONIA's false-alarm cases. Don't let it
    silently drift back up without re-running that investigation."""
    graph = load_ontology(ONTOLOGY_DIR)
    edge = next(
        e for e in graph.out_edges("F_INFILTRATION", rel="suggestive_of") if e.dst == "D_PNEUMONIA"
    )
    assert edge.weight == pytest.approx(0.4)


def test_every_label_maps_to_a_real_finding_node():
    graph = load_ontology(ONTOLOGY_DIR)
    finding_ids = {n.id for n in graph.nodes_of_type("Finding")}
    for label, node_id in get_labels_map().items():
        assert node_id in finding_ids, f"{label} -> {node_id} missing from ontology"


def test_validate_catches_dangling_edge():
    nodes = [KGNode(id="F_X", type="Finding", name="X", definition="d")]
    edges = [KGEdge(src="F_X", rel="suggestive_of", dst="D_MISSING", weight=0.5)]
    graph = KnowledgeGraph(nodes, edges)
    errors = validate(graph)
    assert any("unknown dst" in e for e in errors)


def test_validate_catches_bad_weight():
    nodes = [
        KGNode(id="F_X", type="Finding", name="X", definition="d"),
        KGNode(id="D_Y", type="Disease", name="Y", definition="d"),
    ]
    edges = [KGEdge(src="F_X", rel="suggestive_of", dst="D_Y", weight=1.5)]
    graph = KnowledgeGraph(nodes, edges)
    errors = validate(graph)
    assert any("out of [0,1]" in e for e in errors)


def test_validate_catches_unreachable_disease():
    nodes = [
        KGNode(id="F_X", type="Finding", name="X", definition="d"),
        KGNode(id="D_Y", type="Disease", name="Y", definition="d"),
        KGNode(id="D_ORPHAN", type="Disease", name="Orphan", definition="d"),
    ]
    edges = [KGEdge(src="F_X", rel="suggestive_of", dst="D_Y", weight=0.5)]
    graph = KnowledgeGraph(nodes, edges)
    errors = validate(graph)
    assert any("D_ORPHAN" in e and "not reachable" in e for e in errors)


def test_validate_catches_bad_relation_type():
    nodes = [
        KGNode(id="F_X", type="Finding", name="X", definition="d"),
        KGNode(id="D_Y", type="Disease", name="Y", definition="d"),
    ]
    edges = [KGEdge(src="F_X", rel="causes", dst="D_Y", weight=0.5)]
    graph = KnowledgeGraph(nodes, edges)
    errors = validate(graph)
    assert any("unknown relation" in e for e in errors)


def test_build_graph_raises_on_invalid_ontology(tmp_path: Path):
    (tmp_path / "findings.yaml").write_text(
        "nodes:\n  - id: F_X\n    type: Finding\n    name: X\n    definition: d\n"
    )
    (tmp_path / "relations.yaml").write_text(
        "edges:\n  - src: F_X\n    rel: suggestive_of\n    dst: D_MISSING\n    weight: 0.5\n"
    )
    with pytest.raises(KnowledgeGraphBuildError):
        build_graph(tmp_path)


def test_save_and_load_json_roundtrip(tmp_path: Path):
    graph = build_graph(ONTOLOGY_DIR, get_labels_map())
    out = tmp_path / "graph.json"
    save_json(graph, out)
    reloaded = load_json(out)
    assert len(reloaded) == len(graph)
    assert len(reloaded.edges) == len(graph.edges)
    assert reloaded.get_node("F_EFFUSION").name == "Pleural Effusion"
