"""Build the MEDVISION knowledge graph from kg/ontology/ YAML and persist it as JSON.

This is the only place the hand-authored ontology is turned into the artefact the
rest of the app reads (data/kg/graph.json). Validation always runs first — a build
never produces an invalid graph on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from medvision.kg.graph import KGEdge, KGNode, KnowledgeGraph, load_ontology
from medvision.kg.validate import validate


class KnowledgeGraphBuildError(ValueError):
    """Raised when the ontology fails validation; message lists every error found."""


def build_graph(ontology_dir: Path, labels_map: dict[str, str] | None = None) -> KnowledgeGraph:
    graph = load_ontology(ontology_dir)
    errors = validate(graph, labels_map)
    if errors:
        formatted = "\n  - ".join(errors)
        raise KnowledgeGraphBuildError(f"Knowledge graph failed validation:\n  - {formatted}")
    return graph


def to_json_dict(graph: KnowledgeGraph) -> dict:
    return {
        "nodes": [asdict(n) for n in graph.nodes_by_id.values()],
        "edges": [asdict(e) for e in graph.edges],
    }


def save_json(graph: KnowledgeGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(to_json_dict(graph), f, indent=2)


def load_json(path: Path) -> KnowledgeGraph:
    with path.open() as f:
        raw = json.load(f)
    nodes = [KGNode(**n) for n in raw["nodes"]]
    edges = [KGEdge(**e) for e in raw["edges"]]
    return KnowledgeGraph(nodes, edges)
