"""Loading and in-memory representation of the MEDVISION knowledge graph.

The source of truth is the hand-authored YAML in kg/ontology/ (see kg/schema.md).
This module only loads and holds that data; validation lives in validate.py and
persistence of the built graph lives in build.py, so each concern stays testable
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import yaml

VALID_NODE_TYPES = {"Finding", "Anatomy", "Disease"}
VALID_RELATIONS = {
    "suggestive_of",
    "contradicts",
    "differential_of",
    "complication_of",
    "located_in",
    "part_of",
    "associated_with",
}

ONTOLOGY_NODE_FILES = ("findings.yaml", "anatomy.yaml", "diseases.yaml")
ONTOLOGY_EDGE_FILE = "relations.yaml"


@dataclass(frozen=True)
class KGNode:
    id: str
    type: str
    name: str
    definition: str
    aliases: list[str] = field(default_factory=list)
    mesh_id: str | None = None
    source: str = "team"


@dataclass(frozen=True)
class KGEdge:
    src: str
    rel: str
    dst: str
    weight: float
    source: str = "team"
    note: str | None = None


class KnowledgeGraph:
    """A thin wrapper around networkx.MultiDiGraph with typed node/edge access."""

    def __init__(self, nodes: list[KGNode], edges: list[KGEdge]) -> None:
        self.nodes_by_id: dict[str, KGNode] = {n.id: n for n in nodes}
        self.edges: list[KGEdge] = edges
        self.graph = nx.MultiDiGraph()
        for node in nodes:
            self.graph.add_node(node.id, **node.__dict__)
        for edge in edges:
            self.graph.add_edge(edge.src, edge.dst, key=edge.rel, **edge.__dict__)

    def get_node(self, node_id: str) -> KGNode | None:
        return self.nodes_by_id.get(node_id)

    def nodes_of_type(self, node_type: str) -> list[KGNode]:
        return [n for n in self.nodes_by_id.values() if n.type == node_type]

    def out_edges(self, node_id: str, rel: str | None = None) -> list[KGEdge]:
        return [e for e in self.edges if e.src == node_id and (rel is None or e.rel == rel)]

    def in_edges(self, node_id: str, rel: str | None = None) -> list[KGEdge]:
        return [e for e in self.edges if e.dst == node_id and (rel is None or e.rel == rel)]

    def __len__(self) -> int:
        return len(self.nodes_by_id)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {t: 0 for t in VALID_NODE_TYPES}
        for n in self.nodes_by_id.values():
            counts[n.type] = counts.get(n.type, 0) + 1
        counts["edges"] = len(self.edges)
        return counts


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_ontology(ontology_dir: Path) -> KnowledgeGraph:
    """Load all node/edge YAML files under ontology_dir into a KnowledgeGraph."""
    nodes: list[KGNode] = []
    for filename in ONTOLOGY_NODE_FILES:
        path = ontology_dir / filename
        if not path.exists():
            continue
        raw = _load_yaml(path)
        for n in raw.get("nodes", []):
            nodes.append(
                KGNode(
                    id=n["id"],
                    type=n["type"],
                    name=n["name"],
                    definition=n.get("definition", ""),
                    aliases=n.get("aliases", []) or [],
                    mesh_id=n.get("mesh_id"),
                    source=n.get("source", "team"),
                )
            )

    edges: list[KGEdge] = []
    edge_path = ontology_dir / ONTOLOGY_EDGE_FILE
    if edge_path.exists():
        raw = _load_yaml(edge_path)
        for e in raw.get("edges", []):
            edges.append(
                KGEdge(
                    src=e["src"],
                    rel=e["rel"],
                    dst=e["dst"],
                    weight=float(e["weight"]),
                    source=e.get("source", "team"),
                    note=e.get("note"),
                )
            )

    return KnowledgeGraph(nodes, edges)
