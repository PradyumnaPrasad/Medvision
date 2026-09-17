"""Validation rules for the MEDVISION knowledge graph (see kg/schema.md).

Run via scripts/build_kg.py before any generated artefact (graph.json/graph.pt) is
written, and in CI, so a broken ontology edit never silently reaches the app.
"""

from __future__ import annotations

from medvision.kg.graph import VALID_NODE_TYPES, VALID_RELATIONS, KnowledgeGraph


def validate(graph: KnowledgeGraph, labels_map: dict[str, str] | None = None) -> list[str]:
    """Return a list of human-readable error strings. Empty list means the graph is valid."""
    errors: list[str] = []

    # Rule 1: unique ids (guaranteed by dict keying in KnowledgeGraph, but check the raw count
    # in case two nodes with the same id silently overwrote each other during load).
    node_ids = list(graph.nodes_by_id.keys())
    if len(node_ids) != len(set(node_ids)):
        errors.append("Duplicate node ids detected.")

    # Rule 4 (checked early so later messages are meaningful): every rel is a known type.
    for edge in graph.edges:
        if edge.rel not in VALID_RELATIONS:
            errors.append(
                f"Edge {edge.src} -{edge.rel}-> {edge.dst}: unknown relation '{edge.rel}'."
            )

    # node types are sane
    for node in graph.nodes_by_id.values():
        if node.type not in VALID_NODE_TYPES:
            errors.append(f"Node {node.id}: unknown type '{node.type}'.")
        if not node.definition:
            errors.append(f"Node {node.id}: missing definition.")

    # Rule 2: every edge references an existing node.
    for edge in graph.edges:
        if edge.src not in graph.nodes_by_id:
            errors.append(f"Edge references unknown src node '{edge.src}'.")
        if edge.dst not in graph.nodes_by_id:
            errors.append(f"Edge references unknown dst node '{edge.dst}'.")

    # Rule 3: weight in [0, 1].
    for edge in graph.edges:
        if not (0.0 <= edge.weight <= 1.0):
            errors.append(
                f"Edge {edge.src} -{edge.rel}-> {edge.dst}: weight {edge.weight} out of [0,1]."
            )

    # Rule 5: every labels.yaml target exists as a Finding node.
    if labels_map:
        finding_ids = {n.id for n in graph.nodes_of_type("Finding")}
        for label, node_id in labels_map.items():
            if node_id not in finding_ids:
                errors.append(
                    f"labels.yaml maps '{label}' -> '{node_id}', but no such Finding node exists."
                )

    # Rule 6: every Disease is reachable from at least one Finding via suggestive_of
    # (directly, or transitively through differential_of/complication_of chains).
    disease_ids = {n.id for n in graph.nodes_of_type("Disease")}
    reachable: set[str] = set()
    frontier = [e.dst for e in graph.edges if e.rel == "suggestive_of"]
    reachable.update(frontier)
    chain_rels = {"differential_of", "complication_of"}
    changed = True
    while changed:
        changed = False
        for edge in graph.edges:
            if edge.rel in chain_rels and edge.src in reachable and edge.dst not in reachable:
                reachable.add(edge.dst)
                changed = True
    unreachable = disease_ids - reachable
    for d in sorted(unreachable):
        errors.append(f"Disease node '{d}' is not reachable from any Finding via suggestive_of.")

    return errors
