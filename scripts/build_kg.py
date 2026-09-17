#!/usr/bin/env python3
"""Build and validate the MEDVISION knowledge graph from kg/ontology/, writing
data/kg/graph.json. Run this after any edit to kg/ontology/*.yaml.

Usage: uv run python scripts/build_kg.py
"""

from __future__ import annotations

import sys

from medvision.config import get_labels_map, get_settings
from medvision.kg.build import KnowledgeGraphBuildError, build_graph, save_json


def main() -> int:
    settings = get_settings()
    ontology_dir = settings.paths.resolve("kg_ontology_dir")
    out_path = settings.paths.resolve("kg_dir") / "graph.json"
    labels_map = get_labels_map()

    print(f"Loading ontology from {ontology_dir} ...")
    try:
        graph = build_graph(ontology_dir, labels_map)
    except KnowledgeGraphBuildError as exc:
        print(f"FAILED:\n{exc}", file=sys.stderr)
        return 1

    summary = graph.summary()
    print(f"Valid. {summary}")
    save_json(graph, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
