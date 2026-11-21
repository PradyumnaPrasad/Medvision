"""Build the ChromaDB index (KG nodes + reference cases) from the trained alignment model.
Idempotent: collections are dropped and recreated, so re-running gives an identical index.

Usage: uv run python scripts/index_chroma.py   (after scripts/train_alignment.py)
"""

from __future__ import annotations

import logging

import numpy as np

from medvision.align.data import load_feature_sets
from medvision.align.evaluate import embed_images
from medvision.align.model import AlignmentModel
from medvision.config import get_labels_map, get_settings
from medvision.kg.build import build_graph
from medvision.retrieval.index import rebuild_case_index, rebuild_node_index

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("index_chroma")


def main() -> None:
    settings = get_settings()
    cfg = settings.align
    labels_map = get_labels_map()
    kg = build_graph(settings.paths.resolve("kg_ontology_dir"), labels_map)
    checkpoint = settings.paths.resolve("models_dir") / cfg.checkpoint_file
    model = AlignmentModel.load(checkpoint)
    version = f"{checkpoint.stat().st_mtime_ns}"
    chroma_dir = settings.paths.resolve("chroma_dir")

    data = load_feature_sets(
        settings.paths.resolve("processed_dir") / cfg.features_file, model.finding_ids, labels_map
    )
    # Only findings that had at least one positive training image were ever aligned with
    # images. The rest (e.g. Fracture, Lung Opacity: not NIH labels) would be suggested on
    # the strength of graph position alone, so they are left out of the node-hint index.
    trained = data["train"].masks.sum(axis=0) > 0
    node_rows = [i for i, ok in enumerate(trained) if ok]
    node_embeddings = model.finding_embeddings().detach().numpy()[node_rows]
    node_ids = [model.finding_ids[i] for i in node_rows]
    n_nodes = rebuild_node_index(
        chroma_dir,
        cfg.chroma_collection_nodes,
        node_ids,
        [kg.get_node(n).name for n in node_ids],
        node_embeddings,
        version,
    )

    case_ids, embeddings, label_names, finding_lists, splits = [], [], [], [], []
    for split, subset in data.items():
        embeddings.append(embed_images(model, subset.feats))
        case_ids += subset.case_ids
        label_names += subset.label_names
        finding_lists += [[model.finding_ids[j] for j in row.nonzero()[0]] for row in subset.masks]
        splits += [split] * len(subset)
    n_cases = rebuild_case_index(
        chroma_dir,
        cfg.chroma_collection_cases,
        case_ids,
        np.concatenate(embeddings),
        label_names,
        finding_lists,
        splits,
        version,
    )
    logger.info("Indexed %d KG nodes and %d reference cases in %s", n_nodes, n_cases, chroma_dir)


if __name__ == "__main__":
    main()
