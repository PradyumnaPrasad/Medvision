"""ChromaDB index over the aligned embedding space (PLAN.md 4.6 / T19).

Two collections, both cosine-space:
  * kg_nodes: the KG Finding nodes (embeddings from the graph tower);
  * cases: reference images (embeddings from the image tower) with their labels as metadata.
Only embeddings and label text are stored; no image is written to the index.

Rebuilding drops and recreates a collection, so running the indexer twice gives the same
index (idempotent) and a retrained model never leaves stale vectors behind.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np

LIST_SEP = "|"


def _client(chroma_dir: Path) -> chromadb.api.ClientAPI:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def _rebuild(
    chroma_dir: Path,
    name: str,
    ids: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict],
    version: str,
) -> int:
    client = _client(chroma_dir)
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)
    collection = client.create_collection(
        name, metadata={"hnsw:space": "cosine", "version": version}
    )
    batch = 4000
    for start in range(0, len(ids), batch):
        sl = slice(start, start + batch)
        collection.add(
            ids=ids[sl],
            embeddings=embeddings[sl].astype(np.float32).tolist(),
            metadatas=metadatas[sl],
        )
    return collection.count()


def rebuild_node_index(
    chroma_dir: Path,
    name: str,
    node_ids: list[str],
    node_names: list[str],
    embeddings: np.ndarray,
    version: str,
) -> int:
    metadatas = [{"name": n, "version": version} for n in node_names]
    return _rebuild(chroma_dir, name, node_ids, embeddings, metadatas, version)


def rebuild_case_index(
    chroma_dir: Path,
    name: str,
    case_ids: list[str],
    embeddings: np.ndarray,
    label_names: list[list[str]],
    finding_node_ids: list[list[str]],
    splits: list[str],
    version: str,
) -> int:
    metadatas = [
        {
            "labels": LIST_SEP.join(labels),
            "findings": LIST_SEP.join(nodes),
            "split": split,
            "version": version,
        }
        for labels, nodes, split in zip(label_names, finding_node_ids, splits, strict=True)
    ]
    return _rebuild(chroma_dir, name, case_ids, embeddings, metadatas, version)


def get_collection(chroma_dir: Path, name: str):
    return _client(chroma_dir).get_collection(name)
