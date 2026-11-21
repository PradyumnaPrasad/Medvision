"""Retrieval service (PLAN.md 4.6 / T20): turns an image's pooled features into
  * `node_hints`: the KG finding nodes the image sits closest to in the aligned space, and
  * `similar_cases`: the most similar labelled reference cases.
These are supporting context. They never change the KG-reasoned diagnosis ranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from medvision.align.model import AlignmentModel
from medvision.config import get_settings
from medvision.kg.graph import KnowledgeGraph
from medvision.retrieval.index import LIST_SEP, get_collection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeHint:
    node_id: str
    name: str
    score: float  # cosine similarity in the aligned space


@dataclass(frozen=True)
class SimilarCase:
    case_id: str
    similarity: float
    labels: list[str]
    finding_node_ids: list[str]
    split: str


@dataclass(frozen=True)
class RetrievalResult:
    node_hints: list[NodeHint] = field(default_factory=list)
    similar_cases: list[SimilarCase] = field(default_factory=list)


def artifacts_present(chroma_dir: Path, checkpoint: Path) -> bool:
    return checkpoint.exists() and chroma_dir.exists() and any(chroma_dir.iterdir())


class RetrievalService:
    def __init__(self, model: AlignmentModel, kg: KnowledgeGraph, chroma_dir: Path) -> None:
        cfg = get_settings()
        self.model = model.cpu().eval()
        self.kg = kg
        self.top_k_nodes = cfg.retrieval.top_k_nodes
        self.top_k_cases = cfg.retrieval.top_k_cases
        self.nodes = get_collection(chroma_dir, cfg.align.chroma_collection_nodes)
        self.cases = get_collection(chroma_dir, cfg.align.chroma_collection_cases)

    @classmethod
    def load_if_available(cls, kg: KnowledgeGraph) -> RetrievalService | None:
        """Return a service, or None when the alignment artifacts haven't been built yet.
        The rest of the pipeline works without it."""
        settings = get_settings()
        checkpoint = settings.paths.resolve("models_dir") / settings.align.checkpoint_file
        chroma_dir = settings.paths.resolve("chroma_dir")
        if not artifacts_present(chroma_dir, checkpoint):
            logger.info("Alignment artifacts not found; similar-case retrieval is disabled.")
            return None
        try:
            return cls(AlignmentModel.load(checkpoint), kg, chroma_dir)
        except Exception:  # corrupt/stale index must not take the whole API down
            logger.exception("Could not load alignment artifacts; retrieval disabled.")
            return None

    @torch.no_grad()
    def search(self, features: np.ndarray) -> RetrievalResult:
        embedding = self.model.image_embeddings(torch.tensor(features[None], dtype=torch.float32))
        query = embedding.numpy().tolist()

        node_res = self.nodes.query(query_embeddings=query, n_results=self.top_k_nodes)
        hints = [
            NodeHint(node_id=node_id, name=meta["name"], score=1.0 - float(dist))
            for node_id, meta, dist in zip(
                node_res["ids"][0], node_res["metadatas"][0], node_res["distances"][0], strict=True
            )
        ]
        case_res = self.cases.query(query_embeddings=query, n_results=self.top_k_cases)
        cases = [
            SimilarCase(
                case_id=case_id,
                similarity=1.0 - float(dist),
                labels=[x for x in meta["labels"].split(LIST_SEP) if x],
                finding_node_ids=[x for x in meta["findings"].split(LIST_SEP) if x],
                split=meta["split"],
            )
            for case_id, meta, dist in zip(
                case_res["ids"][0], case_res["metadatas"][0], case_res["distances"][0], strict=True
            )
        ]
        return RetrievalResult(node_hints=hints, similar_cases=cases)
