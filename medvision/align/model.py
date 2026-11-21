"""Image <-> knowledge-graph alignment model (PLAN.md 4.5 / T18).

Two towers meet in one L2-normalised space of size `align.embed_dim`:
  * the image tower is a small MLP over frozen DenseNet-121 pooled features;
  * the graph tower is the GATv2 GraphEncoder over the whole knowledge graph.
An image is matched to the *set* of KG finding nodes that are true for it (the "positive
graph" for that image): its embedding is the mean of those nodes' embeddings. The same node
embeddings also give open-vocabulary hints, i.e. which KG nodes a new image is closest to.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from medvision.align.graph_encoder import GraphEncoder, GraphTensors
from medvision.config import AlignConfig, get_settings
from medvision.kg.graph import KnowledgeGraph


class AlignmentModel(nn.Module):
    def __init__(self, tensors: GraphTensors, cfg: AlignConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or get_settings().align
        self.cfg = cfg
        self.node_ids = list(tensors.node_ids)
        self.graph = GraphEncoder(
            tensors, cfg.hidden_dim, cfg.embed_dim, cfg.gnn_layers, cfg.gnn_heads, cfg.dropout
        )
        self.image_head = nn.Sequential(
            nn.LayerNorm(cfg.feature_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.init_temperature)))
        self._graph_tensors = tensors
        self.register_buffer("finding_index", torch.zeros(0, dtype=torch.long))

    def set_findings(self, finding_ids: list[str]) -> None:
        """Declare which graph nodes are label-able findings (the columns of every mask)."""
        position = {n: i for i, n in enumerate(self.node_ids)}
        self.finding_ids = list(finding_ids)
        self.finding_index = torch.tensor(
            [position[f] for f in finding_ids], device=self.logit_scale.device
        )

    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def image_embeddings(self, feats: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_head(feats), dim=-1)

    def node_embeddings(self) -> torch.Tensor:
        return F.normalize(self.graph(), dim=-1)

    def finding_embeddings(self) -> torch.Tensor:
        return self.node_embeddings()[self.finding_index]

    def label_set_embeddings(self, masks: torch.Tensor) -> torch.Tensor:
        """Mean embedding of each row's true findings. `masks` is (B, n_findings) in {0, 1}."""
        raw = self.graph()[self.finding_index]
        pooled = (masks @ raw) / masks.sum(dim=1, keepdim=True).clamp(min=1.0)
        return F.normalize(pooled, dim=-1)

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "graph_tensors": self._graph_tensors.to_dict(),
                "finding_ids": self.finding_ids,
                "cfg": self.cfg.model_dump(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> AlignmentModel:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(GraphTensors.from_dict(blob["graph_tensors"]), AlignConfig(**blob["cfg"]))
        model.set_findings(blob["finding_ids"])  # sizes the finding_index buffer first
        model.load_state_dict(blob["state_dict"])
        model.extra = blob.get("extra", {})
        return model.to(device).eval()


def finding_node_ids(kg: KnowledgeGraph) -> list[str]:
    return [n.id for n in kg.nodes_of_type("Finding")]
