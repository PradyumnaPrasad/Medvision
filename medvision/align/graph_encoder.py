"""Graph encoder over the MEDVISION knowledge graph (PLAN.md 4.5 / T17).

A small GATv2 network (PyTorch Geometric) turns each KG node into an embedding. Node inputs
are the node's own text (TF-IDF over name, aliases and definition, so clinically related
nodes start close together) plus a learned id and type embedding. Every edge carries its
relation type (one-hot, forward and reverse as separate types) and its weight, so the
hand-set edge weights actually influence the embeddings.

The graph tensors are stored as buffers, so a saved checkpoint is self-contained: inference
does not need to refit the TF-IDF vocabulary or rebuild the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.nn import GATv2Conv

from medvision.kg.graph import VALID_NODE_TYPES, KnowledgeGraph

_NODE_TYPES = sorted(VALID_NODE_TYPES)


@dataclass(frozen=True)
class GraphTensors:
    node_ids: list[str]
    node_types: torch.Tensor  # (N,) long
    text: torch.Tensor  # (N, V) float TF-IDF
    edge_index: torch.Tensor  # (2, E) long, forward and reverse edges
    edge_attr: torch.Tensor  # (E, 2R + 1) float: relation one-hot + weight

    def to_dict(self) -> dict:
        return {
            "node_ids": self.node_ids,
            "node_types": self.node_types,
            "text": self.text,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GraphTensors:
        return cls(**data)


def build_graph_tensors(kg: KnowledgeGraph, relations: list[str] | None = None) -> GraphTensors:
    from medvision.kg.graph import VALID_RELATIONS

    relations = relations or sorted(VALID_RELATIONS)
    rel_index = {rel: i for i, rel in enumerate(relations)}
    node_ids = list(kg.nodes_by_id)
    position = {node_id: i for i, node_id in enumerate(node_ids)}

    texts = []
    for node_id in node_ids:
        node = kg.nodes_by_id[node_id]
        texts.append(f"{node.name}. {' '.join(node.aliases)}. {node.definition}")
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(texts)

    num_rel = len(relations)
    src, dst, attrs = [], [], []
    for edge in kg.edges:
        if edge.src not in position or edge.dst not in position:
            continue
        for direction, (a, b) in enumerate(((edge.src, edge.dst), (edge.dst, edge.src))):
            attr = torch.zeros(2 * num_rel + 1)
            attr[direction * num_rel + rel_index[edge.rel]] = 1.0
            attr[-1] = edge.weight
            src.append(position[a])
            dst.append(position[b])
            attrs.append(attr)

    return GraphTensors(
        node_ids=node_ids,
        node_types=torch.tensor([_NODE_TYPES.index(kg.nodes_by_id[n].type) for n in node_ids]),
        text=torch.tensor(tfidf.toarray(), dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.stack(attrs),
    )


class GraphEncoder(nn.Module):
    def __init__(
        self,
        tensors: GraphTensors,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        n_nodes, text_dim = tensors.text.shape
        self.register_buffer("node_types", tensors.node_types)
        self.register_buffer("text", tensors.text)
        self.register_buffer("edge_index", tensors.edge_index)
        self.register_buffer("edge_attr", tensors.edge_attr)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.id_embedding = nn.Embedding(n_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(len(_NODE_TYPES), hidden_dim)
        edge_dim = tensors.edge_attr.shape[1]
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=True,
                fill_value=1.0,
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self) -> torch.Tensor:
        ids = torch.arange(self.text.shape[0], device=self.text.device)
        h = (
            self.text_proj(self.text)
            + self.id_embedding(ids)
            + self.type_embedding(self.node_types)
        )
        for conv, norm in zip(self.convs, self.norms, strict=True):
            update = conv(h, self.edge_index, self.edge_attr)
            h = norm(h + self.dropout(torch.nn.functional.gelu(update)))
        return self.out(h)
