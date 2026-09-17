"""Read-only knowledge-graph exploration endpoints, used by the UI's KG path viewer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from medvision.pipeline import get_models

router = APIRouter(tags=["kg"], prefix="/kg")


class KgNodeOut(BaseModel):
    id: str
    type: str
    name: str
    definition: str
    aliases: list[str]


class KgEdgeOut(BaseModel):
    src: str
    rel: str
    dst: str
    weight: float
    note: str | None


class KgNodeDetail(BaseModel):
    node: KgNodeOut
    outgoing: list[KgEdgeOut]
    incoming: list[KgEdgeOut]


def _node_out(node) -> KgNodeOut:
    return KgNodeOut(
        id=node.id, type=node.type, name=node.name, definition=node.definition, aliases=node.aliases
    )


def _edge_out(edge) -> KgEdgeOut:
    return KgEdgeOut(src=edge.src, rel=edge.rel, dst=edge.dst, weight=edge.weight, note=edge.note)


@router.get("/nodes/{node_id}", response_model=KgNodeDetail)
def get_node(node_id: str) -> KgNodeDetail:
    graph = get_models().graph
    node = graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return KgNodeDetail(
        node=_node_out(node),
        outgoing=[_edge_out(e) for e in graph.out_edges(node_id)],
        incoming=[_edge_out(e) for e in graph.in_edges(node_id)],
    )


@router.get("/subgraph", response_model=list[KgEdgeOut])
def get_subgraph(ids: str) -> list[KgEdgeOut]:
    """Return every edge touching any of the comma-separated node ids (for path visualization)."""
    graph = get_models().graph
    node_ids = set(ids.split(","))
    edges = [e for e in graph.edges if e.src in node_ids or e.dst in node_ids]
    return [_edge_out(e) for e in edges]
