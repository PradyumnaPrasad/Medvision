"""API-only request/response schemas that don't belong in medvision's pure ML core.

AnalysisResult itself (the response body of POST /analyze) is defined in
medvision/pipeline.py since it's the pipeline's own output contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FeedbackIn(BaseModel):
    node_id: str
    verdict: Literal["agree", "disagree"]
    comment: str | None = None


class FeedbackOut(BaseModel):
    id: int
    analysis_id: str
    node_id: str
    verdict: str
    comment: str | None
    created_at: float


class AnalysisSummary(BaseModel):
    id: str
    created_at: float
    top_finding: str | None
    top_differential: str | None


class HealthOut(BaseModel):
    status: Literal["ok"]
    device: str
    kg_nodes: int
    kg_edges: int
    disclaimer: str
