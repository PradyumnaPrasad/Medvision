"""SQLite persistence for analysis history (PLAN.md FR8: log inference metadata).

One row per /analyze call: the full AnalysisResult is stored as JSON (simplest schema
that satisfies "log for experimental evaluation and error analysis" without over-
normalizing data whose shape is still evolving through the ticket backlog).
"""

from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Field, Session, SQLModel, create_engine

from medvision.config import get_settings


class AnalysisRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: float
    result_json: str
    upload_dir: str


class FeedbackRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    analysis_id: str = Field(index=True)
    node_id: str
    verdict: str  # "agree" | "disagree"
    comment: str | None = None
    created_at: float


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_path = settings.paths.resolve("data_dir") / "medvision.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(_engine)
    return _engine


def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session
