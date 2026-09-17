"""The core /analyze endpoint plus history, heatmap serving, and feedback (FR1, FR7, FR8)."""

from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlmodel import Session, select

from backend.app.db import AnalysisRecord, FeedbackRecord, get_session
from backend.app.schemas import AnalysisSummary, FeedbackIn, FeedbackOut
from medvision.config import get_settings
from medvision.pipeline import AnalysisResult, analyze

router = APIRouter(tags=["analyze"])


def _uploads_dir_for(result_id: str) -> Path:
    settings = get_settings()
    return settings.paths.resolve("uploads_dir") / result_id


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile, session: Session = Depends(get_session)
) -> AnalysisResult:
    settings = get_settings()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.api.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.api.allowed_extensions}",
        )
    if suffix == ".dcm":
        # DICOM de-identification + loading (medvision/io, PLAN.md T07) isn't wired up yet.
        raise HTTPException(status_code=400, detail="DICOM upload support is not yet available.")

    raw = await file.read()
    max_bytes = settings.api.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.api.max_upload_mb}MB")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not read image file") from exc

    result_id = str(uuid.uuid4())
    save_dir = _uploads_dir_for(result_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(save_dir / "original.png")

    result = analyze(image, save_dir=save_dir, result_id=result_id)

    record = AnalysisRecord(
        id=result.id,
        created_at=result.created_at,
        result_json=result.model_dump_json(),
        upload_dir=str(save_dir),
    )
    session.add(record)
    session.commit()

    return result


@router.get("/analyses", response_model=list[AnalysisSummary])
def list_analyses(
    limit: int = 20, session: Session = Depends(get_session)
) -> list[AnalysisSummary]:
    records = session.exec(
        select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).limit(limit)
    ).all()
    summaries = []
    for r in records:
        data = json.loads(r.result_json)
        top_finding = data["findings"][0]["name"] if data.get("findings") else None
        top_diff = data["differentials"][0]["name"] if data.get("differentials") else None
        summaries.append(
            AnalysisSummary(
                id=r.id, created_at=r.created_at, top_finding=top_finding, top_differential=top_diff
            )
        )
    return summaries


@router.get("/analyses/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str, session: Session = Depends(get_session)) -> AnalysisResult:
    record = session.get(AnalysisRecord, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisResult.model_validate_json(record.result_json)


@router.get("/analyses/{analysis_id}/heatmap/{node_id}.png")
def get_heatmap(analysis_id: str, node_id: str) -> FileResponse:
    path = _uploads_dir_for(analysis_id) / f"heatmap_{node_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(path, media_type="image/png")


@router.get("/analyses/{analysis_id}/original.png")
def get_original(analysis_id: str) -> FileResponse:
    path = _uploads_dir_for(analysis_id) / "original.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original image not found")
    return FileResponse(path, media_type="image/png")


@router.post("/analyses/{analysis_id}/feedback", response_model=FeedbackOut)
def submit_feedback(
    analysis_id: str, feedback: FeedbackIn, session: Session = Depends(get_session)
) -> FeedbackOut:
    if session.get(AnalysisRecord, analysis_id) is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    record = FeedbackRecord(
        analysis_id=analysis_id,
        node_id=feedback.node_id,
        verdict=feedback.verdict,
        comment=feedback.comment,
        created_at=time.time(),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return FeedbackOut(
        id=record.id,
        analysis_id=record.analysis_id,
        node_id=record.node_id,
        verdict=record.verdict,
        comment=record.comment,
        created_at=record.created_at,
    )
