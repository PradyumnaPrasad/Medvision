"""The core /analyze endpoint plus history, heatmap serving, and feedback (FR1, FR7, FR8)."""

from __future__ import annotations

import io
import json
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError
from sqlmodel import Session, select

from backend.app.db import AnalysisRecord, FeedbackRecord, get_session
from backend.app.report_pdf import render_report_pdf
from backend.app.schemas import AnalysisSummary, FeedbackIn, FeedbackOut
from medvision.config import get_settings
from medvision.io import DicomError, load_dicom
from medvision.pipeline import AnalysisResult, analyze
from medvision.qc.ood import InvalidImageError

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
    raw = await file.read()
    max_bytes = settings.api.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.api.max_upload_mb}MB")

    input_warnings: list[str] = []
    if suffix == ".dcm":
        try:
            dicom = load_dicom(raw)
        except DicomError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        image = dicom.image
        input_warnings = dicom.warnings
    else:
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Could not read image file") from exc

    result_id = str(uuid.uuid4())
    save_dir = _uploads_dir_for(result_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(save_dir / "original.png")

    try:
        result = analyze(
            image, save_dir=save_dir, result_id=result_id, extra_warnings=input_warnings
        )
    except InvalidImageError as exc:
        shutil.rmtree(save_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"This doesn't look like a usable chest X-ray: {exc}",
        ) from exc

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


def _feedback_out(record: FeedbackRecord) -> FeedbackOut:
    assert record.id is not None
    return FeedbackOut(
        id=record.id,
        analysis_id=record.analysis_id,
        node_id=record.node_id,
        verdict=record.verdict,
        comment=record.comment,
        created_at=record.created_at,
    )


def _latest_feedback(session: Session, analysis_id: str) -> list[FeedbackRecord]:
    """One record per node (the most recent), oldest first. Tolerates legacy duplicates."""
    records = session.exec(
        select(FeedbackRecord)
        .where(FeedbackRecord.analysis_id == analysis_id)
        .order_by(FeedbackRecord.created_at, FeedbackRecord.id)
    ).all()
    latest: dict[str, FeedbackRecord] = {r.node_id: r for r in records}
    return sorted(latest.values(), key=lambda r: (r.created_at, r.id or 0))


@router.post("/analyses/{analysis_id}/feedback", response_model=FeedbackOut)
def submit_feedback(
    analysis_id: str, feedback: FeedbackIn, session: Session = Depends(get_session)
) -> FeedbackOut:
    """Record a reviewer verdict on one finding or differential (one verdict per item).

    Submitting again for the same item replaces the earlier verdict rather than adding
    a second row, so the stored feedback always reflects the reviewer's latest opinion.
    """
    analysis = session.get(AnalysisRecord, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    max_chars = get_settings().api.feedback_max_comment_chars
    if feedback.comment is not None and len(feedback.comment) > max_chars:
        raise HTTPException(status_code=422, detail=f"Comment exceeds {max_chars} characters")

    result = AnalysisResult.model_validate_json(analysis.result_json)
    valid_nodes = {f.node_id for f in result.findings} | {d.node_id for d in result.differentials}
    if feedback.node_id not in valid_nodes:
        raise HTTPException(
            status_code=400, detail=f"'{feedback.node_id}' is not a finding or diagnosis here"
        )

    existing = session.exec(
        select(FeedbackRecord)
        .where(FeedbackRecord.analysis_id == analysis_id)
        .where(FeedbackRecord.node_id == feedback.node_id)
        .order_by(FeedbackRecord.created_at, FeedbackRecord.id)
    ).all()
    record = (
        existing[-1]
        if existing
        else FeedbackRecord(
            analysis_id=analysis_id,
            node_id=feedback.node_id,
            verdict=feedback.verdict,
            created_at=0,
        )
    )
    for stale in existing[:-1]:
        session.delete(stale)
    record.verdict = feedback.verdict
    record.comment = feedback.comment
    record.created_at = time.time()
    session.add(record)
    session.commit()
    session.refresh(record)
    return _feedback_out(record)


@router.get("/analyses/{analysis_id}/feedback", response_model=list[FeedbackOut])
def list_feedback(analysis_id: str, session: Session = Depends(get_session)) -> list[FeedbackOut]:
    if session.get(AnalysisRecord, analysis_id) is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return [_feedback_out(r) for r in _latest_feedback(session, analysis_id)]


@router.get("/analyses/{analysis_id}/report.pdf")
def get_report_pdf(analysis_id: str, session: Session = Depends(get_session)) -> Response:
    record = session.get(AnalysisRecord, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    result = AnalysisResult.model_validate_json(record.result_json)
    feedback = [_feedback_out(r) for r in _latest_feedback(session, analysis_id)]
    pdf = render_report_pdf(result, feedback, Path(record.upload_dir))
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="medvision-report-{analysis_id[:8]}.pdf"'
        },
    )
