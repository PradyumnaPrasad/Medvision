"""Render a stored AnalysisResult (plus clinician feedback) as a downloadable PDF report.

Uses reportlab's built-in Helvetica, which only covers the Windows-1252 character set:
anything outside it renders as a solid black box, so every string goes through `_safe`
first. LLM-written text is untrusted for ReportLab's mini-markup too, hence the escaping.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.app.schemas import FeedbackOut
from medvision.config import get_settings
from medvision.pipeline import AnalysisResult, FindingOut

INK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
RULE = colors.HexColor("#e2e8f0")
ACCENT = colors.HexColor("#4f46e5")
WARN_BG = colors.HexColor("#fef3c7")
WARN_INK = colors.HexColor("#92400e")
HEAD_BG = colors.HexColor("#f1f5f9")
GOOD = colors.HexColor("#047857")
BAD = colors.HexColor("#be123c")

_ASCII_FALLBACKS = str.maketrans(
    {
        "≥": ">=",
        "≤": "<=",
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "≈": "~",
        "−": "-",
        "‑": "-",
        " ": " ",
    }
)

_PAGE_MARGIN = 1.8 * cm
_TOP_MARGIN = 2.2 * cm


def _safe(text: str) -> str:
    """Make text safe for a Paragraph: Helvetica-encodable and markup-escaped."""
    folded = text.translate(_ASCII_FALLBACKS).encode("cp1252", errors="replace").decode("cp1252")
    return escape(folded)


def _region_name(region: str | None) -> str:
    if not region:
        return "-"
    return " ".join(w.capitalize() for w in region.removeprefix("A_").split("_"))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle(
        "body", parent=base, fontName="Helvetica", fontSize=9.5, leading=13.5, spaceAfter=3
    )
    return {
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8, leading=11, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11),
        "title": ParagraphStyle(
            "title", parent=body, fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=INK
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=ACCENT,
            spaceBefore=12,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption", parent=body, fontSize=7.5, leading=10, textColor=MUTED, alignment=TA_CENTER
        ),
        "warn": ParagraphStyle("warn", parent=body, fontSize=8.5, leading=11.5, textColor=WARN_INK),
    }


def _cite(ids: Sequence[str]) -> str:
    return f' <font size="7.5" color="#64748b">[{_safe(", ".join(ids))}]</font>' if ids else ""


def _scaled_image(path: Path, max_w: float, max_h: float) -> Image | None:
    try:
        with PILImage.open(path) as im:
            w, h = im.size
    except (OSError, ValueError):
        return None
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def _inside(base: Path, name: str) -> Path | None:
    path = (base / name).resolve()
    return path if path.is_relative_to(base.resolve()) and path.is_file() else None


def _image_row(result: AnalysisResult, upload_dir: Path, styles: dict) -> Table | None:
    cfg = get_settings().api
    shown = [f for f in result.findings if f.status != "absent" and f.heatmap_path]
    tiles: list[tuple[Image, str]] = []
    max_w, max_h = 4.1 * cm, 4.6 * cm

    original = _inside(upload_dir, "original.png")
    if original and (img := _scaled_image(original, max_w, max_h)):
        tiles.append((img, "Uploaded image (de-identified)"))
    for f in shown[: cfg.pdf_max_heatmaps]:
        path = _inside(upload_dir, f.heatmap_path or "")
        if path and (img := _scaled_image(path, max_w, max_h)):
            tiles.append((img, f"Grad-CAM: {f.name}"))
    if not tiles:
        return None

    table = Table(
        [
            [img for img, _ in tiles],
            [Paragraph(_safe(caption), styles["caption"]) for _, caption in tiles],
        ],
        colWidths=[4.3 * cm] * len(tiles),
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, 0), "MIDDLE")])
    )
    return table


def _verdict_cell(verdict: str | None, styles: dict) -> Paragraph:
    if verdict is None:
        return Paragraph("-", styles["cell"])
    color = "#047857" if verdict == "agree" else "#be123c"
    return Paragraph(
        f'<font color="{color}"><b>{_safe(verdict.capitalize())}</b></font>', styles["cell"]
    )


def _grid(rows: list[list], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _header_text(label: str, styles: dict) -> Paragraph:
    return Paragraph(f"<b>{label}</b>", styles["cell"])


def _findings_table(findings: list[FindingOut], verdicts: dict[str, str], styles: dict) -> list:
    shown = [f for f in findings if f.status != "absent"]
    if not shown:
        return [Paragraph("No findings met the reporting threshold.", styles["body"])]
    rows = [
        [
            _header_text(h, styles)
            for h in ("Finding", "Probability", "Status", "Region", "Reviewer")
        ]
    ]
    for f in shown:
        rows.append(
            [
                Paragraph(_safe(f.name), styles["cell"]),
                Paragraph(f"{f.prob * 100:.0f}%", styles["cell"]),
                Paragraph(_safe(f.status.capitalize()), styles["cell"]),
                Paragraph(_safe(_region_name(f.region)), styles["cell"]),
                _verdict_cell(verdicts.get(f.node_id), styles),
            ]
        )
    out: list = [_grid(rows, [5.6 * cm, 2.4 * cm, 2.4 * cm, 4.0 * cm, 2.8 * cm])]
    hidden = len(findings) - len(shown)
    if hidden:
        out.append(
            Paragraph(
                f"{hidden} further finding(s) were below the reporting threshold.", styles["small"]
            )
        )
    return out


def _differentials_block(result: AnalysisResult, verdicts: dict[str, str], styles: dict) -> list:
    if not result.differentials:
        return [
            Paragraph("No differential diagnosis reached the activation threshold.", styles["body"])
        ]
    out: list = []
    for i, d in enumerate(result.differentials, 1):
        reviewer = verdicts.get(d.node_id)
        suffix = f"  -  reviewer: {_safe(reviewer)}" if reviewer else ""
        lines = [
            Paragraph(
                f"<b>{i}. {_safe(d.name)}</b>  -  score {d.score * 100:.0f}%{suffix}",
                styles["body"],
            )
        ]
        for p in d.paths:
            relation = "contradicts" if p.relation == "contradicts" else "suggestive of"
            lines.append(
                Paragraph(
                    f"&nbsp;&nbsp;&nbsp;{_safe(p.finding_name)} {p.finding_prob * 100:.0f}% "
                    f"{relation} (w={p.weight:.2f}) &nbsp;<font color='#64748b'>"
                    f"{p.contribution:+.3f}</font>",
                    styles["small"],
                )
            )
        lines.append(Spacer(1, 4))
        out.append(KeepTogether(lines))
    return out


def _retrieval_block(result: AnalysisResult, styles: dict) -> list:
    retrieval = result.retrieval
    if retrieval is None:
        return []
    out: list = [Paragraph("Similar reference cases (aligned image-graph space)", styles["h2"])]
    if retrieval.similar_cases:
        rows = [[_header_text(h, styles) for h in ("Case", "Labels", "Similarity")]]
        for case in retrieval.similar_cases:
            rows.append(
                [
                    Paragraph(_safe(case.case_id), styles["cell"]),
                    Paragraph(_safe(", ".join(case.labels) or "-"), styles["cell"]),
                    Paragraph(f"{case.similarity * 100:.0f}%", styles["cell"]),
                ]
            )
        out.append(_grid(rows, [5.2 * cm, 9.0 * cm, 3.0 * cm]))
    if retrieval.node_hints:
        hints = ", ".join(f"{_safe(h.name)} ({h.score * 100:.0f})" for h in retrieval.node_hints)
        out.append(Spacer(1, 3))
        out.append(Paragraph(f"<b>Closest knowledge-graph nodes:</b> {hints}", styles["small"]))
    out.append(
        Paragraph(
            "Supporting context only; it does not change the diagnosis ranking. Reference "
            "cases are a public NIH ChestX-ray14 sample.",
            styles["small"],
        )
    )
    return out


def _report_block(result: AnalysisResult, styles: dict) -> list:
    report = result.report
    out: list = []

    def section(title: str, items: list) -> None:
        out.append(Paragraph(title, styles["h2"]))
        out.extend(items)

    section(
        "Findings",
        [Paragraph(_safe(i.text) + _cite(i.evidence_ids), styles["body"]) for i in report.findings]
        or [Paragraph("No findings met the reporting threshold.", styles["body"])],
    )
    section(
        "Impression",
        [
            Paragraph(_safe(i.text) + _cite(i.evidence_ids), styles["body"])
            for i in report.impression
        ]
        or [Paragraph("No impression could be generated.", styles["body"])],
    )
    if report.differentials:
        section(
            "Differential diagnoses",
            [
                Paragraph(
                    f"<b>{_safe(d.diagnosis)}:</b> {_safe(d.rationale)}{_cite(d.evidence_ids)}",
                    styles["body"],
                )
                for d in report.differentials
            ],
        )
    if report.limitations:
        section(
            "Limitations",
            [Paragraph("- " + _safe(text), styles["body"]) for text in report.limitations],
        )
    if report.recommendation:
        section("Recommendation", [Paragraph(_safe(report.recommendation), styles["body"])])
    return out


def _grounding_block(result: AnalysisResult, styles: dict) -> list:
    g = result.grounding
    source = (
        f"LLM-narrated ({get_settings().llm.model}), checked by a citation verifier"
        if result.report_source == "llm"
        else "Deterministic template (no LLM was used)"
    )
    out = [
        Paragraph(
            f"<b>Report source:</b> {_safe(source)}. <b>Grounding score:</b> {g.score * 100:.0f}% "
            "(share of statements whose citations and claims check out against the evidence).",
            styles["small"],
        )
    ]
    for issue in [*g.unsupported, *g.contradictions]:
        out.append(
            Paragraph(f'<font color="#be123c">Verifier: {_safe(issue)}</font>', styles["small"])
        )
    return out


def _evidence_legend(result: AnalysisResult, styles: dict) -> list:
    ev = result.evidence
    rows = [[_header_text("ID", styles), _header_text("Evidence", styles)]]
    for f in ev.findings:
        region = f", {_region_name(f.region)}" if f.region else ""
        rows.append(
            [
                Paragraph(_safe(f.id), styles["cell"]),
                Paragraph(
                    f"{_safe(f.name)}: {f.prob * 100:.0f}% ({_safe(f.status)}{_safe(region)})",
                    styles["cell"],
                ),
            ]
        )
    for d in ev.differentials:
        rows.append(
            [
                Paragraph(_safe(d.id), styles["cell"]),
                Paragraph(f"{_safe(d.name)}: score {d.score * 100:.0f}%", styles["cell"]),
            ]
        )
    for k in ev.kg_facts:
        rows.append(
            [Paragraph(_safe(k.id), styles["cell"]), Paragraph(_safe(k.text), styles["cell"])]
        )
    for c in ev.cases:
        rows.append(
            [Paragraph(_safe(c.id), styles["cell"]), Paragraph(_safe(c.text), styles["cell"])]
        )
    return [_grid(rows, [1.6 * cm, 15.6 * cm])] if len(rows) > 1 else []


def _feedback_block(result: AnalysisResult, feedback: list[FeedbackOut], styles: dict) -> list:
    if not feedback:
        return [
            Paragraph("No reviewer feedback has been recorded for this analysis.", styles["body"])
        ]
    names = {f.node_id: f.name for f in result.findings}
    names.update({d.node_id: d.name for d in result.differentials})
    rows = [[_header_text(h, styles) for h in ("Item", "Verdict", "Comment")]]
    for fb in feedback:
        rows.append(
            [
                Paragraph(_safe(names.get(fb.node_id, fb.node_id)), styles["cell"]),
                _verdict_cell(fb.verdict, styles),
                Paragraph(_safe(fb.comment or "-"), styles["cell"]),
            ]
        )
    return [_grid(rows, [5.6 * cm, 2.4 * cm, 9.2 * cm])]


def _on_page(disclaimer: str, analysis_id: str):
    def draw(canvas, doc) -> None:
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(WARN_BG)
        canvas.rect(0, height - 1.15 * cm, width, 1.15 * cm, stroke=0, fill=1)
        canvas.setFillColor(WARN_INK)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawCentredString(width / 2, height - 0.7 * cm, _clip(disclaimer, 130))
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(_PAGE_MARGIN, 1.0 * cm, f"MEDVISION analysis {analysis_id}")
        canvas.drawRightString(width - _PAGE_MARGIN, 1.0 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def _clip(text: str, limit: int) -> str:
    safe = text.translate(_ASCII_FALLBACKS).encode("cp1252", errors="replace").decode("cp1252")
    return safe if len(safe) <= limit else safe[: limit - 3] + "..."


def render_report_pdf(
    result: AnalysisResult, feedback: list[FeedbackOut], upload_dir: Path
) -> bytes:
    """Return the PDF bytes for one analysis. Missing image files are skipped, not fatal."""
    styles = _styles()
    settings = get_settings()
    verdicts = {fb.node_id: fb.verdict for fb in feedback}
    created = datetime.fromtimestamp(result.created_at, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    top = result.differentials[0] if result.differentials else None
    shown = [f for f in result.findings if f.status != "absent"]

    story: list = [
        Paragraph("MEDVISION - Chest X-ray Analysis Report", styles["title"]),
        Spacer(1, 3),
        Paragraph(
            f"Analysis <b>{_safe(result.id)}</b> &nbsp;|&nbsp; analysed {created}", styles["small"]
        ),
        Spacer(1, 8),
    ]
    if result.quality_warnings:
        warn = Table(
            [
                [
                    Paragraph(
                        "<b>Input warnings:</b> " + _safe("; ".join(result.quality_warnings)),
                        styles["warn"],
                    )
                ]
            ],
            colWidths=[17.2 * cm],
        )
        warn.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
                    ("BOX", (0, 0), (-1, -1), 0.5, WARN_INK),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story += [warn, Spacer(1, 6)]

    if top:
        summary = (
            f"Findings are most consistent with <b>{_safe(top.name)}</b> "
            f"({top.score * 100:.0f}% score), based on {len(shown)} finding(s) above the "
            "reporting threshold."
        )
    else:
        summary = "No differential diagnosis reached the activation threshold for this image."
    story.append(Paragraph(summary, styles["body"]))

    if images := _image_row(result, upload_dir, styles):
        story += [Spacer(1, 8), images]

    story.append(Paragraph("Findings", styles["h2"]))
    story += _findings_table(result.findings, verdicts, styles)
    story.append(Paragraph("Differential diagnoses (knowledge-graph reasoning)", styles["h2"]))
    story += _differentials_block(result, verdicts, styles)
    story += _retrieval_block(result, styles)
    story.append(Paragraph("Draft report", styles["h2"]))
    story += _grounding_block(result, styles)
    story += _report_block(result, styles)
    story.append(Paragraph("Evidence referenced by the report", styles["h2"]))
    story += _evidence_legend(result, styles)
    story.append(Paragraph("Reviewer feedback", styles["h2"]))
    story += _feedback_block(result, feedback, styles)
    story += [
        Spacer(1, 12),
        Paragraph(
            "System: "
            f"{_safe(settings.vision.classifier)} / {_safe(settings.vision.weights)} "
            "classifier with "
            "per-finding calibration where fitted; Grad-CAM localisation; deterministic "
            "knowledge-graph differential scoring. The language model only narrates this evidence.",
            styles["small"],
        ),
        Paragraph(_safe(result.disclaimer), styles["small"]),
    ]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_PAGE_MARGIN,
        rightMargin=_PAGE_MARGIN,
        topMargin=_TOP_MARGIN,
        bottomMargin=1.8 * cm,
        title=f"MEDVISION report {result.id}",
        author="MEDVISION (research prototype)",
    )
    on_page = _on_page(result.disclaimer, result.id)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue()
