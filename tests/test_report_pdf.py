import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

from backend.app.report_pdf import render_report_pdf
from backend.app.schemas import FeedbackOut
from medvision.pipeline import AnalysisResult, DifferentialOut, FindingOut, PathStepOut
from medvision.reasoning.schema import (
    DifferentialItem,
    EvidenceBlock,
    EvidenceDifferential,
    EvidenceFinding,
    EvidenceKgFact,
    GroundingResult,
    LLMReport,
    ReportItem,
)

DISCLAIMER = "Research prototype for a college major project. Not for clinical use."


def make_result(**overrides) -> AnalysisResult:
    findings = [
        FindingOut(
            node_id="F_EFFUSION",
            label="Effusion",
            name="Pleural Effusion",
            prob=0.82,
            raw_prob=0.7,
            status="present",
            region="A_RIGHT_LOWER_ZONE",
            region_share=0.6,
            heatmap_path="heatmap_F_EFFUSION.png",
        ),
        FindingOut(
            node_id="F_NODULE",
            label="Nodule",
            name="Pulmonary Nodule",
            prob=0.4,
            raw_prob=0.4,
            status="possible",
            region="A_LEFT_UPPER_ZONE",
            region_share=0.5,
            heatmap_path=None,
        ),
        FindingOut(
            node_id="F_MASS",
            label="Mass",
            name="Pulmonary Mass",
            prob=0.05,
            raw_prob=0.05,
            status="absent",
        ),
    ]
    differentials = [
        DifferentialOut(
            node_id="D_CHF",
            name="Congestive Heart Failure",
            score=1.0,
            paths=[
                PathStepOut(
                    finding_node_id="F_EFFUSION",
                    finding_name="Pleural Effusion",
                    finding_prob=0.82,
                    relation="suggestive_of",
                    weight=0.6,
                    contribution=0.492,
                )
            ],
        )
    ]
    report = LLMReport(
        findings=[
            ReportItem(
                text="Pleural Effusion is present in the right lower zone.", evidence_ids=["F1"]
            )
        ],
        impression=[
            ReportItem(text="Findings suggest congestive heart failure.", evidence_ids=["D1"])
        ],
        differentials=[
            DifferentialItem(
                diagnosis="Congestive Heart Failure",
                rationale="Effusion is suggestive of CHF.",
                evidence_ids=["D1", "K1"],
            )
        ],
        limitations=["Single frontal view."],
        recommendation="Correlate clinically.",
    )
    evidence = EvidenceBlock(
        findings=[
            EvidenceFinding(
                id="F1",
                node_id="F_EFFUSION",
                name="Pleural Effusion",
                prob=0.82,
                status="present",
                region="A_RIGHT_LOWER_ZONE",
            )
        ],
        differentials=[
            EvidenceDifferential(
                id="D1",
                node_id="D_CHF",
                name="Congestive Heart Failure",
                score=1.0,
                supporting_fact_ids=["K1"],
            )
        ],
        kg_facts=[EvidenceKgFact(id="K1", text="Pleural Effusion suggestive_of CHF (w=0.60)")],
    )
    base = dict(
        id="11111111-2222-3333-4444-555555555555",
        created_at=1_760_000_000.0,
        findings=findings,
        differentials=differentials,
        report=report,
        report_source="template",
        grounding=GroundingResult(score=1.0, unsupported=[], contradictions=[]),
        evidence=evidence,
        report_text="",
        quality_warnings=[],
        timings_ms={},
        disclaimer=DISCLAIMER,
    )
    base.update(overrides)
    return AnalysisResult(**base)


def fb(node_id: str, verdict: str, comment: str | None = None) -> FeedbackOut:
    return FeedbackOut(
        id=1, analysis_id="x", node_id=node_id, verdict=verdict, comment=comment, created_at=1.0
    )


def pdf_text(pdf: bytes) -> tuple[PdfReader, list[str]]:
    reader = PdfReader(io.BytesIO(pdf))
    return reader, [page.extract_text() for page in reader.pages]


def test_renders_a_valid_pdf_with_core_content(tmp_path):
    pdf = render_report_pdf(make_result(), [], tmp_path)
    assert pdf.startswith(b"%PDF")
    _, pages = pdf_text(pdf)
    text = "\n".join(pages)
    for expected in [
        "MEDVISION",
        "Pleural Effusion",
        "Congestive Heart Failure",
        "Right Lower Zone",
        "Not for clinical use",
        "[F1]",
        "Single frontal view.",
        "Pleural Effusion suggestive_of CHF",
        "No reviewer feedback",
    ]:
        assert expected in text, expected
    assert "Pulmonary Mass" not in text  # below-threshold finding is not listed
    assert "1 further finding(s) were below the reporting threshold" in text


def test_reviewer_feedback_appears_in_tables(tmp_path):
    feedback = [fb("F_EFFUSION", "agree", "clear blunting"), fb("D_CHF", "disagree")]
    text = "\n".join(pdf_text(render_report_pdf(make_result(), feedback, tmp_path))[1])
    assert "Agree" in text and "Disagree" in text
    assert "clear blunting" in text
    assert "No reviewer feedback" not in text


def test_untrusted_text_cannot_inject_markup_or_black_boxes(tmp_path):
    nasty = "<b>bold</b> & <unclosed Effusion ≥ 50% ⚠ → done 中"
    result = make_result(
        report=make_result().report.model_copy(
            update={"findings": [ReportItem(text=nasty, evidence_ids=["F1"])]}
        ),
        quality_warnings=["<script>alert(1)</script> ⚠"],
    )
    text = "\n".join(pdf_text(render_report_pdf(result, [], tmp_path))[1])
    assert "<b>bold</b> & <unclosed" in text  # shown literally, not interpreted
    assert ">= 50%" in text and "-> done" in text  # ASCII fallbacks
    assert "⚠" not in text and "中" not in text  # unsupported glyphs replaced
    assert "<script>alert(1)</script>" in text


def test_images_are_embedded_when_present_and_skipped_when_missing(tmp_path):
    arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    Image.fromarray(arr, mode="L").save(tmp_path / "original.png")
    Image.fromarray(arr, mode="L").convert("RGB").save(tmp_path / "heatmap_F_EFFUSION.png")

    with_images, _ = pdf_text(render_report_pdf(make_result(), [], tmp_path))
    assert len(with_images.pages[0].images) == 2

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    without, _ = pdf_text(render_report_pdf(make_result(), [], empty_dir))
    assert len(without.pages[0].images) == 0


def test_heatmap_path_cannot_escape_the_upload_dir(tmp_path):
    secret = tmp_path / "secret.png"
    Image.new("L", (32, 32), 200).save(secret)
    upload = tmp_path / "upload"
    upload.mkdir()
    result = make_result()
    result.findings[0].heatmap_path = "../secret.png"
    reader, _ = pdf_text(render_report_pdf(result, [], upload))
    assert len(reader.pages[0].images) == 0


def test_long_report_paginates_and_keeps_disclaimer_on_every_page(tmp_path):
    long_items = [
        ReportItem(text=f"Statement number {i}. " + "lorem ipsum " * 25, evidence_ids=["F1"])
        for i in range(40)
    ]
    result = make_result(report=make_result().report.model_copy(update={"findings": long_items}))
    reader, pages = pdf_text(render_report_pdf(result, [], tmp_path))
    assert len(reader.pages) > 1
    assert all("Not for clinical use" in p for p in pages)
    assert pages[-1].count("Page") >= 1


def test_verifier_problems_are_disclosed(tmp_path):
    grounding = GroundingResult(
        score=0.5, unsupported=["finding statement: has no evidence_ids"], contradictions=["x"]
    )
    text = "\n".join(
        pdf_text(
            render_report_pdf(make_result(report_source="llm", grounding=grounding), [], tmp_path)
        )[1]
    )
    assert "50%" in text and "has no evidence_ids" in text and "LLM-narrated" in text


@pytest.mark.parametrize("empty_part", ["findings", "differentials"])
def test_renders_when_sections_are_empty(tmp_path, empty_part):
    result = make_result(**{empty_part: []})
    text = "\n".join(pdf_text(render_report_pdf(result, [], tmp_path))[1])
    assert "MEDVISION" in text
    if empty_part == "differentials":
        assert "No differential diagnosis reached" in text


def test_accepts_a_path_that_does_not_exist(tmp_path):
    assert render_report_pdf(make_result(), [], Path(tmp_path / "nope")).startswith(b"%PDF")


def test_retrieval_section_is_rendered_when_present_and_omitted_otherwise(tmp_path):
    from medvision.pipeline import NodeHintOut, RetrievalOut, SimilarCaseOut
    from medvision.reasoning.schema import EvidenceCase

    base = make_result()
    assert "Similar reference cases" not in "\n".join(
        pdf_text(render_report_pdf(base, [], tmp_path))[1]
    )

    retrieval = RetrievalOut(
        node_hints=[NodeHintOut(node_id="F_EFFUSION", name="Pleural Effusion", score=0.81)],
        similar_cases=[
            SimilarCaseOut(
                case_id="nih-test-42",
                similarity=0.87,
                labels=["Effusion", "Atelectasis"],
                finding_node_ids=["F_EFFUSION", "F_ATELECTASIS"],
                split="test",
            )
        ],
    )
    case = EvidenceCase(
        id="C1", case_id="nih-test-42", similarity=0.87, labels=["Effusion"], text="Similar case C1"
    )
    evidence = base.evidence.model_copy(update={"cases": [case]})
    text = "\n".join(
        pdf_text(
            render_report_pdf(
                base.model_copy(update={"retrieval": retrieval, "evidence": evidence}), [], tmp_path
            )
        )[1]
    )
    for expected in [
        "Similar reference cases",
        "nih-test-42",
        "Effusion, Atelectasis",
        "87%",
        "Closest knowledge-graph nodes",
        "Pleural Effusion (81)",
        "Similar case C1",
    ]:
        assert expected in text, expected
