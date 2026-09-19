#!/usr/bin/env python3
# ruff: noqa: E501
"""Builds MEDVISION_Project_Report.pdf: problem statement, solution, literature
survey (gaps + how MEDVISION addresses them), a workflow flowchart, and current
project status. Run with: uv run --with reportlab python3 scripts/build_report_pdf.py

This is a standalone reporting utility (not part of the medvision package, not
imported anywhere, not covered by the test suite) so its long, readable content
strings are exempt from the project's normal line-length rule.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = str(REPO_ROOT / "MEDVISION_Project_Report.pdf")

NAVY = colors.HexColor("#0f172a")
SKY = colors.HexColor("#0284c7")
GREEN = colors.HexColor("#059669")
GREEN_BG = colors.HexColor("#ecfdf5")
GRAY = colors.HexColor("#64748b")
LIGHT_GRAY = colors.HexColor("#f1f5f9")
AMBER = colors.HexColor("#b45309")
AMBER_BG = colors.HexColor("#fffbeb")

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "MVTitle", parent=styles["Title"], fontSize=24, textColor=NAVY, spaceAfter=6
    )
)
styles.add(
    ParagraphStyle(
        "MVSubtitle", parent=styles["Normal"], fontSize=12, textColor=GRAY, spaceAfter=20
    )
)
styles.add(
    ParagraphStyle(
        "MVH1",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=NAVY,
        spaceBefore=18,
        spaceAfter=8,
        borderPadding=0,
    )
)
styles.add(
    ParagraphStyle(
        "MVH2", parent=styles["Heading2"], fontSize=12.5, textColor=SKY, spaceBefore=12, spaceAfter=6
    )
)
styles.add(
    ParagraphStyle(
        "MVBody", parent=styles["Normal"], fontSize=10, leading=14.5, spaceAfter=8
    )
)
styles.add(
    ParagraphStyle(
        "MVCell", parent=styles["Normal"], fontSize=8.3, leading=11
    )
)
styles.add(
    ParagraphStyle(
        "MVCellBold", parent=styles["Normal"], fontSize=8.3, leading=11, fontName="Helvetica-Bold"
    )
)
styles.add(
    ParagraphStyle(
        "MVCaption", parent=styles["Normal"], fontSize=8.5, textColor=GRAY, spaceAfter=4
    )
)
styles.add(
    ParagraphStyle(
        "MVDisclaimer",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=AMBER,
        backColor=AMBER_BG,
        borderPadding=8,
        borderWidth=0.5,
        borderColor=colors.HexColor("#fde68a"),
    )
)


def cell(text, bold=False):
    return Paragraph(text, styles["MVCellBold"] if bold else styles["MVCell"])


# ---------------------------------------------------------------------------
# Flowchart flowable — drawn directly with canvas primitives, no external image
# ---------------------------------------------------------------------------


class WorkflowFlowchart(Flowable):
    def __init__(self, width=460, height=640):
        super().__init__()
        self.width = width
        self.height = height

    def _box(self, c, x, y, w, h, text, done=True, sub=None, fontsize=9.2):
        fill = GREEN_BG if done else colors.HexColor("#f8fafc")
        stroke = GREEN if done else GRAY
        dash = None if done else (3, 2)
        c.saveState()
        if dash:
            c.setDash(*dash)
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(1.2)
        c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
        c.restoreState()
        c.setFillColor(NAVY if done else GRAY)
        c.setFont("Helvetica-Bold", fontsize)
        c.drawCentredString(x + w / 2, y + h / 2 + (4 if sub else 0), text)
        if sub:
            c.setFont("Helvetica", 7.3)
            c.setFillColor(GRAY)
            c.drawCentredString(x + w / 2, y + h / 2 - 8, sub)
        badge = "✓ DONE" if done else "PLANNED"
        c.setFont("Helvetica-Bold", 6.3)
        c.setFillColor(GREEN if done else GRAY)
        c.drawRightString(x + w - 6, y + h - 9, badge)

    def _arrow(self, c, x, y1, y2):
        c.saveState()
        c.setStrokeColor(GRAY)
        c.setLineWidth(1.1)
        c.line(x, y1, x, y2)
        c.setFillColor(GRAY)
        c.line(x, y2, x - 3, y2 + 6)
        c.line(x, y2, x + 3, y2 + 6)
        c.restoreState()

    def draw(self):
        c = self.canv
        w = self.width
        cx = w / 2
        bw_full = w
        bw_half = w / 2 - 8
        bh = 34
        gap = 16
        y = self.height - bh

        # 1. Upload
        self._box(c, 0, y, bw_full, bh, "Chest X-ray Upload (PNG / JPG)")
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        # 2. QC / OOD gate
        self._box(
            c, 0, y, bw_full, bh,
            "Input Quality / OOD Gate",
            sub="rejects color photos, blank images, undersized uploads",
        )
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        # 3. dual stream: vision classifier + knowledge graph
        self._box(
            c, 0, y, bw_half, bh,
            "Vision Classifier",
            sub="torchxrayvision → 18 findings",
        )
        self._box(
            c, bw_half + 16, y, bw_half, bh,
            "Knowledge Graph",
            sub="51 nodes, 66 weighted edges",
        )
        y2 = y - gap - bh
        self._arrow(c, bw_half / 2, y, y2 + bh)
        self._arrow(c, bw_half + 16 + bw_half / 2, y, y2 + bh)
        # merge arrows toward centre
        c.saveState()
        c.setStrokeColor(GRAY)
        c.setLineWidth(1.1)
        c.line(bw_half / 2, y2 + bh + 8, bw_half + 16 + bw_half / 2, y2 + bh + 8)
        c.restoreState()
        y = y2

        # 4. calibration
        self._box(
            c, 0, y, bw_full, bh,
            "Calibration (Temperature Scaling)",
            sub="fitted on real labeled data for 2/18 findings; identity elsewhere",
        )
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        # 5. Grad-CAM + localization
        self._box(
            c, 0, y, bw_full, bh,
            "Grad-CAM Heatmap + Anatomical Zone Localization",
        )
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        # 6. KG reasoning
        self._box(
            c, 0, y, bw_full, bh,
            "Deterministic Knowledge-Graph Reasoning",
            sub="Σ p(finding) × edge weight → ranked differential diagnoses",
        )
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        # 7. LLM narration + verifier loop
        self._box(
            c, 0, y, bw_full, bh + 8,
            "LLM Narration (local Ollama) <-> Citation Verifier",
            sub="regenerates once if grounding score < 0.8",
            fontsize=9,
        )
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        # 8. UI
        self._box(
            c, 0, y, bw_full, bh,
            "React UI: Heatmap + Findings + Evidence Graph + Cited Report",
            fontsize=8.6,
        )
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        # 9. Future work
        self._box(
            c, 0, y, bw_full, bh,
            "Planned: Contrastive Alignment + ChromaDB Retrieval",
            sub="similar-case retrieval — not yet built (T16–T20)",
            done=False,
        )


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

LITERATURE_ROWS = [
    ("Paper", "Core Contribution", "Gap", "How MEDVISION Addresses It"),
    (
        "ConVIRT (2021)",
        "Contrastive image-text pretraining for medical images",
        "Produces aligned embeddings but no structured, auditable diagnostic reasoning",
        "Predicts KG-anchored finding nodes directly; every diagnosis traces to a specific graph edge, not an opaque embedding",
    ),
    (
        "GLoRIA (2021)",
        "Global-local contrastive representation learning",
        "Alignment quality is not explained in clinical/anatomical terms",
        "Combines Grad-CAM with anatomical zone localization, so every finding is tied to a body region, not just a similarity score",
    ),
    (
        "MedCLIP (2022)",
        "Contrastive alignment from weakly-paired medical data",
        "Optimized for retrieval/classification, not generative explanation",
        "Uses classifier output as structured evidence for LLM narration with mandatory citations (alignment itself is future work, T16)",
    ),
    (
        "CLIP (2021)",
        "General-purpose image-text pretraining",
        "Not medical-domain-specific; no clinical vocabulary",
        "Vision stage uses torchxrayvision, trained specifically on 5 combined hospital chest X-ray datasets",
    ),
    (
        "BiomedCLIP (2023)",
        "Large-scale biomedical image-text pretraining",
        "Still lacks a symbolic/auditable reasoning layer on top of embeddings",
        "Adds a hand-authored, weighted knowledge graph as the reasoning substrate, independently validated with 6 automated rules",
    ),
    (
        "RadGraph (2021)",
        "Entity-relation extraction from radiology reports",
        "Extraction only — no downstream scoring or ranking mechanism",
        "KG reasoning module turns activated findings into a normalized, ranked differential score with a full weighted evidence trail",
    ),
    (
        "GCN (2017) / GAT (2018)",
        "Graph neural network architectures for relational data",
        "Require training data; resulting embeddings are themselves black boxes",
        "Uses a transparent deterministic weighted-sum score instead of a learned embedding; GNN encoding is reserved as explicit future work (T17)",
    ),
    (
        "GAMENet (2019)",
        "Graph-augmented treatment recommendation",
        "Focused on recommendation, not evidence-level explanation",
        "Every ranked diagnosis is returned with the exact contributing findings, edge weights, and signed contribution values",
    ),
    (
        "Grad-CAM (2017) / Integrated Gradients (2017)",
        "Pixel-level visual attribution for CNN decisions",
        "A heatmap alone does not explain the clinical reasoning behind a diagnosis",
        "Combines Grad-CAM with KG evidence paths and a cited natural-language report — visual AND textual explanation together",
    ),
    (
        "BioBERT (2020) / ClinicalBERT (2019)",
        "Biomedical/clinical domain language models",
        "Domain-adapted text understanding, not multimodal evidence narration",
        "A local LLM (Ollama) narrates the combined image + knowledge-graph evidence into a structured clinical report",
    ),
    (
        "R2Gen (2020)",
        "Memory-driven transformer for report generation",
        "No mechanism to verify generated text against real evidence — hallucination risk",
        "Built a citation verifier that checks every generated sentence against the evidence block and regenerates ungrounded reports",
    ),
    (
        "LLaVA-Med (2024)",
        "Biomedical vision-language instruction tuning",
        "Needs large paired instruction-tuning data; not inherently auditable",
        "No fine-tuning required — narration is schema-constrained to cite real evidence IDs (F#, D#, K#) by construction",
    ),
    (
        "QLoRA (2023)",
        "Efficient low-bit fine-tuning for local LLM deployment",
        "A fine-tuning recipe, not a full local-inference application",
        "Runs a local Ollama LLM directly for inference, satisfying the privacy/on-premises requirement without any fine-tuning step",
    ),
    (
        "RAG (2020)",
        "Retrieval-augmented text generation",
        "General text retrieval, not conditioned on a medical image's findings",
        "Evidence retrieval today is graph-based; ChromaDB vector retrieval of similar past cases is explicit planned future work (T19–T20)",
    ),
    (
        "Medical Graph RAG (2025)",
        "Grounds LLM output in medical dictionaries via triple-graph construction",
        "Text-only pipeline with no vision-encoder integration",
        "Integrates a real vision classifier's findings directly into the same graph-grounded reasoning and narration pipeline",
    ),
    (
        "MIMIC-CXR / ChestX-ray14 / CheXpert / CheXNet",
        "Large chest X-ray datasets and CNN benchmarks",
        "MIMIC-CXR requires PhysioNet credentialing; none provide graph-grounded explanations",
        "Uses the freely available pretrained torchxrayvision classifier (already trained on these datasets) plus a real, freely obtained Normal-vs-Pneumonia dataset for calibration and evaluation",
    ),
]

STATUS_ROWS = [
    ("Component", "Status", "Evidence"),
    ("Vision classifier (18 findings)", "Done", "Pretrained torchxrayvision DenseNet-121, verified on real data"),
    ("Grad-CAM + zone localization", "Done", "Heatmaps + coarse anatomical zone assignment, verified visually"),
    ("Knowledge graph (51 nodes, 66 edges)", "Done", "6 automated validation rules pass; hand-authored, source-tagged"),
    ("Deterministic KG differential reasoning", "Done", "Recall@1 = 86.4%, Recall@3 = 96.7% for pneumonia (held-out data)"),
    ("Calibration (temperature scaling)", "Done (2/18 findings)", "ECE reduced 63% and 67% on 2 findings with real ground truth"),
    ("LLM narration + citation verifier", "Done", "100% grounding score achieved on verified real test run"),
    ("Input quality / OOD gate", "Done", "Real color photo correctly rejected in a live server test"),
    ("KG edge-weight correction (false-alarm fix)", "Done", "False-alarm rate reduced from 48.4% to 37.8% on real data"),
    ("Evaluation harness", "Done", "91 automated tests passing; real metrics in reports/eval_report.md"),
    ("React UI (findings, evidence graph, report, dark mode)", "Done", "Verified live in browser, light and dark themes"),
    ("DICOM upload + de-identification", "Not started", "PLAN.md T07"),
    ("Multimodal contrastive alignment", "Not started", "PLAN.md T16–T18"),
    ("ChromaDB similar-case retrieval", "Not started", "PLAN.md T19–T20"),
    ("Knowledge-graph enrichment (PrimeKG, MeSH)", "Not started", "PLAN.md T10"),
    ("Ground truth for other 15 diseases", "Not started", "Only Normal-vs-Pneumonia labels obtained so far"),
    ("PDF report export, Docker packaging", "Not started", "PLAN.md T26 / T33"),
]


def status_style_for_row(i, text):
    if text == "Done" or text.startswith("Done"):
        return GREEN
    return AMBER


def build():
    doc = SimpleDocTemplate(
        OUT_PATH,
        pagesize=LETTER,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="MEDVISION Project Report",
        author="Batch B53 — SIT Tumakuru",
    )
    story = []

    # ---------------- Title ----------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("MEDVISION", styles["MVTitle"]))
    story.append(
        Paragraph(
            "A Knowledge-Graph Anchored Vision-Reasoning Framework for Explainable Clinical Diagnosis",
            styles["MVSubtitle"],
        )
    )
    story.append(
        Paragraph(
            "Project report — problem statement, solution, literature review, workflow, and current status",
            styles["MVBody"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "WARNING: Research prototype for a college major project. Not a certified medical device. "
            "Not for clinical use.",
            styles["MVDisclaimer"],
        )
    )
    story.append(Spacer(1, 30))

    # ---------------- Problem Statement ----------------
    story.append(Paragraph("1. Problem Statement", styles["MVH1"]))
    story.append(
        Paragraph(
            "Modern AI systems for medical imaging achieve strong predictive accuracy, but most operate as "
            '"black boxes": they output a label or probability without exposing the clinical reasoning behind '
            "it. This creates a trust gap in real clinical workflows, where a physician must be able to verify "
            "why a system reached a conclusion before acting on it — especially in high-stakes diagnostic "
            "decisions. Existing explainability methods (e.g. saliency maps) show <i>where</i> a model looked, "
            "but not <i>why</i> that visual pattern implies a specific diagnosis in medical terms. Separately, "
            "privacy-sensitive clinical environments increasingly require local, on-premises inference rather "
            "than cloud-dependent pipelines.",
            styles["MVBody"],
        )
    )

    story.append(Paragraph("2. Our Solution", styles["MVH1"]))
    story.append(
        Paragraph(
            "MEDVISION reframes diagnosis as a <b>grounded reasoning problem</b> rather than a pure "
            "classification task. A pretrained vision model detects 18 radiographic findings on a chest X-ray. "
            "Instead of stopping there, each finding is checked against a hand-authored medical knowledge graph "
            "(findings → diseases, with a weighted, source-tagged edge for every connection), which produces a "
            "ranked differential diagnosis through simple, fully transparent arithmetic — not a second black-box "
            "model. A local large language model then narrates this evidence into a clinician-readable report, "
            "and a dedicated verifier checks every generated sentence against the evidence before it is shown, "
            "rejecting or regenerating anything that isn't properly grounded. The result is a system where every "
            "displayed diagnosis, heatmap, and sentence can be traced back to a specific, inspectable piece of "
            "evidence.",
            styles["MVBody"],
        )
    )
    story.append(
        Paragraph(
            "Key design principles: (1) the diagnosis ranking itself is deterministic and auditable, never "
            "delegated to the LLM; (2) the LLM's only job is narration, constrained to cite real evidence IDs; "
            "(3) every claim is checked before being shown to the user; (4) inference runs entirely on local "
            "hardware for data privacy.",
            styles["MVBody"],
        )
    )

    story.append(PageBreak())

    # ---------------- Literature Survey ----------------
    story.append(Paragraph("3. Literature Survey: Gaps and How MEDVISION Addresses Them", styles["MVH1"]))
    story.append(
        Paragraph(
            "The table below summarizes the most directly relevant prior work, the specific gap each leaves "
            "open relative to MEDVISION's goal of grounded, explainable diagnosis, and the concrete mechanism "
            "MEDVISION uses to close that gap — built and verified where marked, or explicitly scoped as "
            "future work otherwise.",
            styles["MVBody"],
        )
    )
    story.append(Spacer(1, 6))

    table_data = [[cell(h, bold=True) for h in LITERATURE_ROWS[0]]]
    for row in LITERATURE_ROWS[1:]:
        table_data.append([cell(row[0], bold=True), cell(row[1]), cell(row[2]), cell(row[3])])

    lit_table = Table(
        table_data,
        colWidths=[1.05 * inch, 1.35 * inch, 1.55 * inch, 2.05 * inch],
        repeatRows=1,
    )
    lit_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(lit_table)

    story.append(PageBreak())

    # ---------------- Workflow ----------------
    story.append(Paragraph("4. System Workflow", styles["MVH1"]))
    story.append(
        Paragraph(
            "End-to-end pipeline from image upload to a cited, verified diagnostic report. Green boxes are "
            "built, tested, and verified against real data; the dashed gray box is explicitly scoped future "
            "work, not yet implemented.",
            styles["MVCaption"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(WorkflowFlowchart(width=460, height=560))

    story.append(PageBreak())

    # ---------------- Current Status ----------------
    story.append(Paragraph("5. Current Project Status", styles["MVH1"]))
    story.append(
        Paragraph(
            "MEDVISION is implemented and running end-to-end — not a design-only proposal. The table below "
            "reflects what has actually been built, tested, and verified against real data, versus what "
            "remains as scoped future work.",
            styles["MVBody"],
        )
    )
    story.append(Spacer(1, 6))

    status_table_data = [[cell(h, bold=True) for h in STATUS_ROWS[0]]]
    for row in STATUS_ROWS[1:]:
        status_table_data.append([cell(row[0], bold=True), cell(row[1]), cell(row[2])])

    status_table = Table(
        status_table_data,
        colWidths=[2.1 * inch, 1.3 * inch, 2.6 * inch],
        repeatRows=1,
    )
    status_style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i, row in enumerate(STATUS_ROWS[1:], start=1):
        color = GREEN if (row[1] == "Done" or row[1].startswith("Done")) else AMBER
        status_style.append(("TEXTCOLOR", (1, i), (1, i), color))
    status_table.setStyle(TableStyle(status_style))
    story.append(status_table)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Headline real results (not projected — measured):", styles["MVH2"]))
    bullets = [
        "Differential diagnosis Recall@1 (pneumonia, held-out real data): <b>86.4%</b>; Recall@3: <b>96.7%</b>",
        "False-alarm rate on true-negative cases: reduced from <b>48.4% → 37.8%</b> after correcting one "
        "over-weighted knowledge-graph edge, backed by a full root-cause investigation",
        "Calibration error (ECE) reduced by <b>63%</b> and <b>67%</b> for the two findings with real ground "
        "truth, evaluated on data held out from fitting",
        "LLM-narrated reports achieved a <b>100% grounding score</b> (every sentence traceable to real "
        "evidence) after tightening the JSON schema to require at least one citation per statement",
        "<b>91 automated tests</b> passing across the vision, knowledge-graph, calibration, reasoning, "
        "narration, quality-gate, and API layers",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(b, styles["MVBody"])) for b in bullets],
            bulletType="bullet",
            leftIndent=14,
        )
    )

    story.append(Spacer(1, 10))
    story.append(Paragraph("What remains (honest scope, not overclaimed):", styles["MVH2"]))
    remaining = [
        "DICOM upload and de-identification (currently PNG/JPG only)",
        "Multimodal contrastive alignment and ChromaDB-based similar-case retrieval",
        "Knowledge-graph enrichment from PrimeKG and MeSH (currently 100% hand-authored)",
        "Ground-truth evaluation data for the other 15 diseases in the graph (only Pneumonia validated so far)",
        "PDF report export from the application itself, Docker packaging, and a formal model card",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(b, styles["MVBody"])) for b in remaining],
            bulletType="bullet",
            leftIndent=14,
        )
    )

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
