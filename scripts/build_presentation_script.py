#!/usr/bin/env python3
# ruff: noqa: E501
"""Builds MEDVISION_Presentation_Script.pdf: a 4-speaker presentation script
(Introduction, Solution, Live Demo, Research Gaps + Flowchart + Conclusion),
with speaking cues and an embedded workflow flowchart.

Run with: uv run --with reportlab python3 scripts/build_presentation_script.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = str(REPO_ROOT / "MEDVISION_Presentation_Script.pdf")

NAVY = colors.HexColor("#0f172a")
SKY = colors.HexColor("#0284c7")
GREEN = colors.HexColor("#059669")
GREEN_BG = colors.HexColor("#ecfdf5")
GRAY = colors.HexColor("#64748b")
LIGHT_GRAY = colors.HexColor("#f1f5f9")
AMBER_BG = colors.HexColor("#fffbeb")
AMBER = colors.HexColor("#b45309")
CUE_BG = colors.HexColor("#eef2ff")
CUE_BORDER = colors.HexColor("#c7d2fe")
CUE_TEXT = colors.HexColor("#4338ca")

SPEAKERS = [
    ("Dhanush Gowda A S", "Introduction"),
    ("Dhruva D", "Our Solution"),
    ("Pradyumna Prasad", "Live Demo Walkthrough"),
    ("V A Manjunatha", "Research Gaps, Workflow & Conclusion"),
]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("MVTitle", parent=styles["Title"], fontSize=23, textColor=NAVY, spaceAfter=6))
styles.add(ParagraphStyle("MVSubtitle", parent=styles["Normal"], fontSize=12, textColor=GRAY, spaceAfter=16))
styles.add(
    ParagraphStyle(
        "MVSectionHeader",
        parent=styles["Heading1"],
        fontSize=17,
        textColor=colors.white,
        spaceAfter=0,
        spaceBefore=0,
        leading=20,
    )
)
styles.add(
    ParagraphStyle(
        "MVSpeaker", parent=styles["Normal"], fontSize=10.5, textColor=colors.white, spaceAfter=0, leading=13
    )
)
styles.add(ParagraphStyle("MVScript", parent=styles["Normal"], fontSize=11, leading=17, spaceAfter=11))
styles.add(
    ParagraphStyle(
        "MVCue", parent=styles["Normal"], fontSize=9.5, textColor=CUE_TEXT, leading=13, spaceAfter=4
    )
)
styles.add(ParagraphStyle("MVH2", parent=styles["Heading2"], fontSize=12.5, textColor=SKY, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("MVCaption", parent=styles["Normal"], fontSize=8.5, textColor=GRAY, spaceAfter=4))
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
styles.add(ParagraphStyle("MVCell", parent=styles["Normal"], fontSize=9, leading=12))
styles.add(ParagraphStyle("MVCellBold", parent=styles["Normal"], fontSize=9, leading=12, fontName="Helvetica-Bold"))


def cell(text, bold=False):
    return Paragraph(text, styles["MVCellBold"] if bold else styles["MVCell"])


def cue(text):
    """A stage-direction / speaking cue, visually distinct from the script itself."""
    t = Table([[Paragraph(f"→ {text}", styles["MVCue"])]], colWidths=[6.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CUE_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, CUE_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def section_header(number, title, speaker, minutes):
    t = Table(
        [
            [
                Paragraph(f"{number}. {title}", styles["MVSectionHeader"]),
                Paragraph(f"{speaker}<br/>~{minutes} min", styles["MVSpeaker"]),
            ]
        ],
        colWidths=[4.6 * inch, 1.8 * inch],
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return t


# ---------------------------------------------------------------------------
# Flowchart flowable (same design as the project report's, kept in sync)
# ---------------------------------------------------------------------------


class WorkflowFlowchart(Flowable):
    def __init__(self, width=440, height=560):
        super().__init__()
        self.width = width
        self.height = height

    def _box(self, c, x, y, w, h, text, done=True, sub=None, fontsize=8.8):
        fill = GREEN_BG if done else colors.HexColor("#f8fafc")
        stroke = GREEN if done else GRAY
        c.saveState()
        if not done:
            c.setDash(3, 2)
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(1.1)
        c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
        c.restoreState()
        c.setFillColor(NAVY if done else GRAY)
        c.setFont("Helvetica-Bold", fontsize)
        c.drawCentredString(x + w / 2, y + h / 2 + (4 if sub else 0), text)
        if sub:
            c.setFont("Helvetica", 6.9)
            c.setFillColor(GRAY)
            c.drawCentredString(x + w / 2, y + h / 2 - 8, sub)
        badge = "DONE" if done else "PLANNED"
        c.setFont("Helvetica-Bold", 6)
        c.setFillColor(GREEN if done else GRAY)
        c.drawRightString(x + w - 6, y + h - 9, badge)

    def _arrow(self, c, x, y1, y2):
        c.saveState()
        c.setStrokeColor(GRAY)
        c.setLineWidth(1.0)
        c.line(x, y1, x, y2)
        c.line(x, y2, x - 3, y2 + 6)
        c.line(x, y2, x + 3, y2 + 6)
        c.restoreState()

    def draw(self):
        c = self.canv
        w = self.width
        cx = w / 2
        bw_half = w / 2 - 8
        bh = 32
        gap = 14
        y = self.height - bh

        self._box(c, 0, y, w, bh, "Chest X-ray Upload (PNG / JPG)")
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        self._box(c, 0, y, w, bh, "Input Quality / OOD Gate", sub="rejects non-chest-X-ray uploads")
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        self._box(c, 0, y, bw_half, bh, "Vision Classifier", sub="18 findings")
        self._box(c, bw_half + 16, y, bw_half, bh, "Knowledge Graph", sub="51 nodes, 66 edges")
        y2 = y - gap - bh
        self._arrow(c, bw_half / 2, y, y2 + bh)
        self._arrow(c, bw_half + 16 + bw_half / 2, y, y2 + bh)
        c.saveState()
        c.setStrokeColor(GRAY)
        c.setLineWidth(1.0)
        c.line(bw_half / 2, y2 + bh + 8, bw_half + 16 + bw_half / 2, y2 + bh + 8)
        c.restoreState()
        y = y2

        self._box(c, 0, y, w, bh, "Calibration (Temperature Scaling)", sub="fitted on real labeled data")
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        self._box(c, 0, y, w, bh, "Grad-CAM Heatmap + Zone Localization")
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        self._box(c, 0, y, w, bh, "Deterministic KG Reasoning", sub="ranked differential diagnoses")
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        self._box(c, 0, y, w, bh + 6, "LLM Narration ↔ Citation Verifier", sub="regenerates if ungrounded")
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        self._box(c, 0, y, w, bh, "React UI: Heatmap + Findings + Report", fontsize=8.2)
        y2 = y - gap - bh
        self._arrow(c, cx, y, y2 + bh)
        y = y2

        self._box(
            c, 0, y, w, bh,
            "Planned: Contrastive Alignment + Retrieval",
            sub="not yet built (future work)",
            done=False,
        )


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

GAP_ROWS = [
    ("Prior Work", "Gap", "How MEDVISION Addresses It"),
    (
        "CLIP / MedCLIP / BiomedCLIP",
        "Strong image-text alignment, but no structured, auditable reasoning layer",
        "Predicts KG-anchored findings directly; every diagnosis traces to a specific graph edge",
    ),
    (
        "RadGraph / GCN / GAT",
        "Graph extraction or embeddings that are themselves black boxes",
        "Deterministic weighted-sum scoring over a hand-authored graph — fully inspectable, no hidden embeddings",
    ),
    (
        "R2Gen / LLaVA-Med",
        "Fluent report generation with a real hallucination risk",
        "A dedicated verifier checks every generated sentence against real evidence before display",
    ),
    (
        "QLoRA",
        "An efficient fine-tuning recipe, not a full local application",
        "Runs a local Ollama LLM directly for inference — private, on-premises, no fine-tuning needed",
    ),
]


def build():
    doc = SimpleDocTemplate(
        OUT_PATH,
        pagesize=LETTER,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="MEDVISION Presentation Script",
        author="Batch B53 — SIT Tumakuru",
    )
    story = []

    # ---------------- Title / cue sheet ----------------
    story.append(Spacer(1, 20))
    story.append(Paragraph("MEDVISION", styles["MVTitle"]))
    story.append(Paragraph("Team Presentation Script — 4 Speakers", styles["MVSubtitle"]))
    story.append(
        Paragraph(
            "WARNING: Research prototype for a college major project. Not a certified medical device. "
            "Not for clinical use.",
            styles["MVDisclaimer"],
        )
    )
    story.append(Spacer(1, 16))
    story.append(Paragraph("Running order", styles["MVH2"]))

    order_data = [[cell("#", True), cell("Speaker", True), cell("Section", True), cell("Time", True)]]
    for i, (name, section) in enumerate(SPEAKERS, start=1):
        order_data.append([cell(str(i)), cell(name), cell(section), cell("~2–4 min")])
    order_table = Table(order_data, colWidths=[0.3 * inch, 1.7 * inch, 3.2 * inch, 0.9 * inch])
    order_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(order_table)
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Total estimated time: ~13–14 minutes, plus Q&amp;A. Blue boxes throughout this script are "
            "stage directions — things to do or point at, not things to say aloud. Swap speakers/sections "
            "freely to match who's most comfortable with each part.",
            styles["MVCaption"],
        )
    )

    story.append(PageBreak())

    # ---------------- Section 1: Introduction ----------------
    story.append(section_header(1, "Introduction", "Dhanush Gowda A S", 2))
    story.append(Spacer(1, 10))
    story.append(cue("Stand centrally, make eye contact with the panel before starting."))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Good morning/afternoon everyone. I'm Dhanush, and on behalf of my team — Dhruva, Pradyumna, "
            "and Manjunatha — welcome to our presentation of MEDVISION.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "Let me start with the problem we're solving. Artificial intelligence has gotten remarkably good "
            "at reading medical images like chest X-rays. Modern deep learning models can spot patterns doctors "
            "sometimes miss. But there's a catch: most of these systems are black boxes. They'll say "
            "‘there's a 90% chance this is pneumonia’ — but they won't tell you why. And in medicine, "
            "‘why’ matters enormously. A doctor won't act on a number alone; they need to see the "
            "reasoning, the evidence, before they trust a recommendation — especially when a patient's "
            "health is on the line.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "This is the trust gap MEDVISION is built to close. Our project is called MEDVISION: a "
            "Knowledge-Graph Anchored Vision-Reasoning Framework for Explainable Clinical Diagnosis. The core "
            "idea is simple to state but genuinely hard to build: instead of asking an AI model to just spit "
            "out a label, we ask it to reason the way a doctor might — connect what it sees on the image "
            "to established medical knowledge, and show its work at every single step.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "Over the next few minutes, my teammates will walk you through our solution architecture, show you "
            "a live demo running on real chest X-rays, connect our approach back to the research literature, "
            "and close with where the project stands today. Let me hand over to Dhruva, who'll explain our "
            "actual solution.",
            styles["MVScript"],
        )
    )

    story.append(PageBreak())

    # ---------------- Section 2: Solution ----------------
    story.append(section_header(2, "Our Solution", "Dhruva D", 3))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Thanks, Dhanush. So — how does MEDVISION actually work?",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "We built the system around one central design decision: separate perception from reasoning. Most "
            "AI diagnostic tools blend these together into one opaque model. We deliberately split them apart.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "First, perception. We use a pretrained vision model — a DenseNet-121 from a library called "
            "torchxrayvision — already trained on hundreds of thousands of real chest X-rays from five "
            "major hospital datasets. It looks at an uploaded X-ray and scores 18 possible findings: things "
            "like consolidation, effusion, an enlarged heart, and more.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "But here's where it gets interesting. Instead of stopping at a probability, we built a knowledge "
            "graph — think of it as a chart connecting findings to diseases, the same way a textbook connects "
            "symptoms to diagnoses. We hand-authored this ourselves: 51 medical concepts and 66 weighted "
            "connections, each backed by a real clinical reason. When the vision model says ‘consolidation, "
            "85% likely,’ our knowledge graph says ‘consolidation is strong evidence for pneumonia, "
            "weight 0.85’ — and we simply add up this evidence using transparent arithmetic, not another "
            "black box. Every diagnosis we produce comes with an exact evidence trail you can inspect.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "We didn't stop there. A raw model's confidence numbers aren't always trustworthy, so we calibrated "
            "them against real labeled data — and we can prove, with measured numbers, that this made our "
            "confidence estimates significantly more accurate.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "Finally, to make the output readable, we use a local language model — running entirely on our "
            "own machine, no cloud, no internet dependency — to turn this structured evidence into a "
            "plain-English report. But language models can hallucinate, so we built a verifier: it checks every "
            "sentence in the generated report against real evidence before showing it to you. If a sentence "
            "isn't backed up, the report gets rewritten.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "The result is a system where every heatmap, every diagnosis, and every sentence in the final "
            "report can be traced back to something real and checkable. Now let's actually see it in action "
            "— over to Pradyumna.",
            styles["MVScript"],
        )
    )

    story.append(PageBreak())

    # ---------------- Section 3: Demo ----------------
    story.append(section_header(3, "Live Demo Walkthrough", "Pradyumna Prasad", 4))
    story.append(Spacer(1, 10))
    story.append(cue("Have the MEDVISION app open and the backend/frontend already running before you start talking."))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Thank you, Dhruva. I'll walk you through exactly what happens, step by step, when someone uses "
            "MEDVISION — live, on our actual running application.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "<b>Step one: upload.</b> A user drops in a chest X-ray — here I'm using a real X-ray from a "
            "public medical dataset.",
            styles["MVScript"],
        )
    )
    story.append(cue("Drag-and-drop or click to upload one of the prepared demo X-rays now."))
    story.append(
        Paragraph(
            "<b>Step two</b> — and this happens instantly and invisibly — our quality gate checks "
            "whether this is even a valid chest X-ray. We found that color is a surprisingly reliable signal: "
            "a real chest X-ray is grayscale, even stored as a color image its red, green and blue channels are "
            "nearly identical. A real photograph's channels are wildly different. If someone accidentally "
            "uploads a photo instead of an X-ray, we catch it right here, before wasting any computation.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "<b>Step three:</b> the vision model runs and produces probability scores across 18 findings — "
            "you can see them appearing on screen now, ranked by probability.",
            styles["MVScript"],
        )
    )
    story.append(cue("Point at the Findings table as it populates."))
    story.append(
        Paragraph(
            "<b>Step four:</b> for every finding above a threshold, we generate this heatmap overlaid on the "
            "X-ray — the warm colors show exactly where the model is looking. This isn't decoration; it's "
            "a direct visual explanation.",
            styles["MVScript"],
        )
    )
    story.append(cue("Toggle the heatmap opacity slider and click a different finding row to swap the heatmap."))
    story.append(
        Paragraph(
            "<b>Step five,</b> and this is the heart of the project: our knowledge graph reasons over these "
            "findings. Watch what happens when I open this differential-diagnosis panel — you can see "
            "exactly which findings fed into each diagnosis, with what strength, and how much each one "
            "contributed. This is not a black box; every number here is fully auditable.",
            styles["MVScript"],
        )
    )
    story.append(cue("Expand a differential diagnosis to show the evidence-path chips."))
    story.append(
        Paragraph(
            "<b>Step six:</b> down here is our AI-generated report. Notice these small citation tags next to "
            "every sentence — F1, D2, K3, and so on. If I hover over one, it shows exactly which piece of "
            "evidence that sentence is based on. And up top, this grounding-score badge tells you what fraction "
            "of the report was actually verified against real evidence.",
            styles["MVScript"],
        )
    )
    story.append(cue("Hover a citation chip to show the tooltip; point at the grounding score badge."))
    story.append(
        Paragraph(
            "And that's the complete loop: upload, quality-check, detect, explain visually, reason "
            "transparently, narrate in plain language, and verify before showing anything. The whole thing "
            "runs in under thirty seconds on ordinary hardware, entirely locally — nothing leaves this "
            "machine.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "I'll now hand over to Manjunatha, who'll connect this back to the research landscape and wrap "
            "things up.",
            styles["MVScript"],
        )
    )

    story.append(PageBreak())

    # ---------------- Section 4: Gaps + Flowchart + Conclusion ----------------
    story.append(section_header(4, "Research Gaps, Workflow &amp; Conclusion", "V A Manjunatha", 4))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Thanks, Pradyumna. Before we close, I want to briefly place MEDVISION within the research "
            "landscape — this project didn't come out of nowhere, it directly addresses real gaps in "
            "existing work.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "Take contrastive models like CLIP, or medical variants like MedCLIP and BiomedCLIP — excellent "
            "at aligning images and text, but they stop at a similarity score, with no structured, auditable "
            "reasoning chain. Our knowledge graph is exactly that missing reasoning layer, sitting on top of a "
            "real classifier's output.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "Or take graph-based approaches like RadGraph, GCNs, and GATs — powerful, but the resulting "
            "graph embeddings are themselves black boxes, and they need training data we don't have access to. "
            "We took a more transparent path: a simple, deterministic weighted calculation — no hidden "
            "embeddings, fully inspectable.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "And for report generation, models like R2Gen or LLaVA-Med write fluent reports but carry a real "
            "hallucination risk. We built a dedicated verifier specifically to catch this, checking every "
            "generated sentence against real evidence before it ever reaches the user.",
            styles["MVScript"],
        )
    )
    story.append(Spacer(1, 4))
    gap_data = [[cell(h, bold=True) for h in GAP_ROWS[0]]]
    for row in GAP_ROWS[1:]:
        gap_data.append([cell(row[0], bold=True), cell(row[1]), cell(row[2])])
    gap_table = Table(gap_data, colWidths=[1.4 * inch, 2.1 * inch, 2.7 * inch])
    gap_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(gap_table)
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "The full literature survey, with all 18 papers we reviewed, is in our written project report — "
            "I'm just highlighting the three most direct comparisons here.",
            styles["MVScript"],
        )
    )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Let me quickly walk through our overall workflow on this diagram.",
            styles["MVScript"],
        )
    )
    story.append(cue("Point at each box in turn as you narrate the flow top to bottom."))
    story.append(Spacer(1, 6))
    story.append(WorkflowFlowchart(width=440, height=500))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "An X-ray comes in, passes our quality gate, and flows through two parallel stages: the vision "
            "classifier extracting findings, and our knowledge graph standing ready with medical relationships. "
            "These merge into calibration, then Grad-CAM heatmap generation, then the core knowledge-graph "
            "reasoning stage that produces ranked diagnoses. From there, our local language model narrates the "
            "evidence, checked continuously by our verifier, and finally everything is presented through our "
            "web interface. The one box in gray, dashed — that's retrieval and multimodal alignment, which "
            "we've deliberately scoped as future work, not yet built. We say that honestly rather than pretend "
            "it exists.",
            styles["MVScript"],
        )
    )
    story.append(Paragraph("Conclusion", styles["MVH2"]))
    story.append(
        Paragraph(
            "To conclude: MEDVISION demonstrates that explainable, locally-deployable clinical AI is achievable "
            "today, not just as a future promise. We have a fully working pipeline, verified against real data, "
            "with measured results — not projections. Our differential-diagnosis reasoning correctly "
            "identifies pneumonia over 86% of the time on real held-out cases. We found and fixed a real "
            "false-alarm problem through data-driven investigation. And every part of this system, from the "
            "heatmap to the final sentence of the report, can be explained and checked.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "There's still work ahead — DICOM support, deeper multimodal alignment, broader validation "
            "across more diseases — and we're continuing to build. But what we have today is a real, "
            "running, explainable diagnostic assistant, not a concept on paper.",
            styles["MVScript"],
        )
    )
    story.append(
        Paragraph(
            "Thank you all for your time. We're happy to take any questions.",
            styles["MVScript"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Anticipated questions (optional prep): ‘Why not just use ChatGPT/a cloud LLM?’ (privacy, "
            "no internet dependency, cost); ‘How do you know the knowledge graph weights are correct?’ "
            "(hand-authored from clinical teaching sources, one edge weight already corrected using real "
            "held-out evaluation data, and we've flagged clinician review as future validation); ‘What "
            "happens with a normal, healthy X-ray?’ (honest answer: our false-alarm rate is still ~38% on "
            "true negatives — a known, measured limitation we're actively working on, not hidden).",
            styles["MVCaption"],
        )
    )

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
