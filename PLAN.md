# MEDVISION — Implementation Plan (v2)

> Knowledge-Graph Anchored Vision-Reasoning Framework for Explainable Clinical Diagnosis
> Team B53, SIT Tumakuru. Plan written 2026-09-17 and meant to be carried out ticket by ticket (Sonnet codes, Opus reviews).
> **Research prototype. Not for clinical use.**

---

## 0. Review of the Phase-II report: what is solid and what is missing

### Solid
- Framing diagnosis as grounded reasoning rather than bare classification.
- Keeping perception, knowledge, retrieval and reasoning as separate modules (easy to ablate and debug).
- Running inference locally for privacy, and the choice of stack (PyTorch, PyG, ChromaDB, local LLM, FastAPI).

### Gaps that would block implementation (fixed in this plan)

| # | Gap in report | Why it matters | Fix in this plan |
|---|---|---|---|
| G1 | InfoNCE aligns image `v_i` with graph `g_i`, but **the report never defines what the matched graph `g_i` is** for an image. | Without defined positive pairs, contrastive training can't be run. | `g_i` = embedding of the **report-derived subgraph** for image *i* (the findings and anatomy mentioned in its radiology report, mapped onto KG nodes). See §4.3. |
| G2 | No source for the knowledge graph. | UMLS needs a licence. RadGraph and MIMIC need PhysioNet credentialing, which takes weeks. | Hand-curated chest-X-ray ontology (YAML, versioned), enriched with **PrimeKG** (open) and **MeSH** scope notes (public domain). See §4.2. |
| G3 | The report depends on MIMIC-CXR, which requires credentialing. | Weeks of delay, and the approval may not come in time. | Use freely available **IU X-Ray (Open-i)**, **NIH ChestX-ray14** and **RSNA Pneumonia** first. Treat MIMIC-CXR-JPG as a stretch goal and **apply on PhysioNet today** (CITI course). |
| G4 | The LLM does both the diagnosis and the narration. | That is a hallucination risk and the opposite of explainable. | The diagnosis is computed **deterministically** by scoring findings along KG paths. The LLM **only narrates** structured evidence and must cite evidence IDs. A **verifier** then checks every sentence (§4.6). |
| G5 | No evaluation protocol, baselines or ablations. | Examiners will ask "better than what?" | Full evaluation harness (§6): AUROC, retrieval Recall@k, pointing game, grounding score, hallucination rate, latency, and 4 ablations. |
| G6 | No uncertainty handling and no out-of-distribution (OOD) rejection. | Uploading a selfie would still produce a "diagnosis". | Quality/OOD gate, temperature-scaled calibration, and abstention below a set threshold. |
| G7 | Inconsistent UI choice: the abstract says Streamlit, Table 3.1 and Appendix C say React. | Inconsistency in the report. | **React (Vite + TS)**. Fix the abstract. |
| G8 | Grad-CAM heatmaps don't say *where* in clinical terms. | "Blob on image" is weak as an explanation. | Grad-CAM combined with **anatomical segmentation** gives statements like "opacity, right lower zone" that link to anatomy nodes in the KG. |
| G9 | Report nits | — | The survey claims "22 studies", but refs [19], [23], [25]–[28] are not in Table 2.1. The timeline table is numbered 1.1 inside Appendix A. Fix these before final submission. |

### "Make it better": upgrades beyond the report
1. **Concept bottleneck anchoring.** The classifier outputs probabilities over **KG finding nodes**, not arbitrary labels, so every prediction is already a graph node.
2. **Transparent KG path reasoning.** Differential diagnoses come with explicit evidence paths, e.g. `Consolidation(0.87) —suggestive_of(w=0.8)→ Pneumonia`.
3. **Anatomy-aware localisation.** Grad-CAM mass is overlapped with lung-zone segmentation.
4. **Case-based retrieval.** Shows the top-k most similar prior cases, with their reports, from the aligned embedding space.
5. **Grounding verifier and grounding score** for each generated report, plus a regenerate-once policy when the score is low.
6. **Calibration, abstention and OOD gate.**
7. **DICOM input** with de-identification of tags.
8. **Clinician feedback capture** (agree/disagree per finding), logged for later analysis.
9. **PDF report export**, a model card, and one-command Docker Compose startup.

---

## 1. Final architecture

```
                ┌───────────────────────── React UI (Vite+TS) ─────────────────────────┐
                │ upload · heatmap overlay · findings · KG path graph · report+citations │
                └───────────────▲──────────────────────────────────────────┬────────────┘
                                │ JSON                                      │ multipart
                        ┌───────┴───────────── FastAPI (backend/) ─────────▼────────┐
                        │ /analyze orchestrator · SQLite log · static heatmaps       │
                        └──┬───────────┬────────────┬──────────────┬────────────┬───┘
                           │           │            │              │            │
   (1) Input/QC      (2) Vision   (3) Knowledge  (4) Retrieval  (5) Reasoning  (6) Verify
   DICOM/PNG load    classifier    KG subgraph    ChromaDB:      KG path       claim ↔
   normalise         over KG       + GAT embed    kg_nodes,      scoring →     evidence
   OOD gate          findings      dx path        snippets,      LLM narrates  check,
                     Grad-CAM +    scoring        cases          (Ollama,      grounding
                     anatomy seg                  (aligned       JSON schema)  score
                     img embed ────► projection ► space)
```

### Inference flow (one request)
1. Load the image (PNG/JPG/DICOM), de-identify, resize to 224/512, and run the **QC/OOD gate** (reject if not a frontal CXR).
2. **Vision:** calibrated probabilities for about 14–18 finding nodes, a Grad-CAM map for each positive finding, anatomy masks, and a 512-d image embedding.
3. **Localise:** overlap each heatmap with the anatomy masks to get a `located_in` region per finding.
4. **KG reasoning:** activate finding nodes with p ≥ τ, extract the 2-hop subgraph, and score candidate diagnoses: `score(dx) = Σ_f p(f)·w(f→dx) − Σ_f p(f)·w_contra(f→dx)`. Normalise and keep the top 3, with their paths.
5. **Retrieval:** project the image embedding into the aligned space and query ChromaDB for (a) top-k KG nodes (open-vocabulary hints), (b) top-k knowledge snippets, and (c) top-k similar cases.
6. **Reasoning:** build a prompt that contains only the structured evidence, each item with an ID (`F:`, `K:`, `S:`, `C:`). The LLM returns JSON: findings text, impression, differentials, limitations, and `evidence_ids` for each sentence.
7. **Verify:** check that the cited IDs exist, that mentioned findings are consistent with the probabilities (no "no effusion" when p=0.9), and compute the grounding score. If it is below 0.8, regenerate once with the verifier feedback.
8. Persist everything to SQLite (FR8) and return an `AnalysisResult`.

---

## 2. Tech stack (pinned choices)

| Layer | Choice | Notes |
|---|---|---|
| Python | 3.11, **uv** for envs | `uv sync` |
| Vision baseline | **torchxrayvision** `densenet121-res224-all` | Pretrained, 18 pathologies, no training on day 1. Also ships the `PSPNet` chest anatomy segmenter. |
| Image embedding / text space | **BiomedCLIP** (`open_clip`, `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) | Image and text share one space, so KG node text and snippets embed in the same space. |
| Explainability | `pytorch-grad-cam`, Captum (Integrated Gradients for the audit notebook) | |
| Graph | NetworkX (authoring), **PyTorch Geometric** `HeteroData` + `HGTConv`/`GATv2Conv` | Graph is small (<5k nodes), so train on CPU. |
| Vector DB | **ChromaDB** (persistent, local) | 3 collections |
| LLM | **Ollama**, model set in config (start with `llama3.1:8b`; also try `qwen2.5:7b` and a medical model if available) | Use `format=<json schema>` for structured output |
| API | FastAPI, Pydantic v2, SQLModel (SQLite) | |
| UI | React 18 + Vite + TypeScript + Tailwind; `cytoscape.js` for KG paths | |
| PDF | WeasyPrint (or ReportLab) | |
| Config | `pydantic-settings` + `configs/*.yaml` | |
| Tracking | Weights & Biases or MLflow (local) | |
| Training compute | Kaggle/Colab GPU for fine-tuning and contrastive runs. Mac (MPS/CPU) for inference. | |
| Packaging | Docker Compose (api, ui, ollama) | |

---

## 3. Repository layout

```
medvision/
├── CLAUDE.md                  # coding conventions for the AI coder
├── PLAN.md                    # this file
├── pyproject.toml             # uv workspace
├── configs/
│   ├── default.yaml           # thresholds, model names, paths, k values
│   └── labels.yaml            # classifier label -> KG node id map
├── data/                      # gitignored: raw/, processed/, chroma/, kg/
├── kg/
│   ├── ontology/              # hand-authored YAML (versioned!)
│   │   ├── findings.yaml
│   │   ├── anatomy.yaml
│   │   ├── diseases.yaml
│   │   └── relations.yaml
│   └── schema.md
├── medvision/                 # python package (ML core, no web code)
│   ├── config.py
│   ├── io/                    # dicom.py, image.py, deid.py
│   ├── qc/                    # ood.py, quality.py
│   ├── vision/                # classifier.py, embedder.py, calibrate.py, train.py
│   ├── explain/               # gradcam.py, anatomy.py, localize.py
│   ├── kg/                    # build.py, validate.py, enrich_primekg.py, mesh.py, graph.py, encoder.py, reasoning.py
│   ├── data/                  # iu_xray.py, nih.py, rsna.py, pairs.py, entity_linking.py
│   ├── align/                 # heads.py, losses.py, train.py
│   ├── retrieval/             # index.py, search.py
│   ├── reasoning/             # prompt.py, llm.py, schema.py, verifier.py
│   ├── pipeline.py            # end-to-end analyze(image) -> AnalysisResult
│   └── eval/                  # classification.py, retrieval.py, localization.py, grounding.py, latency.py, ablations.py
├── backend/
│   └── app/  main.py, api/v1/*.py, schemas.py, db.py, storage.py, pdf.py
├── frontend/                  # vite react ts
├── scripts/                   # download_*.sh, build_kg.py, index_chroma.py, train_*.py, run_eval.py
├── notebooks/                 # EDA, audits (not imported by code)
├── tests/                     # pytest; small fixtures in tests/fixtures
└── docker-compose.yml
```

---

## 4. Module specifications

### 4.1 Input, QC and OOD (`medvision/io`, `medvision/qc`)
- Accept `.png/.jpg/.jpeg/.dcm` up to 20 MB. For DICOM: `pydicom`, apply VOI LUT, handle MONOCHROME1 inversion, and **strip PHI tags** before anything is stored.
- Normalise to the torchxrayvision range [-1024, 1024] for the classifier and to BiomedCLIP preprocessing for the embedder.
- OOD gate, v1: cosine similarity of the BiomedCLIP image embedding to the text prompts "a frontal chest X-ray" and "a photograph", plus a Mahalanobis distance to the training-set embedding centroid. Threshold tuned on a held-out mix (CXR vs. ImageNet/other-modality images). Output `{is_cxr, score, reason}`.
- Quality flags: too dark or bright, low resolution, likely lateral view.

### 4.2 Knowledge graph (`kg/`, `medvision/kg`)
**Node types:** `Finding`, `Anatomy`, `Disease`, `Device` (tubes/lines), `Sign`.
**Edge types (typed, weighted):** `located_in`, `suggestive_of (w∈[0,1])`, `contradicts`, `associated_with`, `differential_of`, `complication_of`, `part_of` (anatomy hierarchy), `synonym_of` (kept as node aliases, not edges).

Every node in YAML:
```yaml
- id: F_CONSOLIDATION
  type: Finding
  name: Consolidation
  aliases: [airspace opacity, airspace disease]
  definition: "..."          # from MeSH scope note, or written by the team
  mesh_id: D011014            # optional
  source: mesh|team|primekg
```
Every edge: `{src, rel, dst, weight, source, note}`. **Weights and sources are mandatory**, since they are what makes the explanation traceable.

Build steps:
1. Hand-author about 150–300 nodes: the CheXpert-14 findings plus extras (e.g. nodule, mass, fibrosis, pleural thickening, hernia), about 25 anatomy regions (lungs → zones/lobes, pleura, mediastinum, heart, diaphragm, costophrenic angles), and about 30–40 thoracic diseases.
2. `enrich_primekg.py`: filter PrimeKG to disease↔phenotype edges for the curated diseases and add them as `associated_with` edges (weight 0.3, source=primekg).
3. `mesh.py`: fill in missing definitions from the MeSH XML (public domain).
4. `validate.py`: unique IDs, no dangling edges, weights within range, every finding in `labels.yaml` exists, every disease reachable from at least one finding. Run it in CI.
5. `build.py`: YAML → NetworkX → `data/kg/graph.json` + PyG `HeteroData` (`data/kg/graph.pt`).
6. **The team reviews edge weights.** If possible, ask a doctor (college hospital contact) to spot-check 50 edges and record this in the report.

### 4.3 Entity linking and training pairs (`medvision/data`) — this fixes G1
For each (image, report) pair:
- **IU X-Ray:** use the provided MeSH/"Problems" tags, plus rule-based extraction from the report with negation handling (`negspacy`/NegEx) against the KG alias dictionary.
- **MIMIC-CXR-JPG (stretch):** CheXpert labels + RadGraph.
- Output: `pairs.parquet` with columns `image_path, report_text, pos_nodes[], neg_nodes[], anatomy_nodes[], split`.
- The **positive graph** `g_i` is the induced subgraph over `pos_nodes ∪ anatomy_nodes` plus 1-hop `suggestive_of` neighbours. "Normal" studies map to the `F_NO_FINDING` node.
- Splits are **by patient**, never by image.

### 4.4 Vision module (`medvision/vision`, `medvision/explain`)
- **Stage A (day 1):** torchxrayvision DenseNet, mapped to KG node IDs via `labels.yaml`. This is the baseline and the fallback.
- **Stage B:** fine-tune on NIH-14 + IU (multi-label BCE, class-balanced), with the head outputs named by KG node. Keep it only if it beats Stage A on the validation AUROC.
- **Calibration:** per-label temperature scaling on the validation set. Report ECE before and after. Two thresholds per label: `τ_pos` (report as present) and `τ_uncertain` (report as "possible").
- **Grad-CAM** on the last dense block for each positive finding, saved as a PNG overlay. **Anatomy:** torchxrayvision PSPNet masks, grouped into zones (R/L × upper/mid/lower, plus heart and mediastinum). `localize.py` assigns each finding the zone with the highest share of Grad-CAM energy (and reports that share).
- **Embedder:** frozen BiomedCLIP image tower → 512-d.

### 4.5 Graph encoder and alignment (`medvision/kg/encoder.py`, `medvision/align`)
- Node input features: BiomedCLIP **text** embedding of `name + definition` (512-d), plus a learned type embedding.
- Encoder: 2-layer heterogeneous GNN (`HGTConv` or `GATv2Conv` per relation) → 256-d. Pre-train it with link prediction on the KG (80/10/10 edge split, report AUC).
- Subgraph readout: attention pooling over the nodes of `g_i` → `z_g`.
- Image head: MLP 512→256 → `z_v`.
- Loss: symmetric InfoNCE, learnable τ (initialised at 0.07). **Handle false negatives:** many studies share the same subgraph (especially "normal"), so use a **multi-positive (SupCon-style) target** in which every batch item with an identical `pos_nodes` set counts as a positive.
- Optional auxiliary loss: node-level alignment, i.e. `z_v` against each positive node embedding (a GLoRIA-like local term).
- Training: freeze BiomedCLIP; train the heads and the GNN. Batch 256, 30–50 epochs, on a Kaggle GPU.
- Output artefacts: `align_heads.pt`, `gnn.pt`, and precomputed `z_node` for all KG nodes.
- **Fallback:** if alignment is weak, the system still works through the concept-bottleneck path, and the ablation (§6) measures what alignment adds.

### 4.6 Retrieval, reasoning and verification
**ChromaDB collections** (cosine similarity):
| collection | contents | vector |
|---|---|---|
| `kg_nodes` | every KG node + definition | aligned `z_node` (256-d) |
| `snippets` | chunked definitions and team-written teaching notes for each disease (≤200 tokens, with `node_ids` metadata) | BiomedCLIP text (512-d) |
| `cases` | training-split studies (image path, report, labels) | aligned `z_v` (256-d) |

Query with the image's `z_v` (for kg_nodes and cases) and with BiomedCLIP text for "findings: …" (for snippets). Also restrict snippets with a metadata filter on the activated node IDs, then take the union and re-rank.

**Deterministic diagnosis scoring** (`kg/reasoning.py`): as in §1 step 4. Return `paths: [[node, rel, weight], ...]` for each differential. This, not the LLM, is the source of truth for differentials.

**Prompt** (`reasoning/prompt.py`): the system prompt fixes the role (radiology assistant draft for clinician review), forbids facts that aren't in the evidence, and requires every sentence to carry `evidence_ids`. The user prompt is a JSON evidence block:
```json
{"findings":[{"id":"F1","node":"F_EFFUSION","label":"Pleural effusion","p":0.91,"status":"present","region":"left lower zone"}],
 "differentials":[{"id":"D1","node":"D_CHF","score":0.62,"path_ids":["K3","K7"]}],
 "kg_facts":[{"id":"K3","text":"Pleural effusion suggestive_of Congestive heart failure (w=0.7)"}],
 "snippets":[{"id":"S1","text":"..."}],
 "similar_cases":[{"id":"C1","impression":"...","similarity":0.83}],
 "quality":{"flags":[]}}
```
**Output schema** (`reasoning/schema.py`, passed to Ollama `format`):
```json
{"findings":[{"text":str,"evidence_ids":[str]}],
 "impression":[{"text":str,"evidence_ids":[str]}],
 "differentials":[{"diagnosis":str,"rationale":str,"evidence_ids":[str]}],
 "limitations":[str],
 "recommendation":str}
```
LLM settings: temperature 0.1, fixed seed, and a timeout. Retry on a JSON parse failure. If the LLM is unreachable, return a **template-based report** built from the same evidence, so the demo never breaks.

**Verifier** (`reasoning/verifier.py`) runs these checks in order:
1. Every cited ID exists in the evidence block.
2. Every sentence has at least one ID; sentences without one are flagged.
3. Ontology-alias scan of each sentence: a mentioned finding must be `present/possible` in the evidence, or be explicitly negated with p < τ_neg. Anything else is a contradiction.
4. Differential names must match the deterministic top-3 (via aliases).

`grounding_score = supported_sentences / total`. Output: `{score, unsupported:[...], contradictions:[...]}`.

### 4.7 Backend API (`backend/app`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | model, LLM and chroma status |
| POST | `/api/v1/analyze` | multipart `file`, optional `options` JSON → `AnalysisResult` |
| GET | `/api/v1/analyses` | paginated history |
| GET | `/api/v1/analyses/{id}` | full result |
| GET | `/api/v1/analyses/{id}/heatmap/{node_id}.png` | overlay image |
| GET | `/api/v1/analyses/{id}/report.pdf` | PDF export |
| POST | `/api/v1/analyses/{id}/feedback` | `{node_id, verdict: agree/disagree, comment}` |
| GET | `/api/v1/kg/nodes/{id}` | node detail and neighbours |
| GET | `/api/v1/kg/subgraph?ids=...` | for visualisation |

`AnalysisResult` (Pydantic): `id, created_at, image{w,h,modality,view}, qc{is_cxr,score,flags}, findings[{node_id,label,prob,calibrated_prob,status,region,region_share,heatmap_url}], differentials[{node_id,label,score,paths}], retrieval{kg_nodes,snippets,cases}, report{...schema above}, verification{score,unsupported,contradictions}, timings_ms{qc,vision,kg,retrieval,llm,verify,total}, model_versions{...}, disclaimer`.

Models are loaded once at startup (lifespan) and inference runs in a threadpool. SQLite tables: `analyses`, `feedback`. Uploaded files are stored under `data/uploads/{id}` after de-identification.

### 4.8 Frontend (`frontend/`)
Pages: **Analyze** (drag-drop upload → result view), **History**, **KG Explorer**, **About/Model card**.
The result view has:
- Left panel: image with a heatmap overlay toggle per finding and an opacity slider.
- Centre panel: a findings table (prob bar, status chip, region) and a differentials list; clicking one shows its KG path in Cytoscape.
- Right panel: the report. Each sentence shows citation chips; hovering a chip highlights the evidence item, and unsupported sentences are shown in amber. There is also a grounding-score badge and a similar-cases carousel.
- Footer: timings, model versions, and a permanent **"Research prototype — not for clinical use"** banner.
- Feedback 👍/👎 on each finding.

---

## 5. Datasets and access (start today)

| Dataset | Access | Use |
|---|---|---|
| IU X-Ray / Open-i (≈7.4k images, 3.9k reports) | free (Kaggle mirror) — check licence terms | image–report pairs, alignment, report evaluation |
| NIH ChestX-ray14 (112k, 14 labels, ~1k bboxes) | free | classifier fine-tuning, pointing-game evaluation |
| RSNA Pneumonia (bboxes) | free (Kaggle) | localisation evaluation |
| CheXpert-small | free registration (Stanford AIMI) | external validation |
| MIMIC-CXR-JPG + RadGraph | **PhysioNet credentialing (apply now)** | stretch: larger alignment set, RadGraph entities |
| PrimeKG | free (Harvard Dataverse) | KG enrichment |
| MeSH XML | free (NLM) | definitions and aliases |

Don't commit data. `scripts/download_*.sh` documents each source, and `data/README.md` records licences.

---

## 6. Evaluation plan (`medvision/eval`, `scripts/run_eval.py`)

| Aspect | Metric | Data |
|---|---|---|
| Classification | per-label AUROC, macro-AUROC, F1 at τ_pos, **ECE** before/after calibration | NIH test (patient split), CheXpert val (external) |
| Localisation | **pointing game** accuracy, IoU of the thresholded CAM | NIH bbox subset, RSNA |
| Explanation faithfulness | deletion/insertion AUC (Grad-CAM vs. random) | 200-image sample |
| KG encoder | link-prediction AUC | KG edge split |
| Alignment and retrieval | image→subgraph Recall@1/5/10, image→case R@k with label-match precision | IU test |
| Report quality | BLEU-4, ROUGE-L, BERTScore; **CheXbert-label F1** if the weights are available | IU test |
| Grounding | grounding score, contradiction rate, **hallucination rate** (manual audit of 50 reports) | IU test |
| OOD | AUROC CXR vs non-CXR | mixed set |
| System | p50/p95 latency per stage, peak RAM | Mac M-series |
| Human (optional) | 3–5 clinicians rate 20 cases on a 1–5 scale for correctness, usefulness and trust | — |

**Ablations** (the core result table for the paper):
1. Vision-only (label list + template report)
2. + LLM narration without evidence (ungrounded; expect hallucinations)
3. + KG path reasoning
4. + retrieval (snippets + cases)
5. Full (+ alignment + verifier)

Targets: don't promise absolute numbers. The claim to support is **"full ≥ baselines on AUROC, with a large gain in grounding score and a lower hallucination rate"**.

---

## 7. Timeline (today = Thu 17 Sep 2026 → end of November, about 10.5 weeks)

| Week | Dates | Milestone |
|---|---|---|
| W1 | Sep 17–23 | **P0 Walking skeleton.** Repo, uv, configs, download scripts, PhysioNet application, torchxrayvision → FastAPI `/analyze` → bare React page showing probabilities and one Grad-CAM. |
| W2 | Sep 24–30 | **P1a.** KG ontology v1 (YAML, validator, builder), `labels.yaml`, QC/OOD gate, DICOM + de-id, anatomy localisation. |
| W3 | Oct 1–7 | **P1b.** IU/NIH loaders, entity linking + `pairs.parquet`, calibration, Stage-B fine-tune started. KG enrichment (PrimeKG, MeSH). |
| W4 | Oct 8–14 | **P2a.** GNN encoder + link-prediction pre-training; alignment heads + multi-positive InfoNCE training. |
| W5 | Oct 15–21 | **P2b.** ChromaDB indexing (3 collections), search API, deterministic dx path scoring. |
| W6 | Oct 22–28 | **P3.** Prompt, Ollama client, schema, template fallback, verifier, full `pipeline.py`, all API endpoints, SQLite log. |
| W7 | Oct 29–Nov 4 | **P4.** Full React UI (overlay, KG path view, cited report, history, feedback, PDF). |
| W8 | Nov 5–11 | **P5a.** Evaluation harness + ablations + latency. Freeze models. |
| W9 | Nov 12–18 | **P5b.** Human rating study (optional), error analysis, model card, Docker Compose, README. |
| W10 | Nov 19–25 | Report chapters 5–7 (implementation, results), 4-page IEEE paper, poster, video. |
| buffer | Nov 26–30 | Fixes and demo rehearsal |

### Work split (maps to report §A.2)
- **Member A — Knowledge & Graph:** §4.2, 4.3 entity linking, 4.5 GNN, dx path scoring.
- **Member B — Vision:** §4.1, 4.4, localisation, calibration, and classification/localisation evaluation.
- **Member C — Fusion & Retrieval & Reasoning:** alignment training, §4.6, verifier, retrieval/grounding evaluation.
- **Member D — Interface & Deployment:** §4.7, 4.8, PDF, Docker, latency evaluation, demo.
- **Everyone:** tests, KG edge review, the 50-report hallucination audit, and writing.

---

## 8. Ticket backlog (hand these to Sonnet one at a time)

Each ticket should be implemented, tested (`pytest` must pass), and kept small. **Acceptance** is the definition of done.

### P0 — Walking skeleton
- **T01 Scaffold repo.** uv `pyproject.toml` (python 3.11), package layout from §3, ruff + pytest + pre-commit, `.gitignore` (data/, *.pt, .env), `configs/default.yaml`, `medvision/config.py` (pydantic-settings, YAML + env override). *Acceptance:* `uv run pytest` passes a smoke test, and `uv run ruff check` is clean.
- **T02 Download scripts.** `scripts/download_iu_xray.sh`, `download_nih.sh` (sample mode: 5k images), `download_rsna.sh`, `data/README.md` with licences. *Acceptance:* idempotent, and prints where the files landed.
- **T03 Baseline classifier.** `vision/classifier.py` wraps torchxrayvision densenet `all`; `predict(image) -> dict[node_id, prob]` using `labels.yaml`; picks the device automatically (mps/cuda/cpu). *Acceptance:* test on a fixture CXR returns probabilities in [0,1] for all mapped nodes.
- **T04 Grad-CAM.** `explain/gradcam.py`: `heatmap(image, node_id) -> np.ndarray` plus `overlay_png(...)`. *Acceptance:* the PNG is saved, and the heatmap shape equals the input shape.
- **T05 Minimal API.** FastAPI app, `/health`, and `/analyze` returning findings plus heatmap URLs; models loaded in the lifespan. *Acceptance:* `httpx` test uploads the fixture and gets 200 with findings.
- **T06 Minimal UI.** Vite React TS + Tailwind; upload → findings table → heatmap image. Dev proxy to the API. *Acceptance:* `npm run build` succeeds, and the manual demo works.

### P1 — Input, vision, KG
- **T07 Image IO + DICOM + de-id** (§4.1). *Acceptance:* tests for PNG, JPG, a MONOCHROME1 DICOM (use the pydicom test data), and PHI tags removed.
- **T08 QC/OOD gate** (§4.1) with a threshold from the config and `scripts/tune_ood.py`. *Acceptance:* rejects a non-CXR fixture and accepts the CXR fixture.
- **T09 KG YAML schema + validator + builder** (§4.2 steps 4–5), plus a starter ontology (≥ 60 nodes; the team expands it later). *Acceptance:* `scripts/build_kg.py` writes graph.json/graph.pt, and the validator tests catch a dangling edge and a bad weight.
- **T10 KG enrichment** (PrimeKG filter, MeSH definitions). *Acceptance:* node count grows, sources are tagged, and the validator still passes.
- **T11 Anatomy localisation** (§4.4). *Acceptance:* `localize(heatmap, image) -> (zone_node_id, share)`; a synthetic heatmap test places mass in the expected zone.
- **T12 Dataset loaders + patient-level splits** (IU, NIH, RSNA). *Acceptance:* no patient appears in two splits (tested).
- **T13 Entity linking + pairs.parquet** (§4.3) with NegEx. *Acceptance:* unit tests on 10 hand-written report sentences (including negations), and the parquet has the documented schema.
- **T14 Calibration** (temperature scaling, ECE, thresholds persisted to `data/models/calibration.json`). *Acceptance:* ECE after ≤ ECE before on validation, and the calibration is used by `predict`.
- **T15 Stage-B fine-tuning script** (`scripts/train_classifier.py`, Kaggle/Colab friendly, W&B/MLflow logging, checkpoint selection by val macro-AUROC). *Acceptance:* runs 1 epoch on the sample subset locally.

### P2 — Graph encoder, alignment, retrieval
- **T16 BiomedCLIP embedder** (image + text, batched, cached to disk). *Acceptance:* 512-d L2-normalised outputs.
- **T17 Hetero GNN encoder + link-prediction pre-training** (§4.5). *Acceptance:* the script reports test AUC and saves `gnn.pt`.
- **T18 Alignment heads + multi-positive InfoNCE + training script.** *Acceptance:* a loss unit test on a toy batch with duplicate positives, and training on 500 pairs lowers the loss.
- **T19 ChromaDB indexer** for the 3 collections (§4.6), with rebuild and version metadata. *Acceptance:* `scripts/index_chroma.py` is idempotent, and the counts match.
- **T20 Search service** (image→nodes, image→cases, text→snippets with node filter, merge/re-rank). *Acceptance:* top-k results carry IDs, scores and metadata.
- **T21 Deterministic dx path scoring** (§1 step 4). *Acceptance:* a toy-graph test with known scores, including a `contradicts` penalty, and returned paths.

### P3 — Reasoning and pipeline
- **T22 Evidence builder + prompt + JSON schema** (§4.6). *Acceptance:* a snapshot test of the prompt for a fixed evidence block.
- **T23 Ollama client** (timeout, retries, JSON-schema format, seed) + **template fallback report**. *Acceptance:* with the LLM mocked-down, the fallback is returned and `report.source="template"`.
- **T24 Verifier** (§4.6 checks 1–4, grounding score). *Acceptance:* tests for an uncited sentence, a fake ID, a contradiction ("no effusion" at p=0.9), and a wrong differential.
- **T25 `pipeline.analyze()`** with the full flow, per-stage timings, regenerate-once when the score is < 0.8, and `AnalysisResult`. *Acceptance:* an end-to-end test with the LLM mocked.
- **T26 Full API + SQLite + feedback + KG endpoints + PDF** (§4.7). *Acceptance:* an API test for every endpoint.

### P4 — UI
- **T27 Result view** (3-panel layout, heatmap toggle/opacity, findings table, differentials).
- **T28 KG path visualisation** (cytoscape), shown for the selected differential and finding.
- **T29 Cited report panel** (citation chips, hover highlights evidence, unsupported sentences in amber, grounding badge, similar cases).
- **T30 History, feedback, PDF download, About/model card page, disclaimer banner.** *Acceptance for T27–T30:* builds cleanly, has no TS errors, and passes a manual demo checklist in `docs/demo.md`.

### P5 — Evaluation and delivery
- **T31 Eval harness:** classification, localisation, retrieval, report, grounding, OOD and latency modules; `scripts/run_eval.py --ablation {1..5}` writes CSV + markdown tables to `reports/`.
- **T32 Faithfulness audit notebook** (deletion/insertion, Integrated Gradients comparison).
- **T33 Docker Compose** (api, ui via nginx, ollama with the model pulled on first start) + `make demo`.
- **T34 Docs:** README (setup in ≤ 5 commands), model card, data card, limitations and ethics section.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| PhysioNet access is late | The plan never depends on it; IU + NIH are enough. |
| Alignment gives weak Recall@k | The concept-bottleneck path carries the system; report the ablation honestly and try the node-level auxiliary loss or unfreezing the last ViT block. |
| The local LLM is slow on a laptop | Use a 7–8B Q4 model and a short evidence block; cap output tokens; stream the response to the UI; the template fallback is always available. |
| LLM invents facts | JSON schema + mandatory citations + verifier + regenerate + amber highlighting. |
| KG edge weights look arbitrary to examiners | Record a source for every edge, get a doctor to spot-check, and run a sensitivity analysis (perturb weights ±20% and measure how much the top-1 dx changes). |
| PyG problems on MPS | Train the GNN on CPU (the graph is small) or on Kaggle. |
| Dataset label noise (NIH) | Say so in the limitations; validate externally on CheXpert. |
| Ethics | De-identification, local-only processing, disclaimer everywhere, no real patient data outside the public datasets. |

---

## 10. How to work with Sonnet
1. Open a Sonnet session in this folder. It reads `CLAUDE.md` automatically.
2. Prompt: *"Implement ticket T0X from PLAN.md. Follow CLAUDE.md. Write tests. Stop when the acceptance criteria pass and summarise what changed."*
3. One ticket per session or commit. Commit after each ticket (`git init` first).
4. Bring hard design questions back to Opus, e.g. the loss for T18, or the verifier for T24 if it misbehaves, and after every phase ask Opus to review the diff.
