# Knowledge graph enrichment: MeSH + PrimeKG (T10)

Both external sources turned out to be directly, publicly downloadable — no
authentication or credentialing delay, unlike the MIMIC-CXR access process
noted elsewhere in this project. Total time: about half a day, most of it
spent confirming schemas and being conservative about what counts as a match,
not on data access.

## Scope

This enrichment only adds cross-references and corroborating edges to the
**existing 51 chest-relevant nodes** (19 Finding, 16 Anatomy, 16 Disease). It
does not expand what the system can analyze: the vision classifier
(`medvision/vision/classifier.py`) still only produces the same 18
torchxrayvision chest-X-ray labels, regardless of what the knowledge graph
contains. PrimeKG itself covers ~4M biomedical relationships across genes,
drugs, and thousands of diseases; only the disease-phenotype slice relevant
to our 16 diseases was extracted.

## MeSH: definition/id backfill (`scripts/enrich_mesh.py`)

28 of 51 nodes were `source: team` (no `mesh_id`). Rather than download the
~500MB MeSH descriptor XML for a handful of lookups, the script queries NLM's
free lookup API (`id.nlm.nih.gov/mesh/lookup/descriptor`) for an **exact**
match on each node's name and aliases, then fetches the matched descriptor's
preferred concept for its scope note.

Result: **5 matches**, all applied to `kg/ontology/anatomy.yaml`:

| Node | MeSH id | Matched term |
|---|---|---|
| A_PLEURA | D010994 | Pleura |
| A_MEDIASTINUM | D008482 | Mediastinum |
| A_DIAPHRAGM | D003964 | Diaphragm |
| A_RIBS | D000070602 | Rib Cage |
| A_TRACHEA | D014132 | Trachea |

The other 23 team-sourced nodes (all correctly) found no match — this is not
a gap. They fall into two groups:
- **Localization zone grid** (Right Upper Zone, Left Costophrenic Angle,
  etc.) — a coarse 3x2 grid this project defines for Grad-CAM zone
  assignment (`medvision/explain/localize.py`), not a real MeSH concept.
- **Composite/nonspecific findings** (Lung Opacity, Infiltration,
  Pneumonia-pattern Opacity) and **clinical-modifier diseases** (Malignant
  Pleural Effusion, Traumatic Rib Injury) — precise radiology-teaching
  phrasing that doesn't correspond to a single MeSH descriptor.

Existing hand-written `definition` text was kept as-is (it's already
radiology-specific and, on inspection, more precise for this use case than
the raw MeSH scope notes); only `mesh_id` was added and `source` flipped to
`mesh` per the schema convention already used for the other 30 nodes.

## PrimeKG: disease-phenotype corroboration (`scripts/enrich_primekg.py`)

Downloaded PrimeKG's full `kg.csv` (Harvard Dataverse,
doi:10.7910/DVN/IXA7BM, ~982MB, 8.1M edges) and filtered to
`disease_phenotype_positive` rows (~300K of the 8.1M).

Matching was deliberately **exact-name only** (case-insensitive, against
each node's `name` and `aliases`), not fuzzy. PrimeKG's disease taxonomy is
Mondo-based and far more granular than our 16 diseases (e.g. dozens of named
pneumonia subtypes), so loose matching would have silently linked concepts
that aren't really the same thing.

Result: **3 new edges**, all added as `associated_with` (weight 0.3,
`source: primekg`) in `kg/ontology/relations.yaml`:

- `F_EFFUSION -> D_SARCOIDOSIS` (PrimeKG: sarcoidosis ↔ Pleural effusion)
- `F_EMPHYSEMA -> D_SARCOIDOSIS` (PrimeKG: sarcoidosis ↔ Emphysema)
- `F_ATELECTASIS -> D_ASBESTOSIS` (PrimeKG: asbestosis ↔ Atelectasis — this
  one lines up with a well-known real finding, rounded atelectasis, which is
  a textbook radiographic sign of asbestos exposure)

**Why only 3 out of 300K rows**, honestly: PrimeKG's phenotype vocabulary is
HPO (Human Phenotype Ontology), built to describe rare/genetic disease
presentations (e.g. "recurrent pneumonia" as a feature of an
immunodeficiency syndrome). It has very little overlap with the
radiographic-sign vocabulary our findings use (Consolidation, Lung Opacity,
etc.), which describes what a chest X-ray *looks like*, not a syndromic
phenotype. This is a real result of the data, not a bug in the matching —
PrimeKG is the wrong shape of resource to bulk-expand this specific KG's
finding-disease edges; it's useful as a small independent corroboration
check, which is what it was used for.

## Deliberately not done: wiring `associated_with` into scoring

`medvision/kg/reasoning.py:score_differentials()` only walks `suggestive_of`
and `contradicts` edges. The 3 new `associated_with` edges exist in the graph
(available for future display/explanation use) but do **not** change any
diagnosis score or ranking. Folding them into the scoring formula would
silently change the Recall@1 / false-alarm numbers already measured and
written up in `reports/eval_report.md` and the project presentation
materials — that requires its own deliberate re-evaluation pass, the same
discipline applied to the `F_INFILTRATION -> D_PNEUMONIA` weight change in
`reports/false_alarm_investigation.md`, not a silent side effect of a data
enrichment ticket.

## Verification

- `uv run python scripts/build_kg.py` — validator passes: 19 Finding /
  16 Anatomy / 16 Disease / 69 edges (was 66).
- `uv run pytest -q` — 91/91 passing.
- `uv run ruff check . && uv run ruff format --check .` — clean.

## Reproducing

```bash
uv run python scripts/enrich_mesh.py          # writes reports/mesh_enrichment_candidates.json
curl -sL "https://dataverse.harvard.edu/api/access/datafile/6180620" -o /tmp/primekg_kg.csv
uv run python scripts/enrich_primekg.py /tmp/primekg_kg.csv   # writes reports/primekg_enrichment_candidates.json
```

Both scripts are dry-run only (candidate files, no YAML writes) by design —
matches were reviewed by hand before being added to `kg/ontology/*.yaml`,
the same review-before-apply discipline used for the calibration acceptance
policy elsewhere in this project.
