# MEDVISION Knowledge Graph Schema

Source of truth: hand-authored YAML files in `kg/ontology/`. Generated artefacts
(`data/kg/graph.json`, `data/kg/graph.pt`) are built from these by
`scripts/build_kg.py` and must never be hand-edited.

## Files

| File | Node types |
|---|---|
| `findings.yaml` | `Finding` — radiographic signs (classifier outputs + F_NO_FINDING) |
| `anatomy.yaml` | `Anatomy` — chest regions/zones used for localization |
| `diseases.yaml` | `Disease` — differential diagnoses |
| `relations.yaml` | edges only, referencing node ids from the files above |

## Node fields

- `id` (required, unique, `TYPE_PREFIX_NAME` convention: `F_`, `A_`, `D_`)
- `type` (required: `Finding` \| `Anatomy` \| `Disease`)
- `name` (required, human-readable)
- `aliases` (optional list, used for entity linking in report text, T13)
- `definition` (required, one sentence)
- `mesh_id` (optional, MeSH descriptor id if sourced from MeSH)
- `source` (required: `mesh` \| `primekg` \| `team`)

## Edge fields

- `src`, `dst` (required, must reference existing node ids)
- `rel` (required, one of: `suggestive_of`, `contradicts`, `differential_of`,
  `complication_of`, `located_in`, `part_of`, `associated_with`)
- `weight` (required, float in `[0, 1]`)
- `source` (required: `mesh` \| `primekg` \| `team`)
- `note` (optional, free text — why this edge exists)

## Relation semantics

| rel | src type | dst type | meaning |
|---|---|---|---|
| `suggestive_of` | Finding | Disease | positive evidence toward a diagnosis |
| `contradicts` | Finding | Disease | negative evidence against a diagnosis |
| `differential_of` | Disease | Disease | commonly-confused / competing diagnoses |
| `complication_of` | Disease | Disease | one disease can progress into another |
| `located_in` | Finding | Anatomy | typical/possible site (not exclusive) |
| `part_of` | Anatomy | Anatomy | containment hierarchy |
| `associated_with` | any | any | weaker/general association (reserved for PrimeKG enrichment, T10) |

## Validation rules (enforced by `medvision/kg/validate.py`)

1. Every node id is unique across all three node files.
2. Every edge's `src`/`dst` references an existing node id.
3. Every edge weight is in `[0, 1]`.
4. Every `rel` is one of the relation types listed above.
5. Every id in `configs/labels.yaml` (`label_to_node` values) exists as a `Finding` node.
6. Every `Disease` node is reachable from at least one `Finding` node via
   `suggestive_of` (directly or through `differential_of`/`complication_of`).

## Weight provenance disclaimer

Edge weights authored with `source: team` are illustrative associations from general
radiology/clinical teaching material, set by the project team for a college major
project. They are **not** derived from a validated clinical dataset or expert panel
and must not be read as calibrated probabilities. See PLAN.md's risk table for the
plan to have a clinician spot-check a sample of edges.
