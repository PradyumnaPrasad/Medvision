"""One-off data-prep script: find PrimeKG disease<->phenotype edges that
corroborate an existing Finding-Disease pair already in our ontology.

This is a DRY RUN only: it never edits the ontology YAML. It reads a local
copy of PrimeKG's kg.csv (download separately - see reports/ for provenance)
and writes candidate `associated_with` edges to a JSON file for human review
before anything is applied by hand.

Matching is deliberately conservative: only exact (case-insensitive) matches
on a disease's name/aliases against PrimeKG's disease-side node name, and a
finding's name/aliases against PrimeKG's phenotype-side node name. PrimeKG's
disease taxonomy (Mondo-based) is far more granular than our 16 diseases, so
loose/fuzzy matching would silently link unrelated concepts - not acceptable
for a clinical-adjacent tool even as a college project.

Usage: uv run python scripts/enrich_primekg.py /path/to/kg.csv
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "reports/primekg_enrichment_candidates.json"
RELATION = "disease_phenotype_positive"


def _name_index(nodes: list[dict]) -> dict[str, str]:
    """Map every lowercased name/alias to its node id."""
    index: dict[str, str] = {}
    for node in nodes:
        index[node["name"].lower()] = node["id"]
        for alias in node.get("aliases", []):
            index[alias.lower()] = node["id"]
    return index


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    kg_csv_path = Path(sys.argv[1])

    diseases = yaml.safe_load((REPO_ROOT / "kg/ontology/diseases.yaml").read_text())["nodes"]
    findings = yaml.safe_load((REPO_ROOT / "kg/ontology/findings.yaml").read_text())["nodes"]
    relations = yaml.safe_load((REPO_ROOT / "kg/ontology/relations.yaml").read_text())["edges"]

    disease_index = _name_index(diseases)
    finding_index = _name_index(findings)
    existing_pairs = {
        (edge["src"], edge["dst"])
        for edge in relations
        if edge["rel"] in ("suggestive_of", "contradicts")
    }

    candidates: dict[tuple[str, str], dict] = {}
    rows_scanned = 0
    rows_matched_relation = 0

    with kg_csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_scanned += 1
            if row["relation"] != RELATION:
                continue
            rows_matched_relation += 1

            # PrimeKG stores each undirected pair once per direction; disease
            # can be on either the x or y side for this relation.
            for disease_side, phenotype_side in (("x", "y"), ("y", "x")):
                d_type = row[f"{disease_side}_type"]
                p_type = row[f"{phenotype_side}_type"]
                if d_type != "disease" or p_type != "effect/phenotype":
                    continue
                d_name = row[f"{disease_side}_name"].strip().lower()
                p_name = row[f"{phenotype_side}_name"].strip().lower()
                disease_id = disease_index.get(d_name)
                finding_id = finding_index.get(p_name)
                if not disease_id or not finding_id:
                    continue
                key = (finding_id, disease_id)
                if key in existing_pairs or key in candidates:
                    continue
                candidates[key] = {
                    "src": finding_id,
                    "rel": "associated_with",
                    "dst": disease_id,
                    "weight": 0.3,
                    "source": "primekg",
                    "note": (
                        f"PrimeKG disease-phenotype association: "
                        f"'{row[f'{disease_side}_name']}' <-> '{row[f'{phenotype_side}_name']}' "
                        f"(HPO/MONDO, relation={RELATION})."
                    ),
                }

            if rows_scanned % 2_000_000 == 0:
                logger.info("...scanned %d rows", rows_scanned)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(list(candidates.values()), indent=2))
    logger.info(
        "Scanned %d rows (%d matched relation=%s). Wrote %d candidate edges to %s",
        rows_scanned,
        rows_matched_relation,
        RELATION,
        len(candidates),
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
