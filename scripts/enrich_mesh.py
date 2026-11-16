"""One-off data-prep script: find MeSH descriptor ids for kg/ontology nodes still
tagged `source: team`, using NLM's public MeSH lookup API (no auth, no bulk download
needed for ~28 targeted terms).

This is a DRY RUN only: it never edits the ontology YAML. It writes proposed
matches to a JSON file so a human can review each one for medical correctness
before anything is applied by hand (see reports/mesh_enrichment_candidates.json).

Usage: uv run python scripts/enrich_mesh.py
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_FILES = [
    REPO_ROOT / "kg/ontology/findings.yaml",
    REPO_ROOT / "kg/ontology/anatomy.yaml",
    REPO_ROOT / "kg/ontology/diseases.yaml",
]
OUTPUT_PATH = REPO_ROOT / "reports/mesh_enrichment_candidates.json"

LOOKUP_URL = "https://id.nlm.nih.gov/mesh/lookup/descriptor?label={term}&match=exact&limit=1"
RESOURCE_URL = "https://id.nlm.nih.gov/mesh/{mesh_id}.json"
REQUEST_DELAY_S = 0.34  # be polite to a free public API


def _get_json(url: str) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None


def lookup_exact(term: str) -> tuple[str, str] | None:
    """Return (mesh_id, matched_label) for an exact MeSH descriptor label match."""
    encoded = urllib.parse.quote(term)
    results = _get_json(LOOKUP_URL.format(term=encoded))
    if not results:
        return None
    resource = results[0]["resource"]
    mesh_id = resource.rsplit("/", 1)[-1]
    return mesh_id, results[0]["label"]


def fetch_scope_note(mesh_id: str) -> str | None:
    descriptor = _get_json(RESOURCE_URL.format(mesh_id=mesh_id))
    if not descriptor:
        return None
    preferred_concept = descriptor.get("preferredConcept")
    if not preferred_concept:
        return None
    concept_id = preferred_concept.rsplit("/", 1)[-1]
    concept = _get_json(RESOURCE_URL.format(mesh_id=concept_id))
    if not concept:
        return None
    scope_note = concept.get("scopeNote")
    return scope_note["@value"] if scope_note else None


def main() -> None:
    candidates = []
    for path in ONTOLOGY_FILES:
        data = yaml.safe_load(path.read_text())
        for node in data["nodes"]:
            if node.get("source") != "team":
                continue
            search_terms = [node["name"], *node.get("aliases", [])]
            match = None
            for term in search_terms:
                match = lookup_exact(term)
                time.sleep(REQUEST_DELAY_S)
                if match:
                    break
            if not match:
                logger.info("NO MATCH   %-30s tried %s", node["id"], search_terms)
                continue
            mesh_id, matched_label = match
            scope_note = fetch_scope_note(mesh_id)
            time.sleep(REQUEST_DELAY_S)
            logger.info("MATCH      %-30s -> %s (%s)", node["id"], mesh_id, matched_label)
            candidates.append(
                {
                    "id": node["id"],
                    "file": path.name,
                    "current_name": node["name"],
                    "current_definition": node["definition"],
                    "matched_mesh_label": matched_label,
                    "mesh_id": mesh_id,
                    "mesh_scope_note": scope_note,
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(candidates, indent=2))
    logger.info("\nWrote %d candidates to %s", len(candidates), OUTPUT_PATH)


if __name__ == "__main__":
    main()
