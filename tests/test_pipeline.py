from pathlib import Path

import httpx
import pytest
from PIL import Image

from medvision.pipeline import analyze, get_models

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cxr.png"


@pytest.fixture
def force_template_report(monkeypatch):
    """Force the LLM path off so pipeline tests exercise the deterministic template
    report — fast and reproducible, per CLAUDE.md's "mock the LLM in tests"."""
    monkeypatch.setattr(get_models().llm_client, "is_available", lambda: False)


@pytest.mark.slow
def test_analyze_end_to_end(tmp_path: Path, force_template_report):
    image = Image.open(FIXTURE)
    result = analyze(image, save_dir=tmp_path)

    assert result.id
    assert result.findings, "expected at least one finding"
    assert all(0.0 <= f.prob <= 1.0 for f in result.findings)
    # findings sorted descending by probability
    probs = [f.prob for f in result.findings]
    assert probs == sorted(probs, reverse=True)

    shown = [f for f in result.findings if f.status != "absent"]
    assert shown, "expected at least one present/possible finding on this fixture"
    for f in shown:
        assert f.region is not None
        assert f.heatmap_path is not None
        assert (tmp_path / f.heatmap_path).exists()

    assert result.differentials
    top = result.differentials[0]
    assert top.paths
    assert all(0.0 <= d.score <= 1.0 for d in result.differentials)

    assert result.report_source == "template"
    assert result.grounding.score == 1.0
    assert result.evidence.findings
    assert result.report.findings

    assert "FINDINGS:" in result.report_text
    assert "IMPRESSION:" in result.report_text
    assert result.disclaimer in result.report_text

    assert result.timings_ms["total_ms"] > 0


@pytest.mark.slow
def test_analyze_without_save_dir_skips_heatmap_files(force_template_report):
    image = Image.open(FIXTURE)
    result = analyze(image, save_dir=None)
    shown = [f for f in result.findings if f.status != "absent"]
    assert all(f.heatmap_path is None for f in shown)
    assert all(f.region is not None for f in shown)  # localization doesn't need saving


def _ollama_reachable() -> bool:
    try:
        return (
            httpx.get(get_models().llm_client.base_url + "/api/version", timeout=1.0).status_code
            == 200
        )
    except httpx.HTTPError:
        return False


@pytest.mark.slow
def test_analyze_with_real_llm_when_available():
    """Integration check against a real local Ollama server, when one happens to be
    running. Skipped (not failed) when it isn't — CI and other machines shouldn't
    need Ollama installed for the rest of the suite to pass."""
    if not _ollama_reachable():
        pytest.skip("Ollama is not reachable on this machine")

    image = Image.open(FIXTURE)
    result = analyze(image, save_dir=None)

    assert result.report_source in {"llm", "template"}
    assert 0.0 <= result.grounding.score <= 1.0
    assert result.report.findings or result.report.impression
