from pathlib import Path

import pytest
from PIL import Image

from medvision.pipeline import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cxr.png"


@pytest.mark.slow
def test_analyze_end_to_end(tmp_path: Path):
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

    assert "FINDINGS:" in result.report_text
    assert "IMPRESSION:" in result.report_text
    assert result.disclaimer in result.report_text

    assert result.timings_ms["total_ms"] > 0


@pytest.mark.slow
def test_analyze_without_save_dir_skips_heatmap_files():
    image = Image.open(FIXTURE)
    result = analyze(image, save_dir=None)
    shown = [f for f in result.findings if f.status != "absent"]
    assert all(f.heatmap_path is None for f in shown)
    assert all(f.region is not None for f in shown)  # localization doesn't need saving
