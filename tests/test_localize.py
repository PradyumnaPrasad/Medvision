import numpy as np

from medvision.explain.localize import localize, zone_energy_shares


def test_zone_shares_sum_to_one():
    heatmap = np.random.rand(224, 224).astype(np.float32)
    shares = zone_energy_shares(heatmap)
    assert len(shares) == 6
    assert sum(s.share for s in shares) == 1.0 or abs(sum(s.share for s in shares) - 1.0) < 1e-6


def test_localize_finds_hot_zone_bottom_left():
    heatmap = np.zeros((300, 200), dtype=np.float32)
    # bottom-left quadrant of the image -> patient's right lower zone (radiological convention)
    heatmap[210:290, 10:90] = 1.0
    result = localize(heatmap)
    assert result.zone_id == "A_RIGHT_LOWER_ZONE"
    assert result.share > 0.9


def test_localize_finds_hot_zone_top_right():
    heatmap = np.zeros((300, 200), dtype=np.float32)
    heatmap[10:90, 110:190] = 1.0
    result = localize(heatmap)
    assert result.zone_id == "A_LEFT_UPPER_ZONE"


def test_all_zero_heatmap_does_not_crash():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    shares = zone_energy_shares(heatmap)
    assert all(s.share == 0.0 for s in shares)
