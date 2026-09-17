"""Anatomical localization of Grad-CAM heatmaps (PLAN.md section 4.4, T11).

Splits the classifier's square input into a coarse 3x2 zone grid (upper/mid/lower x
left/right) matching the Anatomy nodes in kg/ontology/anatomy.yaml, and reports which
zone holds the most Grad-CAM energy for a given finding. This turns "there's a hot
blob on the image" into a clinically phrased "opacity, right lower zone" — the
localization half of MEDVISION's explanations, paired with the KG evidence path from
medvision/kg/reasoning.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Zone layout: rows are upper/mid/lower thirds, columns are patient-left/patient-right.
# On a standard PA frontal chest X-ray, patient's right lung appears on the *left* side
# of the image (radiological convention), so image-left maps to A_RIGHT_*.
ZONE_GRID = [
    ["A_RIGHT_UPPER_ZONE", "A_LEFT_UPPER_ZONE"],
    ["A_RIGHT_MID_ZONE", "A_LEFT_MID_ZONE"],
    ["A_RIGHT_LOWER_ZONE", "A_LEFT_LOWER_ZONE"],
]


@dataclass(frozen=True)
class ZoneShare:
    zone_id: str
    share: float  # fraction of total CAM energy in this zone, in [0, 1]


def zone_energy_shares(heatmap: np.ndarray) -> list[ZoneShare]:
    """Split `heatmap` (H, W) into the 3x2 zone grid and return each zone's energy share."""
    h, w = heatmap.shape
    row_bounds = [0, h // 3, 2 * h // 3, h]
    col_bounds = [0, w // 2, w]

    total = float(heatmap.sum())
    shares: list[ZoneShare] = []
    for row_idx in range(3):
        for col_idx in range(2):
            r0, r1 = row_bounds[row_idx], row_bounds[row_idx + 1]
            c0, c1 = col_bounds[col_idx], col_bounds[col_idx + 1]
            energy = float(heatmap[r0:r1, c0:c1].sum())
            share = energy / total if total > 0 else 0.0
            shares.append(ZoneShare(zone_id=ZONE_GRID[row_idx][col_idx], share=share))
    return shares


def localize(heatmap: np.ndarray) -> ZoneShare:
    """Return the single zone with the most Grad-CAM energy for this heatmap."""
    shares = zone_energy_shares(heatmap)
    return max(shares, key=lambda z: z.share)
