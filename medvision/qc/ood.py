"""Input quality / out-of-distribution gate (PLAN.md T08).

Runs before the classifier, so an obviously-wrong upload (a color photo, a blank
image) never reaches the model and gets a confident-looking but meaningless
diagnosis. Uses cheap image statistics rather than a learned embedding model —
there's no BiomedCLIP integration yet (that's T16) — but the primary check
(color_variance) is unusually clean for this specific job:

Real chest X-rays are grayscale medical images. Even when a dataset stores them
as 3-channel RGB, the R/G/B values are ~identical at every pixel. A real color
photo's channels differ substantially. Measured on real data (see
reports/ood_gate_calibration.md): our chest X-ray fixtures scored ~0.0 on this
metric; a real color photograph scored ~110. That gap is what max_color_variance
polices. The metric is measured after removing any uniform colour cast, so a tinted
screenshot of a real X-ray is accepted (with a warning) instead of rejected.

Known limitation: this catches color photos, not other grayscale non-CXR images
(e.g. a different body part, a hand X-ray). A proper fix needs an embedding-based
check (T16) — this module says so explicitly in its warnings rather than
pretending to catch everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from medvision.config import get_settings


@dataclass(frozen=True)
class QualityCheckResult:
    is_valid: bool  # False = hard-reject; caller should refuse to run the pipeline
    flags: list[str] = field(default_factory=list)  # soft warnings; pipeline still runs
    color_variance: float = 0.0
    contrast_std: float = 0.0
    aspect_ratio: float = 0.0
    resolution: tuple[int, int] = (0, 0)


class InvalidImageError(ValueError):
    """Raised when check_image_quality hard-rejects an upload — the caller should
    refuse to run the rest of the pipeline rather than diagnose a meaningless image."""

    def __init__(self, result: QualityCheckResult) -> None:
        self.result = result
        super().__init__("; ".join(result.flags) or "image failed quality checks")


def _color_variance(rgb: np.ndarray) -> float:
    """Sum of mean pairwise absolute channel differences. ~0 for grayscale-as-RGB
    images (real chest X-rays); much higher for real color photos."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))


def _remove_uniform_tint(rgb: np.ndarray) -> np.ndarray:
    """Rescale each channel so all three have the same mean.

    A radiograph that was screenshotted or saved with a colour cast is still a grayscale
    image multiplied by a constant per-channel gain, so this removes its cast entirely
    (measured: even an extreme blue cast drops to ~3.6). A real photograph has colours that
    vary across the picture, so it stays far above the limit (measured: 22 to 217).
    """
    means = rgb.reshape(-1, 3).mean(axis=0)
    return rgb * (means.mean() / np.maximum(means, 1e-6))


def check_image_quality(image: Image.Image) -> QualityCheckResult:
    """Cheap heuristic checks run before the classifier ever sees the image."""
    qc = get_settings().qc

    rgb = np.array(image.convert("RGB")).astype(np.float64)
    gray = np.array(image.convert("L")).astype(np.float64)
    width, height = image.size

    raw_color_variance = _color_variance(rgb)
    color_variance = _color_variance(_remove_uniform_tint(rgb))
    contrast_std = float(gray.std())
    aspect_ratio = width / height if height > 0 else 0.0

    hard_reasons: list[str] = []
    soft_flags: list[str] = []

    if min(width, height) < qc.min_resolution:
        hard_reasons.append(
            f"image resolution {width}x{height} is below the minimum {qc.min_resolution}px"
        )
    if color_variance > qc.max_color_variance:
        hard_reasons.append(
            f"image does not look like a grayscale radiograph (color_variance="
            f"{color_variance:.1f} after removing any uniform tint, expected <= "
            f"{qc.max_color_variance}) — this looks like a color photo, not a chest X-ray"
        )
    if contrast_std < qc.min_contrast_std:
        hard_reasons.append(
            f"image has almost no contrast (std={contrast_std:.1f}) — likely a blank or "
            "corrupted upload"
        )

    if color_variance <= qc.max_color_variance < raw_color_variance:
        soft_flags.append(
            f"image has a colour tint (color_variance={raw_color_variance:.1f}); it was "
            "analysed as grayscale, but check that it is an unaltered radiograph"
        )

    if not (qc.aspect_ratio_min <= aspect_ratio <= qc.aspect_ratio_max):
        soft_flags.append(
            f"unusual aspect ratio ({aspect_ratio:.2f}) for a frontal chest X-ray — this may "
            "be a lateral view, a cropped image, or not a chest X-ray at all"
        )

    is_valid = not hard_reasons
    return QualityCheckResult(
        is_valid=is_valid,
        flags=hard_reasons + soft_flags,
        color_variance=color_variance,
        contrast_std=contrast_std,
        aspect_ratio=aspect_ratio,
        resolution=(width, height),
    )
