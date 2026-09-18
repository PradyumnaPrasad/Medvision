from pathlib import Path

import numpy as np
from PIL import Image

from medvision.qc.ood import InvalidImageError, check_image_quality

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cxr.png"


def _image_from_array(arr: np.ndarray, mode: str = "L") -> Image.Image:
    return Image.fromarray(arr.astype("uint8"), mode=mode)


def test_real_chest_xray_fixture_passes():
    image = Image.open(FIXTURE)
    result = check_image_quality(image)
    assert result.is_valid
    assert result.color_variance < 1.0  # grayscale-stored image, channels ~identical


def test_color_photo_is_hard_rejected():
    rng = np.random.default_rng(0)
    # distinct, uncorrelated R/G/B channels — nothing like a grayscale radiograph
    r = rng.integers(0, 255, size=(300, 300), dtype=np.uint8)
    g = rng.integers(0, 255, size=(300, 300), dtype=np.uint8)
    b = rng.integers(0, 255, size=(300, 300), dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    image = _image_from_array(rgb, mode="RGB")

    result = check_image_quality(image)

    assert not result.is_valid
    assert result.color_variance > 8.0
    assert any("color photo" in f for f in result.flags)


def test_blank_image_is_hard_rejected():
    blank = np.full((300, 300), 128, dtype=np.uint8)
    image = _image_from_array(blank)

    result = check_image_quality(image)

    assert not result.is_valid
    assert result.contrast_std < 1.0
    assert any("blank or corrupted" in f for f in result.flags)


def test_too_small_image_is_hard_rejected():
    rng = np.random.default_rng(1)
    tiny = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    image = _image_from_array(tiny)

    result = check_image_quality(image)

    assert not result.is_valid
    assert any("resolution" in f for f in result.flags)


def test_unusual_aspect_ratio_is_a_soft_flag_not_a_hard_reject():
    rng = np.random.default_rng(2)
    # grayscale, good contrast and resolution, but a panorama-like shape
    wide = rng.integers(30, 220, size=(200, 900), dtype=np.uint8)
    image = _image_from_array(wide)

    result = check_image_quality(image)

    assert result.is_valid  # soft flags don't block the pipeline
    assert any("aspect ratio" in f for f in result.flags)


def test_invalid_image_error_carries_the_result_and_a_readable_message():
    blank = np.full((300, 300), 128, dtype=np.uint8)
    result = check_image_quality(_image_from_array(blank))
    assert not result.is_valid

    error = InvalidImageError(result)
    assert error.result is result
    assert "blank or corrupted" in str(error)


def test_plausible_grayscale_radiograph_shape_passes_cleanly():
    rng = np.random.default_rng(3)
    plausible = rng.integers(20, 235, size=(512, 512), dtype=np.uint8)
    image = _image_from_array(plausible)

    result = check_image_quality(image)

    assert result.is_valid
    assert result.flags == []
