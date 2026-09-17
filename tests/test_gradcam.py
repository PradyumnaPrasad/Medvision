from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medvision.explain.gradcam import GradCamExplainer
from medvision.vision.classifier import ChestXrayClassifier

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cxr.png"


@pytest.mark.slow
def test_heatmap_shape_and_range():
    classifier = ChestXrayClassifier()
    explainer = GradCamExplainer(classifier)
    image = Image.open(FIXTURE)

    cam = explainer.heatmap(image, "F_EFFUSION")

    assert cam.shape == (classifier.resolution, classifier.resolution)
    assert cam.min() >= 0.0 - 1e-5
    assert cam.max() <= 1.0 + 1e-5


@pytest.mark.slow
def test_overlay_returns_rgb_image_same_size_as_cam():
    classifier = ChestXrayClassifier()
    explainer = GradCamExplainer(classifier)
    image = Image.open(FIXTURE)

    overlay = explainer.overlay(image, "F_CARDIOMEGALY")

    assert overlay.mode == "RGB"
    assert overlay.size == (classifier.resolution, classifier.resolution)
    arr = np.array(overlay)
    assert arr.std() > 0  # not a flat/blank image


@pytest.mark.slow
def test_unknown_node_raises():
    classifier = ChestXrayClassifier()
    explainer = GradCamExplainer(classifier)
    image = Image.open(FIXTURE)
    with pytest.raises(KeyError):
        explainer.heatmap(image, "F_NOT_A_REAL_NODE")
