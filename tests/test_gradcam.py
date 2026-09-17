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


@pytest.mark.slow
def test_building_explainer_does_not_move_shared_classifier_model():
    """Regression test: GradCamExplainer must not mutate the classifier's cached model
    in place. It used to call classifier.model.to("cpu") directly, which silently moved
    the process-wide cached model off its real device (e.g. mps) and broke every later
    call to classifier.predict()."""
    classifier = ChestXrayClassifier()
    original_device = classifier.device
    image = Image.open(FIXTURE)

    GradCamExplainer(classifier)  # constructing the explainer must not side-effect classifier

    param_device = next(classifier.model.parameters()).device.type
    assert param_device == original_device
    # predict() must still work on the classifier's own device after building an explainer
    predictions = classifier.predict(image)
    assert predictions
