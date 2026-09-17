from pathlib import Path

import pytest
from PIL import Image

from medvision.vision.classifier import ChestXrayClassifier

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cxr.png"


@pytest.mark.slow
def test_predict_returns_probabilities_for_all_mapped_labels():
    classifier = ChestXrayClassifier()
    image = Image.open(FIXTURE)
    predictions = classifier.predict(image)

    assert len(predictions) == 18  # every torchxrayvision label is mapped in labels.yaml
    node_ids = {p.node_id for p in predictions}
    assert "F_EFFUSION" in node_ids
    assert "F_CARDIOMEGALY" in node_ids
    for p in predictions:
        assert 0.0 <= p.prob <= 1.0
        assert p.node_id.startswith("F_")


@pytest.mark.slow
def test_preprocess_shape():
    classifier = ChestXrayClassifier()
    image = Image.open(FIXTURE)
    tensor = classifier.preprocess(image)
    assert tensor.shape == (1, 1, classifier.resolution, classifier.resolution)
