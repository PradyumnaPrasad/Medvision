import numpy as np
import pytest

from medvision.eval.classification import evaluate_classification


def test_perfect_classifier_scores_maximally():
    probs = np.array([0.9, 0.95, 0.1, 0.05, 0.85])
    labels = np.array([1, 1, 0, 0, 1])
    result = evaluate_classification("F_TEST", probs, labels, threshold=0.5)

    assert result.auroc == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.n == 5
    assert result.n_positive == 3


def test_random_guessing_has_low_auroc():
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, size=n)
    probs = rng.random(n)  # uncorrelated with labels
    result = evaluate_classification("F_TEST", probs, labels, threshold=0.5)
    assert 0.4 < result.auroc < 0.6


def test_auroc_is_none_when_only_one_class_present():
    probs = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0])
    result = evaluate_classification("F_TEST", probs, labels, threshold=0.5)
    assert result.auroc is None


def test_precision_recall_with_no_predicted_positives_are_zero_not_crash():
    probs = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 0, 1])
    result = evaluate_classification("F_TEST", probs, labels, threshold=0.9)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_threshold_is_recorded_on_the_result():
    probs = np.array([0.6, 0.4])
    labels = np.array([1, 0])
    result = evaluate_classification("F_TEST", probs, labels, threshold=0.55)
    assert result.threshold == 0.55
