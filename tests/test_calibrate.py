import numpy as np
import pytest

from medvision.vision.calibrate import (
    CalibrationData,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    load_calibration,
    logit,
    sigmoid,
)


def test_sigmoid_logit_are_inverses():
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-6)


def test_apply_temperature_identity_at_one():
    assert apply_temperature(0.73, 1.0) == 0.73


def test_apply_temperature_softens_confidence_when_t_greater_than_one():
    softened = apply_temperature(0.95, temperature=2.0)
    assert 0.5 < softened < 0.95


def test_apply_temperature_sharpens_confidence_when_t_less_than_one():
    sharpened = apply_temperature(0.6, temperature=0.5)
    assert sharpened > 0.6


def test_ece_zero_for_perfectly_calibrated_predictions():
    # In each bin, confidence exactly matches the empirical accuracy
    probs = np.array([0.1] * 10 + [0.9] * 10)
    labels = np.array([0] * 9 + [1] + [1] * 9 + [0])  # 1/10 positive at p=0.1, 9/10 at p=0.9
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_high_for_systematically_overconfident_predictions():
    # model says 0.95 every time, but truth is only positive half the time
    probs = np.full(100, 0.95)
    labels = np.array([1, 0] * 50)
    ece = expected_calibration_error(probs, labels)
    assert ece == pytest.approx(0.45, abs=1e-6)


def test_ece_empty_input_is_zero():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_fit_temperature_softens_systematically_overconfident_model():
    """Simulate a model whose raw probabilities are always too extreme relative to the
    true rate (e.g. it says ~0.9 for something that's only true 50% of the time).
    Fitting should find T > 1 (softening), and applying it should lower the ECE."""
    rng = np.random.default_rng(0)
    n = 2000
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    # true separation exists (labels correlate with a signal) but the model over-amplifies it
    signal = rng.normal(loc=0.0, scale=1.0, size=n) + labels * 1.5
    overconfident_probs = sigmoid(signal * 3.0)  # x3 too sharp

    t = fit_temperature(overconfident_probs, labels)
    assert t > 1.0  # fitting should detect the over-sharpening and want to soften it

    calibrated = np.array([apply_temperature(p, t) for p in overconfident_probs])
    ece_before = expected_calibration_error(overconfident_probs, labels)
    ece_after = expected_calibration_error(calibrated, labels)
    assert ece_after < ece_before


def test_fit_temperature_near_one_for_already_calibrated_model():
    rng = np.random.default_rng(1)
    n = 3000
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    signal = rng.normal(size=n) + labels * 1.5
    well_calibrated_probs = sigmoid(signal)  # no artificial sharpening/softening

    t = fit_temperature(well_calibrated_probs, labels)
    assert t == pytest.approx(1.0, abs=0.25)


# ---------- CalibrationData ----------


def test_identity_calibration_is_a_noop():
    cal = CalibrationData.identity()
    assert cal.apply("F_ANYTHING", 0.42) == 0.42


def test_calibration_data_applies_per_label_temperature():
    cal = CalibrationData(temperatures={"F_PNEUMONIA_OPACITY": 2.0}, default_temperature=1.0)
    calibrated = cal.apply("F_PNEUMONIA_OPACITY", 0.95)
    uncalibrated = cal.apply("F_CARDIOMEGALY", 0.95)  # not in temperatures -> default 1.0
    assert calibrated < 0.95
    assert uncalibrated == 0.95


def test_save_and_load_roundtrip(tmp_path):
    cal = CalibrationData(
        temperatures={"F_PNEUMONIA_OPACITY": 1.8},
        default_temperature=1.0,
        dataset="test-dataset",
        notes="unit test",
    )
    path = tmp_path / "calibration.json"
    cal.save(path)

    loaded = load_calibration(path)
    assert loaded.temperatures["F_PNEUMONIA_OPACITY"] == 1.8
    assert loaded.dataset == "test-dataset"


def test_load_calibration_missing_file_returns_identity(tmp_path):
    cal = load_calibration(tmp_path / "does_not_exist.json")
    assert cal.temperatures == {}
    assert cal.default_temperature == 1.0
    assert cal.apply("F_ANYTHING", 0.5) == 0.5
