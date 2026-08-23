# tests/test_reliability.py

"""Tests for calibration measurement."""

import numpy as np
import pytest

from adserve_sim.eval.distortion import STANDARD_SCENARIOS, Distortion
from adserve_sim.eval.reliability import (
    calibration_gap,
    compare_scenarios,
    expected_calibration_error,
    reliability_curve,
)


@pytest.fixture
def calibrated() -> tuple[np.ndarray, np.ndarray]:
    """Predictions that are correct by construction: labels drawn from them."""
    rng = np.random.default_rng(0)
    probabilities = rng.beta(2, 8, size=40_000)
    labels = rng.binomial(1, probabilities).astype(np.float64)
    return labels, probabilities


def test_perfect_calibration_has_near_zero_error(
    calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Sampling labels from the predictions makes them calibrated by definition."""
    labels, probabilities = calibrated
    assert expected_calibration_error(labels, probabilities) < 0.01


def test_shifted_predictions_have_larger_error(
    calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    labels, probabilities = calibrated

    baseline = expected_calibration_error(labels, probabilities)
    shifted = expected_calibration_error(labels, Distortion(shift=0.5).apply(probabilities))

    assert shifted > baseline * 5


def test_calibration_gap_is_signed(calibrated: tuple[np.ndarray, np.ndarray]) -> None:
    """Unlike ECE, the gap distinguishes over- from under-prediction."""
    labels, probabilities = calibrated

    over = calibration_gap(labels, Distortion(shift=0.5).apply(probabilities))
    under = calibration_gap(labels, Distortion(shift=-0.5).apply(probabilities))

    assert over > 0
    assert under < 0


def test_calibration_gap_misses_offsetting_errors() -> None:
    """A zero gap does not mean calibrated, which is why ECE exists too.

    One group is predicted at 0.8 and clicks half the time, another at 0.2 and
    also clicks half the time. The two errors cancel in the mean, so the gap is
    zero while the model is wrong about every single impression.
    """
    labels = np.array([1.0, 0.0] * 50 + [1.0, 0.0] * 50)
    probabilities = np.array([0.8] * 100 + [0.2] * 100)

    assert calibration_gap(labels, probabilities) == pytest.approx(0.0)
    assert expected_calibration_error(labels, probabilities, n_bins=10) == pytest.approx(0.3)


def test_calibration_is_an_aggregate_property_not_a_per_row_one() -> None:
    """A model can be perfectly calibrated and still useless.

    Here every impression gets the same 0.5, and half of them click. That is
    exactly calibrated, of everything scored 0.5, 50% converted, while telling
    you nothing about which ones. Calibration and discrimination are separate
    axes, which is why this project reports AUC alongside ECE rather than
    replacing one with the other.
    """
    labels = np.array([1.0] * 50 + [0.0] * 50)
    probabilities = np.full(100, 0.5)

    assert expected_calibration_error(labels, probabilities, n_bins=10) == pytest.approx(0.0)
    assert calibration_gap(labels, probabilities) == pytest.approx(0.0)


def test_reliability_curve_tracks_the_diagonal(
    calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    labels, probabilities = calibrated
    curve = reliability_curve(labels, probabilities)

    gaps = np.abs(curve.mean_predicted - curve.observed_frequency)
    assert (gaps * curve.weight).sum() < 0.01


def test_reliability_curve_weights_sum_to_one(
    calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    labels, probabilities = calibrated
    curve = reliability_curve(labels, probabilities)
    assert curve.weight.sum() == pytest.approx(1.0)


def test_reliability_curve_drops_empty_buckets() -> None:
    """An empty bucket reported as zero would put a false point at the origin."""
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    probabilities = np.array([0.51, 0.52, 0.53, 0.54])

    curve = reliability_curve(labels, probabilities, n_bins=20)
    assert len(curve.mean_predicted) == 1


def test_reliability_curve_to_frame(calibrated: tuple[np.ndarray, np.ndarray]) -> None:
    labels, probabilities = calibrated
    frame = reliability_curve(labels, probabilities).to_frame()
    assert list(frame.columns) == ["mean_predicted", "observed_frequency", "weight"]


def test_compare_scenarios_holds_auc_constant(
    calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """The headline result: identical ranking, different probabilities.

    If this ever fails, the distortions are reordering predictions and any
    downstream revenue difference would confound ordering with magnitude.
    """
    labels, probabilities = calibrated
    table = compare_scenarios(labels, probabilities, STANDARD_SCENARIOS)

    assert table["auc"].round(6).nunique() == 1
    assert table["ece"].nunique() == len(STANDARD_SCENARIOS)


def test_compare_scenarios_reports_every_scenario(
    calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    labels, probabilities = calibrated
    table = compare_scenarios(labels, probabilities, STANDARD_SCENARIOS)

    assert set(table.index) == set(STANDARD_SCENARIOS)
    assert table.loc["none", "ece"] == table["ece"].min()


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_calibration_error(np.array([0.0, 1.0]), np.array([0.5]))


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        expected_calibration_error(np.array([]), np.array([]))


def test_non_binary_labels_rejected() -> None:
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.2, 0.3]))
