# tests/test_distortion.py

"""Tests for controlled miscalibration in log-odds space."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from adserve_sim.eval.distortion import STANDARD_SCENARIOS, Distortion


@pytest.fixture
def probabilities() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.beta(2, 8, size=2000)


@pytest.fixture
def labels(probabilities: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.binomial(1, probabilities).astype(np.float64)


def test_identity_leaves_predictions_untouched(probabilities: np.ndarray) -> None:
    result = Distortion().apply(probabilities)
    np.testing.assert_allclose(result, probabilities, atol=1e-9)


def test_is_identity_flag() -> None:
    assert Distortion().is_identity
    assert not Distortion(shift=0.1).is_identity
    assert not Distortion(sharpness=1.2).is_identity


def test_distortion_preserves_ranking(probabilities: np.ndarray, labels: np.ndarray) -> None:
    """The central guarantee: every distortion is invisible to AUC.

    This is what makes the injection design meaningful. If distortion moved the
    ranking, a difference in downstream revenue could be attributed to either
    ordering or magnitude, and the experiment would confound the two.
    """
    original = roc_auc_score(labels, probabilities)

    for distortion in STANDARD_SCENARIOS.values():
        distorted = distortion.apply(probabilities)
        assert roc_auc_score(labels, distorted) == pytest.approx(original, abs=1e-9)


def test_distortion_preserves_pairwise_order(probabilities: np.ndarray) -> None:
    """Stated as order directly, not via a metric that summarises it."""
    distorted = Distortion(sharpness=1.5, shift=0.4).apply(probabilities)
    assert (np.argsort(distorted) == np.argsort(probabilities)).all()


@pytest.mark.parametrize("shift", [0.4, 1.0])
def test_positive_shift_raises_every_prediction(probabilities: np.ndarray, shift: float) -> None:
    distorted = Distortion(shift=shift).apply(probabilities)
    assert (distorted > probabilities).all()


@pytest.mark.parametrize("shift", [-0.4, -1.0])
def test_negative_shift_lowers_every_prediction(probabilities: np.ndarray, shift: float) -> None:
    distorted = Distortion(shift=shift).apply(probabilities)
    assert (distorted < probabilities).all()


def test_sharpness_above_one_spreads_predictions(probabilities: np.ndarray) -> None:
    """Overconfidence pushes mass toward the extremes, widening the spread."""
    distorted = Distortion(sharpness=1.8).apply(probabilities)
    assert distorted.std() > probabilities.std()


def test_sharpness_below_one_compresses_predictions(probabilities: np.ndarray) -> None:
    distorted = Distortion(sharpness=0.5).apply(probabilities)
    assert distorted.std() < probabilities.std()


def test_output_stays_in_range() -> None:
    """Extreme inputs and a strong shift must not escape [0, 1]."""
    extreme = np.array([0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0])
    distorted = Distortion(sharpness=2.0, shift=3.0).apply(extreme)

    assert (distorted >= 0.0).all()
    assert (distorted <= 1.0).all()
    assert np.isfinite(distorted).all()


def test_negative_sharpness_rejected() -> None:
    """A negative multiplier inverts the ranking, which is not miscalibration."""
    with pytest.raises(ValueError, match="preserve ranking"):
        Distortion(sharpness=-1.0)


def test_zero_sharpness_rejected() -> None:
    with pytest.raises(ValueError, match="preserve ranking"):
        Distortion(sharpness=0.0)


def test_probabilities_outside_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Distortion().apply(np.array([0.5, 1.5]))


def test_standard_scenarios_cover_both_failure_modes() -> None:
    assert STANDARD_SCENARIOS["none"].is_identity
    assert STANDARD_SCENARIOS["over-predict"].shift > 0
    assert STANDARD_SCENARIOS["under-predict"].shift < 0
    assert STANDARD_SCENARIOS["overconfident"].sharpness > 1
    assert STANDARD_SCENARIOS["underconfident"].sharpness < 1
