# tests/test_ranking.py

"""Tests for turning a click probability into a bid."""

import numpy as np
import pytest

from adserve_sim.auction.ranking import (
    DEFAULT_VIEWABILITY,
    VIEWABILITY_PRIOR,
    bid,
    expected_value,
    viewability_prior,
)


def test_viewability_prior_maps_known_positions() -> None:
    result = viewability_prior(np.array(["0", "1", "2"]))
    expected = [VIEWABILITY_PRIOR["0"], VIEWABILITY_PRIOR["1"], VIEWABILITY_PRIOR["2"]]
    np.testing.assert_allclose(result, expected)


def test_viewability_prior_falls_back_for_unknown_positions() -> None:
    """An unseen slot must get a value, not a null that poisons the bid."""
    result = viewability_prior(np.array(["99"]))
    assert result[0] == DEFAULT_VIEWABILITY


def test_viewability_prior_accepts_an_override() -> None:
    """The prior is an assumption, so varying it has to be the easy path."""
    result = viewability_prior(np.array(["0", "1"]), prior={"0": 0.9, "1": 0.1})
    np.testing.assert_allclose(result, [0.9, 0.1])


def test_viewability_prior_rejects_impossible_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        viewability_prior(np.array(["0"]), prior={"0": 1.4})


def test_default_prior_decreases_with_position() -> None:
    """The one assumption the table encodes: lower slots are seen less."""
    ordered = [VIEWABILITY_PRIOR[key] for key in sorted(VIEWABILITY_PRIOR, key=int)]
    assert ordered == sorted(ordered, reverse=True)


def test_expected_value_multiplies_the_three_terms() -> None:
    result = expected_value(np.array([0.2]), np.array([0.5]), value_per_click=2.0)
    assert result[0] == pytest.approx(0.2)


def test_expected_value_defaults_to_ignoring_viewability() -> None:
    """Without a viewability term the bid prices raw eCPM, which is the control."""
    result = expected_value(np.array([0.2]), value_per_click=1.0)
    assert result[0] == pytest.approx(0.2)


def test_expected_value_scales_linearly_with_click_value() -> None:
    """Doubling what a click is worth doubles every bid.

    This is why comparisons at a fixed value_per_click are meaningful even
    though the value itself is invented: the constant cancels.
    """
    probabilities = np.array([0.1, 0.3, 0.6])

    single = expected_value(probabilities, value_per_click=1.0)
    double = expected_value(probabilities, value_per_click=2.0)

    np.testing.assert_allclose(double, single * 2.0)


def test_expected_value_rejects_negative_click_value() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        expected_value(np.array([0.2]), value_per_click=-1.0)


@pytest.mark.parametrize("bad", [np.array([1.2]), np.array([-0.1])])
def test_expected_value_rejects_impossible_probabilities(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_value(bad)


def test_bid_without_positions_ignores_viewability() -> None:
    probabilities = np.array([0.1, 0.2])
    np.testing.assert_allclose(bid(probabilities), probabilities)


def test_bid_with_positions_is_lower_than_without() -> None:
    """Viewability can only reduce a bid, since it is a probability."""
    probabilities = np.array([0.2, 0.2, 0.2])
    positions = np.array(["0", "1", "7"])

    with_view = bid(probabilities, positions)
    without = bid(probabilities)

    assert (with_view < without).all()


def test_bid_ranks_positions_by_viewability() -> None:
    """Equal click probability, different slots: the better slot bids more."""
    probabilities = np.full(3, 0.2)
    result = bid(probabilities, np.array(["0", "2", "7"]))

    assert result[0] > result[1] > result[2]


def test_bid_is_proportional_to_click_probability() -> None:
    """The property the whole project turns on.

    A systematically inflated probability produces a systematically inflated
    bid, in exact proportion. There is no saturation or clipping to absorb the
    error, it passes straight through into the price.
    """
    truth = np.array([0.1, 0.2, 0.3])
    inflated = truth * 1.3

    ratio = bid(inflated) / bid(truth)
    np.testing.assert_allclose(ratio, 1.3)


def test_bid_is_monotone_in_click_probability() -> None:
    probabilities = np.array([0.05, 0.1, 0.4, 0.8])
    positions = np.array(["1", "1", "1", "1"])

    result = bid(probabilities, positions)
    assert (np.diff(result) > 0).all()
