# tests/test_second_price.py

"""Tests for auction clearing and settlement."""

import numpy as np
import pytest

from adserve_sim.auction.second_price import (
    clear_second_price,
    run_auction,
    sample_competing_bids,
    settle,
)


def test_winner_pays_the_runner_up_not_their_own_bid() -> None:
    """The defining property of a second-price auction.

    It is also what makes the whole experiment interpretable: because the price
    is set by someone else's bid, truthful bidding is optimal, so any revenue
    difference between two models comes from their probabilities rather than
    from bidding tactics.
    """
    outcome = clear_second_price(np.array([1.0]), np.array([[0.6, 0.3]]))

    assert outcome.won[0]
    assert outcome.price_paid[0] == pytest.approx(0.6)


def test_losing_costs_nothing() -> None:
    outcome = clear_second_price(np.array([0.2]), np.array([[0.9, 0.5]]))

    assert not outcome.won[0]
    assert outcome.price_paid[0] == 0.0


def test_publisher_still_collects_when_we_lose() -> None:
    """The auction clears whoever wins; only our side of it is empty."""
    outcome = clear_second_price(np.array([0.2]), np.array([[0.9, 0.5]]))

    assert outcome.cleared[0]
    assert outcome.clearing_price[0] == pytest.approx(0.5)


def test_our_bid_sets_the_price_when_we_are_the_runner_up() -> None:
    outcome = clear_second_price(np.array([0.7]), np.array([[0.9, 0.1]]))

    assert not outcome.won[0]
    assert outcome.clearing_price[0] == pytest.approx(0.7)


def test_ties_go_to_the_rivals() -> None:
    """Arbitrary, but consistent, and it avoids flattering our own bidder."""
    outcome = clear_second_price(np.array([0.5]), np.array([[0.5, 0.2]]))
    assert not outcome.won[0]


def test_reserve_blocks_auctions_below_the_floor() -> None:
    outcome = clear_second_price(np.array([0.3]), np.array([[0.2, 0.1]]), reserve=0.5)

    assert not outcome.cleared[0]
    assert not outcome.won[0]
    assert outcome.clearing_price[0] == 0.0


def test_reserve_floors_the_price_when_the_runner_up_is_below_it() -> None:
    """The reserve earns its keep here: it lifts a price nobody else would."""
    outcome = clear_second_price(np.array([0.9]), np.array([[0.1, 0.05]]), reserve=0.4)

    assert outcome.won[0]
    assert outcome.price_paid[0] == pytest.approx(0.4)


def test_price_never_exceeds_our_bid_when_we_win() -> None:
    """Truthful bidding must never lose money on the price alone."""
    rng = np.random.default_rng(0)
    ours = rng.uniform(0.05, 0.5, 500)
    rivals = rng.uniform(0.05, 0.5, (500, 4))

    outcome = clear_second_price(ours, rivals, reserve=0.1)
    assert (outcome.price_paid[outcome.won] <= ours[outcome.won] + 1e-12).all()


def test_raising_the_reserve_lowers_fill_rate() -> None:
    """The trade-off a reserve sweep explores, asserted in one direction."""
    rng = np.random.default_rng(1)
    ours = rng.uniform(0.05, 0.5, 1000)
    rivals = rng.uniform(0.05, 0.5, (1000, 4))

    low = clear_second_price(ours, rivals, reserve=0.05)
    high = clear_second_price(ours, rivals, reserve=0.45)

    assert high.fill_rate < low.fill_rate


def test_higher_bids_win_more() -> None:
    """The mechanism the calibration experiment depends on."""
    rng = np.random.default_rng(2)
    ours = rng.uniform(0.05, 0.3, 1000)
    rivals = rng.uniform(0.05, 0.3, (1000, 4))

    modest = clear_second_price(ours, rivals)
    inflated = clear_second_price(ours * 1.5, rivals)

    assert inflated.win_rate > modest.win_rate


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        clear_second_price(np.array([0.5, 0.5]), np.array([[0.1, 0.2]]))


def test_one_dimensional_rivals_rejected() -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        clear_second_price(np.array([0.5]), np.array([0.1, 0.2]))


def test_negative_reserve_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        clear_second_price(np.array([0.5]), np.array([[0.1]]), reserve=-1.0)


def test_competing_bids_are_reproducible() -> None:
    first = sample_competing_bids(100, reference_bid=0.16, seed=7)
    second = sample_competing_bids(100, reference_bid=0.16, seed=7)
    np.testing.assert_array_equal(first, second)


def test_competing_bids_centre_on_the_reference() -> None:
    """Rivals must sit near our own scale, or the experiment measures the gap
    between two arbitrary scales rather than the cost of miscalibration."""
    bids = sample_competing_bids(20_000, reference_bid=0.16, sigma=0.6, seed=0)
    assert np.median(bids) == pytest.approx(0.16, rel=0.05)


def test_competing_bids_are_positive() -> None:
    bids = sample_competing_bids(1000, reference_bid=0.16, seed=0)
    assert (bids > 0).all()


@pytest.mark.parametrize(
    ("impressions", "competitors", "reference", "sigma"),
    [(0, 5, 0.1, 0.6), (10, 0, 0.1, 0.6), (10, 5, 0.0, 0.6), (10, 5, 0.1, 0.0)],
)
def test_invalid_bid_parameters_rejected(
    impressions: int, competitors: int, reference: float, sigma: float
) -> None:
    with pytest.raises(ValueError):
        sample_competing_bids(impressions, reference, competitors, sigma)


def test_settlement_counts_only_auctions_we_won() -> None:
    """A click on an impression we lost belongs to someone else."""
    outcome = clear_second_price(
        np.array([1.0, 0.1]),
        np.array([[0.5, 0.2], [0.9, 0.4]]),
    )
    result = settle(outcome, np.array([1.0, 1.0]), value_per_click=10.0)

    assert result.impressions_won == 1
    assert result.clicks_won == 1
    assert result.advertiser_value == pytest.approx(10.0)


def test_settlement_profit_is_value_minus_spend() -> None:
    outcome = clear_second_price(np.array([1.0]), np.array([[0.6, 0.2]]))
    result = settle(outcome, np.array([1.0]), value_per_click=2.0)

    assert result.advertiser_profit == pytest.approx(2.0 - 0.6)


def test_overpaying_produces_negative_profit() -> None:
    """Winning an impression that does not click is a pure loss."""
    outcome = clear_second_price(np.array([1.0]), np.array([[0.6, 0.2]]))
    result = settle(outcome, np.array([0.0]), value_per_click=2.0)

    assert result.advertiser_profit < 0


def test_publisher_revenue_includes_auctions_we_lost() -> None:
    outcome = clear_second_price(
        np.array([0.1, 0.1]),
        np.array([[0.9, 0.5], [0.8, 0.3]]),
    )
    result = settle(outcome, np.array([0.0, 0.0]), value_per_click=1.0)

    assert result.impressions_won == 0
    assert result.publisher_revenue == pytest.approx(0.8)


def test_roi_and_cpc_handle_zero_denominators() -> None:
    outcome = clear_second_price(np.array([0.01]), np.array([[0.9, 0.5]]))
    result = settle(outcome, np.array([0.0]), value_per_click=1.0)

    assert result.roi == 0.0
    assert result.effective_cpc == 0.0


def test_settlement_shape_mismatch_rejected() -> None:
    outcome = clear_second_price(np.array([0.5]), np.array([[0.1]]))
    with pytest.raises(ValueError, match="shape mismatch"):
        settle(outcome, np.array([1.0, 0.0]), value_per_click=1.0)


def test_run_auction_matches_the_two_step_path() -> None:
    ours = np.array([0.3, 0.2])
    rivals = np.array([[0.1, 0.05], [0.9, 0.4]])
    clicks = np.array([1.0, 0.0])

    combined = run_auction(ours, clicks, rivals, reserve=0.05, value_per_click=1.0)
    stepwise = settle(clear_second_price(ours, rivals, 0.05), clicks, 1.0)

    assert combined == stepwise
