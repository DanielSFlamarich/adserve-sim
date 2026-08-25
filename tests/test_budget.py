# tests/test_budget.py

"""Tests for spending limits and pacing."""

import numpy as np
import pytest

from adserve_sim.auction.budget import pacing_rate, run_budgeted_auction
from adserve_sim.auction.second_price import clear_second_price, run_auction


@pytest.fixture
def market() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bids, clicks and rivals for a market where we win a reasonable share."""
    rng = np.random.default_rng(0)
    n = 5000

    truth = rng.beta(2, 10, n)
    clicks = rng.binomial(1, truth).astype(float)
    bids = truth.copy()
    rivals = rng.lognormal(np.log(np.median(bids)), 0.6, (n, 3))

    return bids, clicks, rivals


def test_generous_budget_matches_the_unconstrained_auction(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A limit nobody reaches must change nothing."""
    bids, clicks, rivals = market

    unconstrained = run_auction(bids, clicks, rivals, value_per_click=1.0)
    budgeted, detail = run_budgeted_auction(bids, clicks, rivals, budget=1e9)

    assert budgeted == unconstrained
    assert detail.exhausted_at is None


def test_spending_never_exceeds_the_budget(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    bids, clicks, rivals = market
    budget = 20.0

    settlement, detail = run_budgeted_auction(bids, clicks, rivals, budget=budget)

    assert settlement.advertiser_spend <= budget
    assert detail.spent <= budget


def test_a_tight_budget_reduces_impressions_won(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    bids, clicks, rivals = market

    generous, _ = run_budgeted_auction(bids, clicks, rivals, budget=1e9)
    tight, _ = run_budgeted_auction(bids, clicks, rivals, budget=10.0)

    assert tight.impressions_won < generous.impressions_won


def test_exhaustion_point_is_reported(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Where the money runs out is the pacing story, so it has to be visible."""
    bids, clicks, rivals = market

    _, detail = run_budgeted_auction(bids, clicks, rivals, budget=5.0)

    assert detail.exhausted_at is not None
    assert 0 < detail.exhausted_at < len(bids)
    assert detail.exhausted_fraction is not None
    assert 0.0 < detail.exhausted_fraction < 1.0


def test_nothing_is_won_after_exhaustion(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Out of money means out of the market, not selectively frugal."""
    bids, clicks, rivals = market

    _, detail = run_budgeted_auction(bids, clicks, rivals, budget=5.0)
    assert detail.exhausted_at is not None

    assert not detail.outcome.won[detail.exhausted_at :].any()


def test_over_bidding_exhausts_the_budget_sooner(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The mechanism that makes miscalibration costly once money is finite.

    Without a budget, over-bidding wins more auctions at prices set by rivals
    and comes out ahead. With one, it spends the same money earlier and on worse
    value, and is out of the market for the rest of the period.
    """
    bids, clicks, rivals = market
    budget = 15.0

    _, honest = run_budgeted_auction(bids, clicks, rivals, budget=budget)
    _, inflated = run_budgeted_auction(bids * 1.5, clicks, rivals, budget=budget)

    assert honest.exhausted_at is not None
    assert inflated.exhausted_at is not None
    assert inflated.exhausted_at < honest.exhausted_at


def test_pacing_extends_participation(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Throttling trades early wins for still being in the market later."""
    bids, clicks, rivals = market
    budget = 10.0

    _, unpaced = run_budgeted_auction(bids, clicks, rivals, budget=budget, pace=False)
    _, paced = run_budgeted_auction(bids, clicks, rivals, budget=budget, pace=True)

    assert paced.participation_rate < 1.0
    assert unpaced.exhausted_at is not None

    reached = paced.exhausted_at if paced.exhausted_at is not None else len(bids)
    assert reached > unpaced.exhausted_at


def test_pacing_rate_is_one_when_the_budget_is_ample(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    bids, _, rivals = market
    outcome = clear_second_price(bids, rivals)

    assert pacing_rate(outcome, budget=1e9) == 1.0


def test_pacing_rate_scales_with_the_shortfall(
    market: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Spend three times the budget, enter one auction in three."""
    bids, _, rivals = market
    outcome = clear_second_price(bids, rivals)
    total = float(outcome.price_paid.sum())

    assert pacing_rate(outcome, budget=total / 3) == pytest.approx(1 / 3)


def test_pacing_is_reproducible(market: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    bids, clicks, rivals = market

    first, _ = run_budgeted_auction(bids, clicks, rivals, budget=10.0, pace=True, seed=3)
    second, _ = run_budgeted_auction(bids, clicks, rivals, budget=10.0, pace=True, seed=3)

    assert first == second


@pytest.mark.parametrize("budget", [0.0, -1.0])
def test_non_positive_budget_rejected(
    market: tuple[np.ndarray, np.ndarray, np.ndarray], budget: float
) -> None:
    bids, clicks, rivals = market
    with pytest.raises(ValueError, match="must be positive"):
        run_budgeted_auction(bids, clicks, rivals, budget=budget)
