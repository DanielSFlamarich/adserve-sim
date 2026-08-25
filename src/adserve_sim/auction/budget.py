# src/adserve_sim/auction/budget.py

"""Apply a spending limit to a sequence of auction wins.

Without a budget, the auction results say something surprising: over-predicting
click probability is *more* profitable than being honest. That is not a bug. In
a second-price auction the winner pays the runner-up's bid, so bidding above
your true value wins extra auctions at prices someone else sets. As long as
those prices stay below what the impression was actually worth, each extra win
adds profit. Efficiency falls (ROI drops sharply) but with unlimited money,
volume more than compensates.

Unlimited money was an unstated assumption doing a great deal of work. Real
advertisers have a cap, and once spending is capped the arithmetic inverts:
buying impressions at poor value uses up money that is then unavailable for good
ones. The cost of miscalibration is not what you overpay, it is what you can no
longer afford.

Two strategies are provided, and the difference between them is the whole point
of pacing:

*Exhaust and stop* bids on everything until the money runs out, then goes dark.
Simple, and it buys the *earliest* impressions rather than the best ones.

*Probabilistic pacing* enters each auction with some probability, chosen so that
spend is spread across the whole period. It sacrifices early opportunities in
exchange for still being in the market later, which matters when traffic quality
varies over the day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from adserve_sim.auction.second_price import AuctionOutcome, Settlement, clear_second_price, settle

logger = logging.getLogger(__name__)

#: default share of auctions entered under probabilistic pacing.
#:
#: set from the budget and expected spend rather than fixed; this is only the
#: fallback when the budget is large enough that no throttling is needed.
FULL_PARTICIPATION: float = 1.0


@dataclass(frozen=True)
class BudgetedOutcome:
    """An auction outcome truncated by a spending limit.

    Attributes:
        outcome: The surviving wins, with everything after exhaustion removed.
        budget: The limit applied.
        spent: What was actually spent, at most ``budget``.
        exhausted_at: Index of the impression at which the money ran out, or
            ``None`` if it never did. The position of this within the stream is
            the pacing story: a bidder that exhausts at 20% of the way through
            spent the rest of the period unable to bid.
        participation_rate: Share of auctions entered, below 1 only under
            throttling.
    """

    outcome: AuctionOutcome
    budget: float
    spent: float
    exhausted_at: int | None
    participation_rate: float

    @property
    def exhausted_fraction(self) -> float | None:
        """How far through the stream the budget ran out, as a fraction."""
        if self.exhausted_at is None:
            return None
        return self.exhausted_at / len(self.outcome.won)


def _truncate_at_budget(
    outcome: AuctionOutcome, budget: float
) -> tuple[AuctionOutcome, int | None]:
    """Drop every win after cumulative spend would exceed the budget.

    Wins are processed in stream order, which is chronological because the
    prepared sample is sorted by timestamp. A win that would breach the limit is
    dropped along with everything after it, the bidder is out of money, not
    selectively skipping expensive auctions.

    Args:
        outcome: Unconstrained auction result.
        budget: Spending limit.

    Returns:
        The truncated outcome and the index where the money ran out, or ``None``.
    """
    cumulative = np.cumsum(outcome.price_paid)

    if cumulative[-1] <= budget:
        return outcome, None

    exhausted_at = int(np.searchsorted(cumulative, budget, side="right"))

    affordable = np.zeros_like(outcome.won)
    affordable[:exhausted_at] = outcome.won[:exhausted_at]

    truncated = AuctionOutcome(
        won=affordable,
        price_paid=np.where(affordable, outcome.price_paid, 0.0),
        clearing_price=outcome.clearing_price,
        cleared=outcome.cleared,
    )
    return truncated, exhausted_at


def run_budgeted_auction(
    our_bids: np.ndarray,
    clicks: np.ndarray,
    competing_bids: np.ndarray,
    budget: float,
    reserve: float = 0.0,
    value_per_click: float = 1.0,
    pace: bool = False,
    seed: int = 42,
) -> tuple[Settlement, BudgetedOutcome]:
    """Clear, cap spending at a budget, and settle.

    Args:
        our_bids: Our bid per impression, in stream order.
        clicks: Observed click outcome per impression.
        competing_bids: Rival bids, shape ``(n_impressions, n_competitors)``.
        budget: Total spending limit.
        reserve: Price floor.
        value_per_click: Advertiser value of a click.
        pace: If true, throttle participation so spending is spread across the
            whole stream instead of front-loaded. See :func:`pacing_rate`.
        seed: Seed for the throttling draw.

    Returns:
        The settlement and the budgeted outcome.

    Raises:
        ValueError: If the budget is not positive.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    outcome = clear_second_price(our_bids, competing_bids, reserve)

    participation = FULL_PARTICIPATION
    if pace:
        participation = pacing_rate(outcome, budget)
        rng = np.random.default_rng(seed)
        entered = rng.random(len(our_bids)) < participation

        outcome = AuctionOutcome(
            won=outcome.won & entered,
            price_paid=np.where(entered, outcome.price_paid, 0.0),
            clearing_price=outcome.clearing_price,
            cleared=outcome.cleared,
        )

    truncated, exhausted_at = _truncate_at_budget(outcome, budget)
    spent = float(truncated.price_paid.sum())

    budgeted = BudgetedOutcome(
        outcome=truncated,
        budget=budget,
        spent=spent,
        exhausted_at=exhausted_at,
        participation_rate=participation,
    )

    if exhausted_at is not None:
        logger.info(
            "budget exhausted at impression %d of %d (%.1f%% through)",
            exhausted_at,
            len(our_bids),
            100 * exhausted_at / len(our_bids),
        )

    return settle(truncated, clicks, value_per_click), budgeted


def pacing_rate(outcome: AuctionOutcome, budget: float) -> float:
    """Share of auctions to enter so that spend lasts the whole stream.

    The simplest possible controller: if unconstrained spending would be three
    times the budget, enter one auction in three. It assumes the spend rate is
    roughly constant, which the hourly traffic curve says is not quite true, a
    real pacer would use feedback on realised spend rather than a fixed rate set
    up front.

    Args:
        outcome: Unconstrained auction result.
        budget: Spending limit.

    Returns:
        Participation probability in ``(0, 1]``.
    """
    unconstrained = float(outcome.price_paid.sum())

    if unconstrained <= budget:
        return FULL_PARTICIPATION

    return budget / unconstrained
