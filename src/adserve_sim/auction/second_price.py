# src/adserve_sim/auction/second_price.py

"""Clear a second-price auction and settle the money on both sides.

Who is who
----------

**A bidder.** A demand-side buyer with a click model, bidding own
expected value on each impression opportunity. pCTR is the thing under
study: when it is wrong, bid is wrong, and this module measures what that
costs.

**Rivals are synthetic.** Competing bids are drawn from a log-normal
distribution rather than modelled: see :func:`sample_competing_bids`. In reality
they would be the outputs of other buyers' own click models, so the distribution
stands in for the aggregate behaviour of a market you cannot observe.

**The publisher runs the auction** and sets the reserve price. It is not
modelled as an agent; it collects clearing prices.

A note on which side this is. The RUNA SDK that shaped this simulator's request
contract is publisher-side software, so a faithful simulation of *that* would put
us in the seller's chair. The buy-side framing is used here because the
calibration question is sharper from it: an over-bidding buyer pays real money
for impressions worth less than the price, measured against a known ground
truth. The publisher's side is reported too, since the same clearing prices
answer both.

Accounting
----------

Two perspectives, and they move in opposite directions:

- **Advertiser profit** = ``value_per_click x click - price_paid``, summed over
  auctions won. Miscalibration hurts here directly, over-bidding wins
  impressions that were not worth the price.
- **Publisher revenue** = sum of clearing prices across all auctions that
  cleared. A buyer bidding too high *raises* this.

So miscalibration does not simply destroy value; it transfers it. Reporting both
is what makes that visible.

Simplifications, stated
-----------------------

*One label per impression.* Avazu records whether the impression shown was
clicked. The simulator treats that outcome as a property of the *opportunity*
rather than of the winning creative, so whoever wins inherits the same click. In
reality different ads have different click rates, and modelling that would need
per-creative data the dataset does not carry.

*Fixed competitor count.* Every auction has the same number of rivals. Real
auctions vary, and thin auctions are where reserve prices bind hardest.

*One global reserve.* The RUNA SDK documents three ad formats: banner,
carousel, interstitial which are three kinds of inventory with different
attention profiles, and it is unlikely a real platform prices them off a single
floor. With server access the reserve would be swept per format, and probably
per placement within format. Avazu has no format column, so the nearest
structural analogue available is ``banner_pos``: slot position segments
inventory by something similar, and sweeping within each segment tests the same
argument by analogy. Whether the resulting optima differ is a question the
public data can answer; whether RUNA sets its floors that way is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Default number of rival bidders per auction.
DEFAULT_N_COMPETITORS: int = 5

#: Default log-scale of the competing-bid distribution.
#:
#: Larger values spread rivals wider, which makes both winning cheaply and
#: losing badly more common.
DEFAULT_BID_SIGMA: float = 0.6


@dataclass(frozen=True)
class AuctionOutcome:
    """Per-impression result of clearing one set of auctions.

    Attributes:
        won: Whether our bid took the impression.
        price_paid: What we paid, zero where we lost.
        clearing_price: What the winner paid, zero where the auction did not
            clear at all. Equals ``price_paid`` where we won.
        cleared: Whether any bid met the reserve.
    """

    won: np.ndarray
    price_paid: np.ndarray
    clearing_price: np.ndarray
    cleared: np.ndarray

    @property
    def win_rate(self) -> float:
        """Share of auctions we took."""
        return float(self.won.mean())

    @property
    def fill_rate(self) -> float:
        """Share of impressions that showed an ad from anyone.

        This is what a reserve price trades against: raising it lifts the price
        of the auctions that still clear while leaving more slots empty.
        """
        return float(self.cleared.mean())

    @property
    def publisher_revenue(self) -> float:
        """Total collected by the publisher across all auctions."""
        return float(self.clearing_price.sum())


@dataclass(frozen=True)
class Settlement:
    """What the auctions were worth, from both sides.

    Attributes:
        advertiser_profit: Click value earned minus prices paid, over auctions
            we won. Negative means we systematically overpaid.
        advertiser_spend: Total paid.
        advertiser_value: Total click value received.
        publisher_revenue: Total collected by the publisher, from every winner.
        impressions_won: How many auctions we took.
        clicks_won: How many of those were clicked.
        win_rate: Share of auctions we took.
        fill_rate: Share of impressions that showed any ad.
    """

    advertiser_profit: float
    advertiser_spend: float
    advertiser_value: float
    publisher_revenue: float
    impressions_won: int
    clicks_won: int
    win_rate: float
    fill_rate: float

    @property
    def roi(self) -> float:
        """Profit per unit spent. Zero spend returns zero rather than raising."""
        return self.advertiser_profit / self.advertiser_spend if self.advertiser_spend else 0.0

    @property
    def effective_cpc(self) -> float:
        """Average price paid per click received."""
        return self.advertiser_spend / self.clicks_won if self.clicks_won else 0.0


def sample_competing_bids(
    n_impressions: int,
    reference_bid: float,
    n_competitors: int = DEFAULT_N_COMPETITORS,
    sigma: float = DEFAULT_BID_SIGMA,
    seed: int = 42,
) -> np.ndarray:
    """Draw rival bids from a log-normal distribution.

    Log-normal because bids are positive and right-skewed: most cluster around a
    typical value, a few go much higher. It is an assumption, not a fit, the
    dataset contains no competing bids so the parameters are chosen to place
    rivals in the same neighbourhood as our own bids rather than to be correct.

    Centring on ``reference_bid`` matters. If rivals were fixed in absolute
    terms, a systematically over-bidding model would win everything and the
    experiment would measure the arbitrary gap between two scales rather than the
    cost of miscalibration.

    Args:
        n_impressions: Number of auctions.
        reference_bid: Median of the rival distribution, normally the median of
            an undistorted bid vector.
        n_competitors: Rivals per auction.
        sigma: Log-scale of the distribution.
        seed: Seed for reproducibility.

    Returns:
        Array of shape ``(n_impressions, n_competitors)``.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if n_impressions < 1 or n_competitors < 1:
        raise ValueError("need at least one impression and one competitor")
    if reference_bid <= 0 or sigma <= 0:
        raise ValueError("reference_bid and sigma must be positive")

    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.lognormal(mean=np.log(reference_bid), sigma=sigma, size=(n_impressions, n_competitors))
    )


def clear_second_price(
    our_bids: np.ndarray,
    competing_bids: np.ndarray,
    reserve: float = 0.0,
) -> AuctionOutcome:
    """Run one second-price auction per impression.

    Highest bid wins if it meets the reserve; the winner pays the *runner-up's*
    bid, floored at the reserve. Paying the runner-up rather than your own bid
    is what makes truthful bidding optimal, and therefore what lets a revenue
    difference between two models be attributed to their probabilities rather
    than to bidding tactics.

    Ties go to the rivals, which is arbitrary but consistent and avoids
    flattering our own bidder.

    Args:
        our_bids: Our bid per impression.
        competing_bids: Rival bids, shape ``(n_impressions, n_competitors)``.
        reserve: Price floor. Auctions where no bid meets it do not clear.

    Returns:
        Per-impression outcome.

    Raises:
        ValueError: If shapes disagree, the reserve is negative, or any bid is
            negative.
    """
    ours = np.asarray(our_bids, dtype=np.float64)
    rivals = np.asarray(competing_bids, dtype=np.float64)

    if rivals.ndim != 2:
        raise ValueError(f"competing_bids must be 2-dimensional, got shape {rivals.shape}")
    if ours.shape[0] != rivals.shape[0]:
        raise ValueError(f"shape mismatch: {ours.shape[0]} bids, {rivals.shape[0]} auctions")
    if reserve < 0:
        raise ValueError(f"reserve must be non-negative, got {reserve}")
    if ours.size and ours.min() < 0:
        raise ValueError("bids must be non-negative")

    best_rival = rivals.max(axis=1)
    second_rival = np.sort(rivals, axis=1)[:, -2] if rivals.shape[1] > 1 else np.zeros(len(ours))

    # Ties go to the rivals: we need to beat the field, not merely match it.
    we_lead = ours > best_rival

    highest = np.where(we_lead, ours, best_rival)
    runner_up = np.where(we_lead, best_rival, np.maximum(ours, second_rival))

    cleared = highest >= reserve
    clearing_price = np.where(cleared, np.maximum(runner_up, reserve), 0.0)

    won = we_lead & cleared
    price_paid = np.where(won, clearing_price, 0.0)

    return AuctionOutcome(
        won=won,
        price_paid=price_paid,
        clearing_price=clearing_price,
        cleared=cleared,
    )


def settle(
    outcome: AuctionOutcome,
    clicks: np.ndarray,
    value_per_click: float,
) -> Settlement:
    """Convert auction outcomes into money, from both sides.

    Only auctions we won generate advertiser value: an impression we lost was
    someone else's click. Publisher revenue counts every auction that cleared,
    ours or not.

    Args:
        outcome: Result of :func:`clear_second_price`.
        clicks: Observed click outcome per impression, 0 or 1.
        value_per_click: Advertiser value of a click.

    Returns:
        The settlement.

    Raises:
        ValueError: If lengths disagree or ``value_per_click`` is negative.
    """
    y = np.asarray(clicks, dtype=np.float64)

    if y.shape != outcome.won.shape:
        raise ValueError(f"shape mismatch: {y.shape} clicks, {outcome.won.shape} auctions")
    if value_per_click < 0:
        raise ValueError(f"value_per_click must be non-negative, got {value_per_click}")

    clicks_won = float((y * outcome.won).sum())
    value = value_per_click * clicks_won
    spend = float(outcome.price_paid.sum())

    settlement = Settlement(
        advertiser_profit=value - spend,
        advertiser_spend=spend,
        advertiser_value=value,
        publisher_revenue=outcome.publisher_revenue,
        impressions_won=int(outcome.won.sum()),
        clicks_won=int(clicks_won),
        win_rate=outcome.win_rate,
        fill_rate=outcome.fill_rate,
    )

    logger.info(
        "won %d of %d auctions, profit %.2f, publisher revenue %.2f",
        settlement.impressions_won,
        len(y),
        settlement.advertiser_profit,
        settlement.publisher_revenue,
    )
    return settlement


def run_auction(
    our_bids: np.ndarray,
    clicks: np.ndarray,
    competing_bids: np.ndarray,
    reserve: float = 0.0,
    value_per_click: float = 1.0,
) -> Settlement:
    """Clear and settle in one call.

    Args:
        our_bids: Our bid per impression.
        clicks: Observed click outcome per impression.
        competing_bids: Rival bids, shape ``(n_impressions, n_competitors)``.
        reserve: Price floor.
        value_per_click: Advertiser value of a click.

    Returns:
        The settlement.
    """
    outcome = clear_second_price(our_bids, competing_bids, reserve)
    return settle(outcome, clicks, value_per_click)
