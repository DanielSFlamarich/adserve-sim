# src/adserve_sim/auction/ranking.py

"""Turn a click probability into a bid.

This is where the score stops being a ranking and becomes a price, which is the
whole subject of this project. The valuation is

    bid = value_per_click x pCTR x p(viewable)

The third term comes from reading the RUNA SDK: it ships an IAB Open Measurement
adapter, so viewability is *measured* on that platform.

The step from "measured" to "therefore part of the objective" is an inference,
not documentation, but if it holds, what a publisher maximises is viewable
eCPM and an impression nobody saw is worth nothing regardless of how likely
a click was.

Avazu carries no viewability labels, so ``p(viewable)`` cannot be fitted. It
enters as a **stated prior over slot position** instead, and the point of
:func:`viewability_prior` being a plain lookup table is that it can be varied and
the result re-measured. The question this can answer is not "how viewable is
each slot" but "how much does the answer change the outcome".

``value_per_click`` is likewise a parameter, not a measurement: Avazu has no
conversion or revenue data, so what a click is worth to an advertiser is
supplied from outside. Absolute currency figures produced downstream are
therefore not claims about money. Comparisons between models at a *fixed* value
are, since the constant cancels.
"""

from __future__ import annotations

import numpy as np

#: assumed probability that a slot is actually seen, by ``banner_pos``.
#:
#: invented, and deliberately so, see the module docstring. The ordering
#: encodes one weak assumption: position 0 is the most prominent slot and later
#: positions are progressively less likely to enter the viewport. Magnitudes are
#: chosen to span a plausible range rather than to be correct.
VIEWABILITY_PRIOR: dict[str, float] = {
    "0": 0.75,
    "1": 0.60,
    "2": 0.45,
    "3": 0.40,
    "4": 0.35,
    "5": 0.30,
    "7": 0.25,
}

#: applied to any slot position absent from the prior.
DEFAULT_VIEWABILITY: float = 0.50

#: assumed advertiser value of one click, in arbitrary currency units.
DEFAULT_VALUE_PER_CLICK: float = 1.0


def viewability_prior(
    banner_positions: np.ndarray, prior: dict[str, float] | None = None
) -> np.ndarray:
    """Look up the assumed viewability of each slot.

    Args:
        banner_positions: Slot positions as strings, one per impression.
        prior: Position-to-probability mapping. Defaults to
            :data:`VIEWABILITY_PRIOR`; pass a different one to test how much the
            assumption matters.

    Returns:
        Assumed viewability per impression.

    Raises:
        ValueError: If any prior value lies outside ``[0, 1]``.
    """
    table = VIEWABILITY_PRIOR if prior is None else prior

    if any(not 0.0 <= value <= 1.0 for value in table.values()):
        raise ValueError("(@_@) viewability priors must lie in [0, 1]")

    positions = np.asarray(banner_positions, dtype=object)
    return np.array(
        [float(table.get(str(position), DEFAULT_VIEWABILITY)) for position in positions]
    )


def expected_value(
    click_probability: np.ndarray,
    viewability: np.ndarray | float = 1.0,
    value_per_click: float = DEFAULT_VALUE_PER_CLICK,
) -> np.ndarray:
    """Expected revenue from one impression.

    Args:
        click_probability: Predicted probability of a click, per impression.
        viewability: Probability the slot is seen. Pass ``1.0`` to price on raw
            eCPM instead of viewable eCPM.
        value_per_click: Advertiser value of a click.

    Returns:
        Expected value per impression.

    Raises:
        ValueError: If ``value_per_click`` is negative or probabilities fall
            outside ``[0, 1]``.
    """
    if value_per_click < 0:
        raise ValueError(f"value_per_click must be non-negative, got {value_per_click}")

    p_click = np.asarray(click_probability, dtype=np.float64)
    p_view = np.asarray(viewability, dtype=np.float64)

    for name, array in (("click_probability", p_click), ("viewability", p_view)):
        if array.size and (array.min() < 0.0 or array.max() > 1.0):
            raise ValueError(f"{name} must lie in [0, 1]")

    return np.asarray(value_per_click * p_click * p_view)


def bid(
    click_probability: np.ndarray,
    banner_positions: np.ndarray | None = None,
    value_per_click: float = DEFAULT_VALUE_PER_CLICK,
    prior: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute a truthful bid per impression.

    Truthful in the second-price sense: the bid *is* the bidder's expected
    value, with no shading. That is the optimal strategy under second-price
    rules, and it is what makes the experiment clean. Any revenue difference
    between two models is attributable to their probabilities rather than to
    their bidding tactics.

    Args:
        click_probability: Predicted click probability, per impression.
        banner_positions: Slot positions. If omitted, viewability is ignored and
            the bid prices raw rather than viewable eCPM.
        value_per_click: Advertiser value of a click.
        prior: Optional viewability table override.

    Returns:
        Bid per impression, in the same units as ``value_per_click``.
    """
    if banner_positions is None:
        viewability: np.ndarray | float = 1.0
    else:
        viewability = viewability_prior(banner_positions, prior)

    return expected_value(click_probability, viewability, value_per_click)
