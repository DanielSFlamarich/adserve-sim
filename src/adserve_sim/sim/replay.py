# src/adserve_sim/sim/replay.py

"""Replay a prepared frame as a chronological stream of ad requests.

The simulator's central discipline is that a policy sees only what a live ad
server would see at decision time. :class:`AdRequest` therefore carries features
and no label; the outcome is a separate object the policy is not handed. Keeping
the two apart in the type system makes the leakage-free path the easy one to
write and the leaky one awkward.

Rows are iterated in file order, which the preparation step guarantees to be
chronological and :func:`check_chronological` re-verifies here rather than
trusting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from adserve_sim.data.schema import (
    CATEGORICAL_COLUMNS,
    IMPRESSION_ID,
    LABEL,
    TIMESTAMP,
    SchemaError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdRequest:
    """One opportunity to serve an ad, as visible at decision time.

    Attributes:
        impression_id: Opaque identifier, used to join outcomes back on.
        timestamp: When the request arrived.
        features: Read-only mapping of feature name to value.
    """

    impression_id: str
    timestamp: pd.Timestamp
    features: Mapping[str, str]


@dataclass(frozen=True)
class Outcome:
    """What actually happened, withheld from the policy until after it decides.

    Attributes:
        impression_id: Matches the corresponding :class:`AdRequest`.
        clicked: Whether the impression was clicked.
    """

    impression_id: str
    clicked: bool


def _require_columns(frame: pd.DataFrame) -> None:
    """Raise if the frame lacks the columns replay depends on.

    Args:
        frame: The frame to check.

    Raises:
        SchemaError: If any required column is absent.
    """
    required = {IMPRESSION_ID, TIMESTAMP, LABEL, *CATEGORICAL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise SchemaError(f"(@_@) frame is missing columns needed for replay: {sorted(missing)}")


def check_chronological(frame: pd.DataFrame) -> None:
    """Verify the frame is sorted by timestamp.

    Replay trusts file order rather than re-sorting, so an unsorted input would
    quietly produce a stream that revisits the past - and any pacing or bandit
    logic reading that stream would be learning from a timeline that never
    happened.

    Args:
        frame: A frame carrying a parsed timestamp column.

    Raises:
        SchemaError: If the timestamp column is absent or out of order.
    """
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"frame has no {TIMESTAMP!r} column; call parse_hour first")

    if not frame[TIMESTAMP].is_monotonic_increasing:
        raise SchemaError("(@_@) frame is not sorted by timestamp; replay requires file order")


def replay(frame: pd.DataFrame) -> Iterator[tuple[AdRequest, Outcome]]:
    """Yield each row as a request/outcome pair, in chronological order.

    The pair is yielded together for convenience, but the two objects are
    distinct so a policy can be handed the request alone.

    Args:
        frame: A prepared, chronologically sorted frame.

    Yields:
        A ``(request, outcome)`` pair per impression.

    Raises:
        SchemaError: If required columns are absent or the frame is unsorted.
    """
    _require_columns(frame)
    check_chronological(frame)

    feature_columns = list(CATEGORICAL_COLUMNS)

    # zipping column iterators rather than using itertuples keeps the stream
    # lazy while staying legible to the type checker.
    columns = zip(
        frame[IMPRESSION_ID],
        frame[TIMESTAMP],
        frame[LABEL],
        *(frame[name] for name in feature_columns),
        strict=True,
    )

    for impression_id, timestamp, label, *values in columns:
        request = AdRequest(
            impression_id=str(impression_id),
            timestamp=timestamp,
            features=MappingProxyType(
                {name: str(value) for name, value in zip(feature_columns, values, strict=True)}
            ),
        )
        outcome = Outcome(impression_id=str(impression_id), clicked=bool(label))

        yield request, outcome


def hourly_volume(frame: pd.DataFrame) -> pd.Series:
    """Return the request count per hour.

    This is the arrival curve a pacing controller has to spend against, so it
    is worth inspecting directly rather than assuming it is flat.

    Args:
        frame: A frame carrying a parsed timestamp column.

    Returns:
        Counts indexed by hour, ascending.

    Raises:
        SchemaError: If the timestamp column is absent.
    """
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"(@o@) frame has no {TIMESTAMP!r} column; call parse_hour first")

    return frame[TIMESTAMP].value_counts().sort_index()
