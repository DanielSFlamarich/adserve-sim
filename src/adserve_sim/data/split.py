# src/adserve_sim/data/split.py

"""Day-boundary temporal splitting of the prepared Avazu sample.

Ad serving is a forecasting problem: a model fitted today is scored on traffic
that has not happened yet. A shuffled split silently breaks that premise, since
rows from the same hour land on both sides of the boundary and the offline
metric flatters a model that would fail online.

Every split here cuts on a whole-day boundary, and the leakage check runs in
:meth:`TemporalSplit.__post_init__` rather than being offered as an optional
call - an invalid split cannot be constructed in the first place.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from adserve_sim.data.schema import IMPRESSION_ID, LABEL, TIMESTAMP, SchemaError

logger = logging.getLogger(__name__)

#: default number of trailing days reserved for the test window.
DEFAULT_TEST_DAYS: int = 2

#: default number of days reserved for validation, immediately before the test window.
DEFAULT_VAL_DAYS: int = 1


class LeakageError(AssertionError):
    """Raised when partitions overlap in time or share impressions.

    Subclasses :class:`AssertionError` deliberately: this signals a violated
    invariant, not a condition a caller should catch and work around.
    """


@dataclass(frozen=True)
class TemporalSplit:
    """A chronologically ordered train/validation/test partition.

    Attributes:
        train: Earliest window, used for fitting.
        validation: Middle window, used for early stopping and calibration.
        test: Latest window, touched once at the end.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def __post_init__(self) -> None:
        """Enforce temporal ordering at construction time."""
        _assert_no_leakage(self)

    @property
    def sizes(self) -> dict[str, int]:
        """Row count of each partition."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def base_rates(self) -> dict[str, float]:
        """Mean click rate of each partition.

        Drift between partitions is not necessarily a bug - real traffic does
        shift - but it should be seen before any offline metric is interpreted,
        since a base-rate shift alone will move log loss and calibration.
        """
        return {
            "train": float(self.train[LABEL].mean()),
            "validation": float(self.validation[LABEL].mean()),
            "test": float(self.test[LABEL].mean()),
        }


def _assert_no_leakage(split: TemporalSplit) -> None:
    """Check partitions are non-empty, time-ordered, and share no impressions.

    Two independent checks, because they fail differently: a timestamp overlap
    means the boundary logic is wrong, while a shared impression id with clean
    timestamps means the same row was emitted twice.

    Args:
        split: The partition to check.

    Raises:
        LeakageError: If any check fails.
    """
    partitions = (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    )

    for name, frame in partitions:
        if frame.empty:
            raise LeakageError(f"partition {name!r} is empty; check the day counts")

    # pairwise over consecutive partitions, so the shorter tail is intended.
    for (earlier_name, earlier), (later_name, later) in zip(
        partitions, partitions[1:], strict=False
    ):
        earlier_end = earlier[TIMESTAMP].max()
        later_start = later[TIMESTAMP].min()
        if earlier_end >= later_start:
            raise LeakageError(
                f"{earlier_name} ends at {earlier_end}, which is not before "
                f"{later_name} starting at {later_start}"
            )

    ids = {name: set(frame[IMPRESSION_ID]) for name, frame in partitions}
    for left, right in (("train", "validation"), ("validation", "test"), ("train", "test")):
        shared = ids[left] & ids[right]
        if shared:
            raise LeakageError(f"{len(shared)} impression id(s) appear in both {left} and {right}")


def _day_of(timestamps: pd.Series) -> pd.Series:
    """Return the midnight-normalised calendar day of each timestamp."""
    normalised: pd.Series = timestamps.dt.normalize()
    return normalised


def _build(frame: pd.DataFrame, val_start: pd.Timestamp, test_start: pd.Timestamp) -> TemporalSplit:
    """Slice ``frame`` at two day boundaries into a validated split.

    Intervals are half-open, so every row lands in exactly one partition.

    Args:
        frame: A frame carrying a parsed :data:`TIMESTAMP` column.
        val_start: First day of the validation window.
        test_start: First day of the test window.

    Returns:
        The partitioned split.
    """
    day = _day_of(frame[TIMESTAMP])

    split = TemporalSplit(
        train=frame[day < val_start].reset_index(drop=True),
        validation=frame[(day >= val_start) & (day < test_start)].reset_index(drop=True),
        test=frame[day >= test_start].reset_index(drop=True),
    )

    logger.info("split sizes %s, base rates %s", split.sizes, split.base_rates)
    return split


def split_by_day(
    frame: pd.DataFrame,
    val_days: int = DEFAULT_VAL_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
) -> TemporalSplit:
    """Reserve the final days of ``frame`` for evaluation.

    Counting backwards from the end keeps the evaluation windows stable as the
    sample grows: adding earlier history moves rows into training without
    reshaping validation or test.

    Args:
        frame: A frame carrying a parsed :data:`TIMESTAMP` column.
        val_days: Number of whole days reserved for validation.
        test_days: Number of trailing whole days reserved for test.

    Returns:
        The partitioned split.

    Raises:
        SchemaError: If the timestamp column is absent.
        ValueError: If either day count is below one.
        LeakageError: If the frame spans too few days for the request.
    """
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"frame has no {TIMESTAMP!r} column; call parse_hour first")
    if val_days < 1 or test_days < 1:
        raise ValueError(f"val_days and test_days must be >= 1, got {val_days} and {test_days}")

    days = sorted(_day_of(frame[TIMESTAMP]).unique())
    required = val_days + test_days + 1
    if len(days) < required:
        raise LeakageError(
            f"frame spans {len(days)} day(s) but {required} are needed for a "
            f"{val_days}-day validation and {test_days}-day test window"
        )

    return _build(
        frame,
        val_start=pd.Timestamp(days[-(test_days + val_days)]),
        test_start=pd.Timestamp(days[-test_days]),
    )


def split_at(frame: pd.DataFrame, val_start: dt.date, test_start: dt.date) -> TemporalSplit:
    """Partition ``frame`` at explicit calendar boundaries.

    Preferred over :func:`split_by_day` when boundaries must stay fixed across
    runs on differently-sized samples: day counts shift with coverage, dates
    do not.

    Args:
        frame: A frame carrying a parsed :data:`TIMESTAMP` column.
        val_start: First day of the validation window.
        test_start: First day of the test window.

    Returns:
        The partitioned split.

    Raises:
        SchemaError: If the timestamp column is absent.
        ValueError: If ``val_start`` is not strictly before ``test_start``.
    """
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"frame has no {TIMESTAMP!r} column; call parse_hour first")
    if val_start >= test_start:
        raise ValueError(f"val_start {val_start} must be before test_start {test_start}")

    return _build(frame, pd.Timestamp(val_start), pd.Timestamp(test_start))
