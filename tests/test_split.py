# tests/test_split.py

"""Tests for day-boundary temporal splitting and its leakage guards."""

import datetime as dt

import pandas as pd
import pytest

from adserve_sim.data.schema import TIMESTAMP, SchemaError
from adserve_sim.data.split import (
    LeakageError,
    TemporalSplit,
    split_at,
    split_by_day,
)


def test_split_by_day_partitions_all_rows(dated_frame: pd.DataFrame) -> None:
    """Every row lands in exactly one partition; none is lost or duplicated."""
    split = split_by_day(dated_frame, val_days=1, test_days=1)
    assert sum(split.sizes.values()) == len(dated_frame)


def test_split_by_day_is_strictly_ordered(dated_frame: pd.DataFrame) -> None:
    """The central invariant: no partition may see a later moment than the next."""
    split = split_by_day(dated_frame, val_days=1, test_days=1)

    assert split.train[TIMESTAMP].max() < split.validation[TIMESTAMP].min()
    assert split.validation[TIMESTAMP].max() < split.test[TIMESTAMP].min()


def test_split_by_day_reserves_trailing_days(dated_frame: pd.DataFrame) -> None:
    """Windows are counted backwards, so test always ends at the data's end."""
    split = split_by_day(dated_frame, val_days=1, test_days=1)

    assert split.test[TIMESTAMP].max() == dated_frame[TIMESTAMP].max()
    assert split.train[TIMESTAMP].min() == dated_frame[TIMESTAMP].min()


def test_split_by_day_rejects_insufficient_history(dated_frame: pd.DataFrame) -> None:
    """The fixture spans three days, so a 2+2 request cannot leave training data."""
    with pytest.raises(LeakageError, match="day"):
        split_by_day(dated_frame, val_days=2, test_days=2)


def test_split_by_day_requires_parsed_timestamp(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="parse_hour"):
        split_by_day(raw_frame)


@pytest.mark.parametrize(("val_days", "test_days"), [(0, 1), (1, 0), (-1, 1)])
def test_split_by_day_rejects_nonpositive_windows(
    dated_frame: pd.DataFrame, val_days: int, test_days: int
) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        split_by_day(dated_frame, val_days=val_days, test_days=test_days)


def test_split_at_uses_explicit_boundaries(dated_frame: pd.DataFrame) -> None:
    split = split_at(dated_frame, dt.date(2014, 10, 22), dt.date(2014, 10, 23))

    assert split.train[TIMESTAMP].max() < pd.Timestamp("2014-10-22")
    assert split.test[TIMESTAMP].min() >= pd.Timestamp("2014-10-23")


def test_split_at_rejects_inverted_boundaries(dated_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must be before"):
        split_at(dated_frame, dt.date(2014, 10, 23), dt.date(2014, 10, 22))


def test_construction_rejects_overlapping_partitions(dated_frame: pd.DataFrame) -> None:
    """A leaky split must be impossible to build, not merely detectable later."""
    with pytest.raises(LeakageError, match="not before"):
        TemporalSplit(train=dated_frame, validation=dated_frame, test=dated_frame)


def test_construction_rejects_empty_partition(dated_frame: pd.DataFrame) -> None:
    empty = dated_frame.iloc[:0]
    with pytest.raises(LeakageError, match="empty"):
        TemporalSplit(train=empty, validation=dated_frame, test=dated_frame)


def test_construction_rejects_shared_impression_ids(dated_frame: pd.DataFrame) -> None:
    """Clean timestamps but a shared id means a row was emitted twice."""
    split = split_by_day(dated_frame, val_days=1, test_days=1)
    duplicated = pd.concat([split.validation, split.train.head(1)], ignore_index=True)

    with pytest.raises(LeakageError):
        TemporalSplit(train=split.train, validation=duplicated, test=split.test)


def test_base_rates_are_reported_per_partition(dated_frame: pd.DataFrame) -> None:
    split = split_by_day(dated_frame, val_days=1, test_days=1)
    rates = split.base_rates

    assert set(rates) == {"train", "validation", "test"}
    assert all(0.0 <= value <= 1.0 for value in rates.values())
