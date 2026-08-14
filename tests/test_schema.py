# tests/test_schema.py

"""Tests for the Avazu column contract and timestamp parsing."""

import pandas as pd
import pytest

from adserve_sim.data.schema import (
    SchemaError,
    hour_bounds,
    parse_hour,
    validate_columns,
)


def test_validate_columns_accepts_expected_schema(raw_frame: pd.DataFrame) -> None:
    validate_columns(raw_frame)


def test_validate_columns_rejects_missing_column(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing"):
        validate_columns(raw_frame.drop(columns=["click"]))


def test_validate_columns_rejects_extra_column(raw_frame: pd.DataFrame) -> None:
    raw_frame["surprise"] = 1
    with pytest.raises(SchemaError, match="unexpected"):
        validate_columns(raw_frame)


def test_parse_hour_decodes_yymmddhh(raw_frame: pd.DataFrame) -> None:
    parsed = parse_hour(raw_frame["hour"])
    assert parsed.min() == pd.Timestamp("2014-10-21 00:00")
    assert parsed.max() == pd.Timestamp("2014-10-23 23:00")


def test_parse_hour_rejects_unparseable_values() -> None:
    """Bad timestamps must stop the pipeline, not silently become NaT.

    A coerced NaT would vanish from a later day-boundary filter without
    appearing in any count, which is the kind of loss that only surfaces as an
    unexplained metric shift much further downstream.
    """
    with pytest.raises(SchemaError, match="non-parseable"):
        parse_hour(pd.Series(["not-a-date"]))


def test_hour_bounds_requires_parsed_timestamp(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="parse_hour"):
        hour_bounds(raw_frame)
