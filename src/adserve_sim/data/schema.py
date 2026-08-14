# src/adserve_sim/data/schema.py

"""Column contract for the Avazu CTR dataset.

The Avazu training file is a flat CSV with 24 columns. This module pins the
expected column names and dtypes so that a schema drift in the raw file fails
loudly at load time rather than silently producing a different feature space.

Reference: https://www.kaggle.com/competitions/avazu-ctr-prediction/data
"""

from __future__ import annotations

import datetime as dt
from typing import Final

import pandas as pd

#: click label column.
LABEL: Final[str] = "click"

#: raw timestamp column, encoded as ``YYMMDDHH`` (e.g. ``14102100``).
RAW_HOUR: Final[str] = "hour"

#: parsed timestamp column added by :func:`parse_hour`.
TIMESTAMP: Final[str] = "timestamp"

#: identifier column, unique per impression. Not a feature.
IMPRESSION_ID: Final[str] = "id"

#: high-cardinality categorical columns.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
)

#: full expected column set of the raw training file, in file order.
RAW_COLUMNS: Final[tuple[str, ...]] = (
    IMPRESSION_ID,
    LABEL,
    RAW_HOUR,
    *CATEGORICAL_COLUMNS,
)

#: dtypes used when reading the raw CSV.
#:
#: every categorical is read as ``str``: the values are anonymised hashes, and
#: reading them as integers both loses leading zeros and invites accidental
#: arithmetic on what are really opaque identifiers.
RAW_DTYPES: Final[dict[str, str]] = {
    IMPRESSION_ID: "str",
    LABEL: "int8",
    RAW_HOUR: "str",
    **{column: "str" for column in CATEGORICAL_COLUMNS},
}


class SchemaError(ValueError):
    """Raised when a loaded frame does not match the expected Avazu contract."""


def validate_columns(frame: pd.DataFrame) -> None:
    """Check that ``frame`` carries exactly the expected raw columns.

    Args:
        frame: A frame loaded from the raw Avazu CSV.

    Raises:
        SchemaError: If any expected column is missing or unexpected columns
            are present.
    """
    actual = set(frame.columns)
    expected = set(RAW_COLUMNS)

    missing = expected - actual
    if missing:
        raise SchemaError(f"missing expected columns (@_@): {sorted(missing)}")

    unexpected = actual - expected
    if unexpected:
        raise SchemaError(f"unexpected columns (@o@): {sorted(unexpected)}")


def parse_hour(raw: pd.Series) -> pd.Series:
    """Convert the ``YYMMDDHH`` hour encoding into timezone-naive timestamps.

    Avazu encodes time as an eight-character string, e.g. ``14102100`` for
    2014-10-21 00:00. The dataset carries no timezone, so timestamps are left
    naive and treated as a single consistent clock.

    Args:
        raw: The raw ``hour`` column as strings.

    Returns:
        A series of ``datetime64[ns]`` values, hour-resolution.

    Raises:
        SchemaError: If any value fails to parse.
    """
    parsed = pd.to_datetime(raw, format="%y%m%d%H", errors="coerce")

    if parsed.isna().any():
        bad = raw[parsed.isna()].unique()[:5]
        raise SchemaError(f"(@_@) non-parseable hour values, first few: {list(bad)}")

    return parsed


def hour_bounds(frame: pd.DataFrame) -> tuple[dt.datetime, dt.datetime]:
    """Return the earliest and latest timestamp present in ``frame``.

    Used by the split and replay layers to assert that no step has silently
    reordered or dropped time coverage.

    Args:
        frame: A frame carrying a parsed :data:`TIMESTAMP` column.

    Returns:
        A ``(minimum, maximum)`` pair of timestamps.

    Raises:
        SchemaError: If the timestamp column is absent.
    """
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"(@_@) frame has no {TIMESTAMP!r} column; call parse_hour first")

    return frame[TIMESTAMP].min(), frame[TIMESTAMP].max()
