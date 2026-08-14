# tests/conftest.py

"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest

from adserve_sim.data.schema import RAW_COLUMNS


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small frame with the same shape as the raw Avazu training file.

    Spans three full days at hourly resolution so that day-boundary logic in
    the split layer has something meaningful to cut on.
    """
    rng = np.random.default_rng(0)
    n = 2000
    hours = [f"1410{day:02d}{hour:02d}" for day in range(21, 24) for hour in range(24)]

    data: dict[str, object] = {
        column: rng.integers(0, 50, n).astype(str)
        for column in RAW_COLUMNS
        if column not in ("id", "click", "hour")
    }
    data["id"] = [str(i) for i in range(n)]
    data["click"] = rng.binomial(1, 0.17, n).astype("int8")
    data["hour"] = rng.choice(hours, n)

    return pd.DataFrame(data)[list(RAW_COLUMNS)]
