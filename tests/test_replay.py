# tests/test_replay.py

"""Tests for chronological replay and the label-withholding guarantee."""

import dataclasses

import pandas as pd
import pytest

from adserve_sim.data.schema import CATEGORICAL_COLUMNS, SchemaError
from adserve_sim.sim.replay import (
    AdRequest,
    check_chronological,
    hourly_volume,
    replay,
)


def test_replay_yields_one_pair_per_row(dated_frame: pd.DataFrame) -> None:
    assert len(list(replay(dated_frame))) == len(dated_frame)


def test_request_carries_no_label(dated_frame: pd.DataFrame) -> None:
    """The core guarantee: a policy handed an AdRequest cannot read the outcome.

    Asserted against the dataclass fields rather than one instance, so adding a
    label-bearing field later fails here rather than silently leaking.
    """
    field_names = {field.name for field in dataclasses.fields(AdRequest)}
    assert field_names == {"impression_id", "timestamp", "features"}

    request, _ = next(replay(dated_frame))
    assert "click" not in request.features


def test_request_features_cover_all_categoricals(dated_frame: pd.DataFrame) -> None:
    request, _ = next(replay(dated_frame))
    assert set(request.features) == set(CATEGORICAL_COLUMNS)


def test_request_features_are_immutable(dated_frame: pd.DataFrame) -> None:
    """A policy must not be able to mutate the request it was given."""
    request, _ = next(replay(dated_frame))
    with pytest.raises(TypeError):
        request.features["banner_pos"] = "tampered"  # type: ignore[index]


def test_outcome_matches_request_by_id(dated_frame: pd.DataFrame) -> None:
    for request, outcome in replay(dated_frame):
        assert request.impression_id == outcome.impression_id


def test_outcomes_preserve_base_rate(dated_frame: pd.DataFrame) -> None:
    clicks = sum(outcome.clicked for _, outcome in replay(dated_frame))
    assert clicks == int(dated_frame["click"].sum())


def test_replay_is_chronological(dated_frame: pd.DataFrame) -> None:
    timestamps = [request.timestamp for request, _ in replay(dated_frame)]
    assert timestamps == sorted(timestamps)


def test_replay_rejects_unsorted_input(dated_frame: pd.DataFrame) -> None:
    """Silently re-sorting would hide an upstream bug; the stream must refuse."""
    shuffled = dated_frame.sample(frac=1.0, random_state=0)
    with pytest.raises(SchemaError, match="not sorted"):
        next(replay(shuffled))


def test_replay_rejects_missing_columns(dated_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        next(replay(dated_frame.drop(columns=["banner_pos"])))


def test_check_chronological_requires_timestamp(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="parse_hour"):
        check_chronological(raw_frame)


def test_hourly_volume_totals_match_row_count(dated_frame: pd.DataFrame) -> None:
    volume = hourly_volume(dated_frame)
    assert volume.sum() == len(dated_frame)
    assert volume.index.is_monotonic_increasing


def test_hourly_volume_requires_timestamp(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="parse_hour"):
        hourly_volume(raw_frame)
