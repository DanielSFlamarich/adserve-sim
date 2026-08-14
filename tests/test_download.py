# tests/test_download.py

"""Tests for raw fetch and stratified sample preparation."""

from pathlib import Path

import pandas as pd

from adserve_sim.data.download import prepare_sample
from adserve_sim.data.schema import TIMESTAMP, parse_hour


def test_prepare_sample_preserves_hourly_distribution(
    raw_frame: pd.DataFrame, tmp_path: Path
) -> None:
    """The sample must keep the shape of the traffic, not just its volume.

    Later steps split on day boundaries and model pacing against the daily
    volume curve, so a sample that flattened the hourly profile would quietly
    invalidate both.
    """
    raw_path = tmp_path / "train.csv"
    raw_frame.to_csv(raw_path, index=False)

    sample = prepare_sample(raw_path, tmp_path / "sample.parquet", sample_rows=1000)

    original_share = parse_hour(raw_frame["hour"]).value_counts(normalize=True).sort_index()
    sample_share = sample[TIMESTAMP].value_counts(normalize=True).sort_index()

    assert (original_share - sample_share).abs().max() < 0.01


def test_prepare_sample_output_is_chronological(raw_frame: pd.DataFrame, tmp_path: Path) -> None:
    """Replay iterates in file order, so the prepared file must be sorted."""
    raw_path = tmp_path / "train.csv"
    raw_frame.to_csv(raw_path, index=False)

    sample = prepare_sample(raw_path, tmp_path / "sample.parquet", sample_rows=1000)

    assert sample[TIMESTAMP].is_monotonic_increasing


def test_prepare_sample_is_deterministic(raw_frame: pd.DataFrame, tmp_path: Path) -> None:
    """Non-reproducible sampling makes every downstream comparison meaningless."""
    raw_path = tmp_path / "train.csv"
    raw_frame.to_csv(raw_path, index=False)

    first = prepare_sample(raw_path, tmp_path / "a.parquet", sample_rows=1000, seed=7)
    second = prepare_sample(raw_path, tmp_path / "b.parquet", sample_rows=1000, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_prepare_sample_keeps_all_rows_when_sample_exceeds_input(
    raw_frame: pd.DataFrame, tmp_path: Path
) -> None:
    """Asking for more rows than exist should be a no-op, not an error."""
    raw_path = tmp_path / "train.csv"
    raw_frame.to_csv(raw_path, index=False)

    sample = prepare_sample(raw_path, tmp_path / "sample.parquet", sample_rows=10_000)

    assert len(sample) == len(raw_frame)
