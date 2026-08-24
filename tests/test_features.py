# tests/test_features.py

"""Tests for time features and out-of-fold target encoding."""

import numpy as np
import pandas as pd
import pytest

from adserve_sim.data.schema import LABEL, TIMESTAMP, SchemaError
from adserve_sim.features.build import (
    TIME_FEATURES,
    OutOfFoldTargetEncoder,
    add_time_features,
    feature_columns,
)


def test_add_time_features_creates_cyclical_columns(dated_frame: pd.DataFrame) -> None:
    out = add_time_features(dated_frame)

    assert set(TIME_FEATURES) <= set(out.columns)
    assert out["hour_of_day"].between(0, 23).all()
    assert out["day_of_week"].between(0, 6).all()


def test_add_time_features_does_not_mutate_input(dated_frame: pd.DataFrame) -> None:
    before = list(dated_frame.columns)
    add_time_features(dated_frame)
    assert list(dated_frame.columns) == before


def test_add_time_features_requires_timestamp(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="parse_hour"):
        add_time_features(raw_frame)


def test_encoder_produces_one_column_per_categorical(dated_frame: pd.DataFrame) -> None:
    encoder = OutOfFoldTargetEncoder(columns=("banner_pos", "site_id"))
    encoded = encoder.fit_transform(dated_frame)

    assert list(encoded.columns) == ["banner_pos_te", "site_id_te"]
    assert len(encoded) == len(dated_frame)


def test_encoder_maps_unseen_categories_to_prior(dated_frame: pd.DataFrame) -> None:
    """Categories absent from training must fall back, not produce nulls."""
    encoder = OutOfFoldTargetEncoder(columns=("banner_pos",)).fit(dated_frame)

    unseen = pd.DataFrame({"banner_pos": ["never-seen-before"]})
    encoded = encoder.transform(unseen)

    assert encoded["banner_pos_te"].iloc[0] == pytest.approx(encoder.prior_)


def test_encoder_smooths_rare_categories_toward_prior(dated_frame: pd.DataFrame) -> None:
    """A category seen once with a click is weak evidence, not a 100% click rate."""
    frame = dated_frame.copy()
    frame.loc[frame.index[0], "banner_pos"] = "singleton"
    frame.loc[frame.index[0], LABEL] = 1

    encoder = OutOfFoldTargetEncoder(columns=("banner_pos",), smoothing=20.0).fit(frame)
    encoding = encoder.mappings_["banner_pos"]["singleton"]

    assert encoding < 0.5
    assert encoding > encoder.prior_


def test_out_of_fold_encoding_does_not_leak_labels(dated_frame: pd.DataFrame) -> None:
    """The central guarantee, tested where naive encoding fails worst.

    A column unique to each row would, under naive target encoding, take the
    value of that row's own label - a perfect predictor in training and useless
    in production. Out-of-fold encoding removes the information entirely: since
    each category appears in exactly one fold, a held-out row never finds its
    own category in the folds used to build its encoding, so every row falls
    back to the prior. A constant column carries no signal, which is the
    correct answer for a feature that never repeats.
    """
    frame = dated_frame.copy()
    frame["unique_id"] = [f"u{i}" for i in range(len(frame))]

    encoder = OutOfFoldTargetEncoder(columns=("unique_id",))
    encoded = encoder.fit_transform(frame)

    assert encoded["unique_id_te"].nunique() == 1
    assert encoded["unique_id_te"].iloc[0] == pytest.approx(encoder.prior_)

    # The naive alternative, for contrast: fitted on all rows and applied to
    # those same rows, it tracks the label closely enough to be a giveaway.
    naive = frame["unique_id"].map(encoder.mappings_["unique_id"])
    assert np.corrcoef(naive, frame[LABEL])[0, 1] > 0.5


def test_out_of_fold_encoding_retains_signal_for_repeated_categories(
    dated_frame: pd.DataFrame,
) -> None:
    """Removing leakage must not remove genuine signal along with it."""
    frame = dated_frame.copy()
    high = frame.index[: len(frame) // 2]
    frame.loc[high, "planted"] = "high"
    frame.loc[frame.index[len(frame) // 2 :], "planted"] = "low"
    frame.loc[high, LABEL] = 1

    encoded = OutOfFoldTargetEncoder(columns=("planted",)).fit_transform(frame)

    assert encoded["planted_te"].nunique() > 1
    assert abs(np.corrcoef(encoded["planted_te"], frame[LABEL])[0, 1]) > 0.5


def test_transform_before_fit_raises(dated_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="not fitted"):
        OutOfFoldTargetEncoder().transform(dated_frame)


def test_fit_requires_label(dated_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        OutOfFoldTargetEncoder().fit(dated_frame.drop(columns=[LABEL]))


def test_encoded_output_is_numeric(dated_frame: pd.DataFrame) -> None:
    encoder = OutOfFoldTargetEncoder(columns=("banner_pos",))
    encoded = encoder.fit_transform(dated_frame)

    assert encoded["banner_pos_te"].dtype == np.float64
    assert encoded.notna().all().all()


def test_feature_columns_switch_on_strategy() -> None:
    native = feature_columns(use_target_encoding=False)
    encoded = feature_columns(use_target_encoding=True)

    assert "banner_pos" in native
    assert "banner_pos_te" in encoded
    assert set(TIME_FEATURES) <= set(native) & set(encoded)


def test_transform_ignores_timestamp_and_label(dated_frame: pd.DataFrame) -> None:
    """Encoding unseen data must not require the label to be present."""
    encoder = OutOfFoldTargetEncoder(columns=("banner_pos",)).fit(dated_frame)
    without_label = dated_frame.drop(columns=[LABEL, TIMESTAMP])

    encoded = encoder.transform(without_label)
    assert encoded.notna().all().all()
