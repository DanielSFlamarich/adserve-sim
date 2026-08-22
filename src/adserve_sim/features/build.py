# src/adserve_sim/features/build.py

"""Feature construction for click prediction.

Two things happen here.

- Time features:
The raw timestamp is not usable as a model input as a tree
would split on "before 21 October" and learn nothing that generalises to
November. What does generalise is the *cyclical* part: hour of day and day of
week. Both are available at serving time, which is the test any feature has to
pass.

Categorical encoding:
 Avazu's categoricals are enormous: ``device_ip`` and
``device_id`` run to millions of distinct values, so they cannot be one-hot
encoded. :class:`OutOfFoldTargetEncoder` implements target encoding with the
two corrections that make it safe; see its docstring for what those are and why
they are needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from adserve_sim.data.schema import CATEGORICAL_COLUMNS, LABEL, TIMESTAMP, SchemaError

logger = logging.getLogger(__name__)

#: cyclical time features derived from the raw timestamp.
TIME_FEATURES: tuple[str, ...] = ("hour_of_day", "day_of_week")

#: default smoothing weight, in units of "pseudo-observations of the prior".
#:
#: a category needs roughly this many impressions before its own click rate
#: outweighs the global rate. 20 is a deliberate middle ground: high enough that
#: a category seen twice does not get an extreme encoding, low enough that a
#: well-observed category is not dragged to the mean.
DEFAULT_SMOOTHING: float = 20.0

#: number of folds used to compute out-of-fold encodings on the training set.
DEFAULT_N_FOLDS: int = 5


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and day-of-week columns derived from the timestamp.

    Both are cyclical and therefore transferable to unseen dates, unlike the
    raw timestamp. Both are also known at request time, so nothing here could
    leak information the server would not have.

    Args:
        frame: A frame carrying a parsed timestamp column.

    Returns:
        A copy of ``frame`` with the time features appended.

    Raises:
        SchemaError: If the timestamp column is absent.
    """
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"frame has no {TIMESTAMP!r} column; call parse_hour first")

    out = frame.copy()
    out["hour_of_day"] = out[TIMESTAMP].dt.hour.astype("int8")
    out["day_of_week"] = out[TIMESTAMP].dt.dayofweek.astype("int8")
    return out


@dataclass
class OutOfFoldTargetEncoder:
    """Target encoding with smoothing and out-of-fold fitting.

    **What target encoding is.** Replace a category with the average click rate
    observed for that category. ``site_id=abc123`` becomes 0.031 if 3.1% of its
    impressions were clicked. This turns a million-valued categorical into one
    informative number, which trees handle well.

    **Why the naive version is broken.** If a category's encoding is computed
    from the same rows the model trains on, each row's own label is baked into
    its own feature. For a category appearing once - and with ``device_ip``,
    most do, the encoding *is* that row's label. The model appears excellent
    in training and is worthless in production. This is the classic target
    leakage failure.

    **The two corrections.**

    *Out-of-fold fitting* splits the training data into folds and encodes each
    fold using only the other folds. A row's own label never contributes to its
    own feature.

    *Smoothing* pulls rare categories toward the global click rate::

        encoding = (clicks + prior x weight) / (impressions + weight)

    A category seen twice with one click is not really a 50% click-rate
    category; it is an unknown category with weak evidence. The smoothing
    weight controls how much evidence is needed before a category's own rate
    dominates the global one.

    At transform time on unseen data, encodings come from the *full* training
    set, no folding needed, since those rows were never trained on. Categories
    absent from training fall back to the prior.

    Attributes:
        columns: Categorical columns to encode.
        smoothing: Prior weight in pseudo-observations.
        n_folds: Number of out-of-fold partitions.
        random_state: Seed for fold assignment.
    """

    columns: tuple[str, ...] = CATEGORICAL_COLUMNS
    smoothing: float = DEFAULT_SMOOTHING
    n_folds: int = DEFAULT_N_FOLDS
    random_state: int = 42

    prior_: float = field(default=float("nan"), init=False)
    mappings_: dict[str, pd.Series] = field(default_factory=dict, init=False)

    def _encode_column(self, values: pd.Series, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Compute a smoothed encoding map and its aggregate statistics.

        Args:
            values: Category values.
            labels: Corresponding binary labels.

        Returns:
            A ``(mapping, counts)`` pair, where mapping is indexed by category.
        """
        grouped = labels.groupby(values, observed=True)
        totals = grouped.sum()
        counts = grouped.count()

        smoothed = (totals + self.prior_ * self.smoothing) / (counts + self.smoothing)
        return smoothed, counts

    def fit(self, frame: pd.DataFrame) -> Self:
        """Learn encodings from the full training set, for use on later data.

        Args:
            frame: Training frame carrying the label column.

        Returns:
            The fitted encoder.

        Raises:
            SchemaError: If the label or any encoded column is absent.
        """
        missing = ({LABEL, *self.columns}) - set(frame.columns)
        if missing:
            raise SchemaError(f"frame is missing columns needed to fit: {sorted(missing)}")

        labels = frame[LABEL]
        self.prior_ = float(labels.mean())
        self.mappings_ = {
            column: self._encode_column(frame[column], labels)[0] for column in self.columns
        }

        logger.info("fitted encoder on %d rows, prior click rate %.4f", len(frame), self.prior_)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply learned encodings to unseen data.

        Args:
            frame: Frame to encode. The label is not required or used.

        Returns:
            A frame of encoded columns, suffixed ``_te``, aligned to the input.

        Raises:
            SchemaError: If :meth:`fit` has not been called.
        """
        if not self.mappings_:
            raise SchemaError("encoder is not fitted; call fit or fit_transform first")

        encoded = {
            f"{column}_te": frame[column].map(self.mappings_[column]).fillna(self.prior_)
            for column in self.columns
        }
        return pd.DataFrame(encoded, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``frame`` and return leakage-free encodings for those same rows.

        This is the method to use on the training set. Each fold is encoded
        using only the other folds, so no row contributes to its own feature.
        The full-data mappings are also stored, ready for :meth:`transform` on
        validation and test.

        Args:
            frame: Training frame carrying the label column.

        Returns:
            A frame of out-of-fold encoded columns, suffixed ``_te``.

        Raises:
            SchemaError: If the label or any encoded column is absent.
        """
        self.fit(frame)

        labels = frame[LABEL]
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Filled fold by fold into plain arrays, then assembled once. Writing
        # into a frame in place invites pandas chained-assignment ambiguity.
        buffers = {column: np.empty(len(frame), dtype=np.float64) for column in self.columns}

        for train_index, held_index in folds.split(frame):
            for column in self.columns:
                mapping, _ = self._encode_column(
                    frame[column].iloc[train_index], labels.iloc[train_index]
                )
                held_values = frame[column].iloc[held_index]
                buffers[column][held_index] = (
                    held_values.map(mapping).fillna(self.prior_).to_numpy()
                )

        return pd.DataFrame(
            {f"{column}_te": values for column, values in buffers.items()},
            index=frame.index,
        )


def feature_columns(use_target_encoding: bool) -> list[str]:
    """Return the model's input columns for a given encoding strategy.

    Args:
        use_target_encoding: If true, use encoded columns; otherwise pass the
            raw categoricals through for the model to handle natively.

    Returns:
        Ordered list of column names.
    """
    if use_target_encoding:
        return [*TIME_FEATURES, *(f"{column}_te" for column in CATEGORICAL_COLUMNS)]
    return [*TIME_FEATURES, *CATEGORICAL_COLUMNS]
