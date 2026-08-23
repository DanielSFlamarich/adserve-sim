# src/adserve_sim/eval/reliability.py

"""Measure whether predicted probabilities are usable as probabilities.

A model outputs 0.2. There are two readings: "this is more likely than the
0.1s", which is a claim about order, or "twenty in a hundred of these will be
clicked", which is a claim about frequency. Only the second makes it a
probability, and a model can be excellent at the first while badly wrong about
the second as AUC sees only order.

This module measures the second claim. The core operation is the same in every
function here: bucket predictions, compare each bucket's mean prediction to the
frequency actually observed in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from adserve_sim.eval.distortion import Distortion

#: Default number of buckets used for calibration measurement.
DEFAULT_N_BINS: int = 20


@dataclass(frozen=True)
class ReliabilityCurve:
    """Observed frequency against mean prediction, per bucket.

    Attributes:
        mean_predicted: Mean prediction within each non-empty bucket.
        observed_frequency: Observed positive rate within each bucket.
        weight: Share of all predictions falling in each bucket. Needed to read
            the curve honestly, a bucket far off the diagonal holding 0.1% of
            traffic is not the same problem as one holding 30%.
    """

    mean_predicted: np.ndarray
    observed_frequency: np.ndarray
    weight: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return the curve as a frame, one row per bucket."""
        return pd.DataFrame(
            {
                "mean_predicted": self.mean_predicted,
                "observed_frequency": self.observed_frequency,
                "weight": self.weight,
            }
        )


def _validate(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to arrays and check they can be compared.

    Args:
        labels: Binary outcomes.
        probabilities: Predicted probabilities.

    Returns:
        The two as float arrays.

    Raises:
        ValueError: If lengths differ, either is empty, or labels are not binary.
    """
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)

    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: labels {y.shape}, probabilities {p.shape}")
    if y.size == 0:
        raise ValueError("cannot measure calibration on an empty array")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")

    return y, p


def reliability_curve(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> ReliabilityCurve:
    """Bucket predictions and compare each bucket's prediction to its outcome.

    Buckets are equal-width over ``[0, 1]``. Empty buckets are dropped rather
    than reported as zero, which would put a spurious point at the origin.

    Args:
        labels: Binary outcomes.
        probabilities: Predicted probabilities.
        n_bins: Number of equal-width buckets.

    Returns:
        The curve, with per-bucket weights.

    Raises:
        ValueError: If inputs are mismatched, empty, or non-binary.
    """
    y, p = _validate(labels, probabilities)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.digitize(p, edges[1:-1])

    predicted, observed, weight = [], [], []
    for bucket in range(n_bins):
        mask = index == bucket
        if not mask.any():
            continue
        predicted.append(p[mask].mean())
        observed.append(y[mask].mean())
        weight.append(float(mask.mean()))

    return ReliabilityCurve(
        mean_predicted=np.array(predicted),
        observed_frequency=np.array(observed),
        weight=np.array(weight),
    )


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Mean absolute prediction/outcome gap, weighted by bucket occupancy.

    Weighting by occupancy is what stops a sparse, wildly wrong tail bucket from
    dominating a summary of a model that is accurate everywhere the traffic is.

    Args:
        labels: Binary outcomes.
        probabilities: Predicted probabilities.
        n_bins: Number of equal-width buckets.

    Returns:
        Expected calibration error. Zero is perfect; the scale is the same as
        the probabilities themselves.
    """
    curve = reliability_curve(labels, probabilities, n_bins)
    gaps = np.abs(curve.mean_predicted - curve.observed_frequency)
    return float((curve.weight * gaps).sum())


def calibration_gap(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean prediction minus observed rate.

    The crudest calibration check, and signed where :func:`expected_calibration_error`
    is not: positive means the model over-predicts on average, which is the
    direction that becomes an over-bid. A model can have a gap of zero and still
    be badly calibrated, if it over-predicts one region and under-predicts
    another by an offsetting amount.

    Args:
        labels: Binary outcomes.
        probabilities: Predicted probabilities.

    Returns:
        The signed difference.
    """
    y, p = _validate(labels, probabilities)
    return float(p.mean() - y.mean())


def compare_scenarios(
    labels: np.ndarray,
    probabilities: np.ndarray,
    scenarios: dict[str, Distortion],
    n_bins: int = DEFAULT_N_BINS,
) -> pd.DataFrame:
    """Score one set of predictions under several distortions.

    The AUC column is the point of the table. Every distortion is monotone, so
    ranking quality is identical across every row while log loss and calibration
    error move substantially. A team tracking AUC alone would see one model.

    Args:
        labels: Binary outcomes.
        probabilities: Undistorted predicted probabilities.
        scenarios: Named distortions to apply.
        n_bins: Buckets used for calibration error.

    Returns:
        One row per scenario, indexed by name.
    """
    y, p = _validate(labels, probabilities)

    rows = []
    for name, distortion in scenarios.items():
        distorted = distortion.apply(p)
        rows.append(
            {
                "scenario": name,
                "sharpness": distortion.sharpness,
                "shift": distortion.shift,
                "auc": float(roc_auc_score(y, distorted)),
                "log_loss": float(log_loss(y, distorted)),
                "ece": expected_calibration_error(y, distorted, n_bins),
                "calibration_gap": calibration_gap(y, distorted),
                "mean_prediction": float(distorted.mean()),
            }
        )

    return pd.DataFrame(rows).set_index("scenario")
