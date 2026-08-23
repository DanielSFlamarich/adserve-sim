# tests/test_train.py

"""Tests for click model training, evaluation, and the baseline comparison."""

import numpy as np
import pandas as pd
import pytest

from adserve_sim.data.split import TemporalSplit, split_by_day
from adserve_sim.models.train import (
    Metrics,
    baseline_metrics,
    evaluate,
    train_model,
)


@pytest.fixture
def learnable_split(learnable_frame: pd.DataFrame) -> TemporalSplit:
    """A split of data with a planted signal, so a model has something to find."""
    return split_by_day(learnable_frame, val_days=1, test_days=1)


def test_evaluate_returns_all_metrics() -> None:
    labels = pd.Series([0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = evaluate(labels, predictions)

    assert metrics.auc == pytest.approx(1.0)
    assert metrics.log_loss > 0
    assert metrics.base_rate == pytest.approx(0.5)


def test_calibration_gap_is_signed() -> None:
    """A positive gap means the model over-predicts, which is the costly direction."""
    over = Metrics(auc=0.7, log_loss=0.4, base_rate=0.10, mean_prediction=0.15)
    under = Metrics(auc=0.7, log_loss=0.4, base_rate=0.10, mean_prediction=0.05)

    assert over.calibration_gap == pytest.approx(0.05)
    assert under.calibration_gap == pytest.approx(-0.05)


def test_auc_is_blind_to_rescaling() -> None:
    """The premise of the whole project, asserted rather than assumed.

    Halving every prediction leaves the ranking untouched, so AUC does not
    move, but log loss gets much worse, because the probabilities are now
    wrong. This is exactly the failure a team tracking only AUC cannot see.
    """
    labels = pd.Series([0, 1, 0, 1, 0, 1])
    calibrated = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])

    original = evaluate(labels, calibrated)
    rescaled = evaluate(labels, calibrated * 0.5)

    assert rescaled.auc == pytest.approx(original.auc)
    assert rescaled.log_loss > original.log_loss


def test_baseline_predicts_the_training_rate(learnable_split: TemporalSplit) -> None:
    baseline = baseline_metrics(learnable_split)

    assert baseline.auc == pytest.approx(0.5)
    assert baseline.mean_prediction == pytest.approx(
        learnable_split.train["click"].mean(),
        abs=1e-9,
    )


@pytest.mark.parametrize("use_target_encoding", [False, True])
def test_model_beats_the_constant_baseline(
    learnable_split: TemporalSplit, use_target_encoding: bool
) -> None:
    """Both encoding strategies must actually learn the planted signal."""
    baseline = baseline_metrics(learnable_split)
    _, metrics = train_model(
        learnable_split,
        use_target_encoding=use_target_encoding,
        log_to_mlflow=False,
    )

    assert metrics["test"].auc > 0.55
    assert metrics["test"].log_loss < baseline.log_loss


def test_model_reports_both_partitions(learnable_split: TemporalSplit) -> None:
    _, metrics = train_model(
        learnable_split,
        log_to_mlflow=False,
    )

    assert set(metrics) == {"validation", "test"}


def test_predictions_are_probabilities(learnable_split: TemporalSplit) -> None:
    _, metrics = train_model(
        learnable_split,
        log_to_mlflow=False,
    )

    assert 0.0 < metrics["test"].mean_prediction < 1.0
