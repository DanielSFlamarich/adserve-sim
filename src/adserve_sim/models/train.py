# src/adserve_sim/models/train.py

"""Train and evaluate click-probability models.

Two encoding strategies are trained and compared:

*Native* hands the raw categoricals to CatBoost, which applies **ordered target
statistics**, target encoding computed only from rows appearing earlier in a
random permutation, so a row never contributes to its own encoding. It is the
same leakage fix as out-of-fold encoding, applied at every split rather than
once up front.

*Target-encoded* uses the explicit encoder in :mod:`adserve_sim.features.build`.
It exists to make the mechanism visible rather than delegated, and to check that
the library's version is doing what it claims.

Every run is logged to MLflow so the comparison is recorded rather than
remembered.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score

from adserve_sim.data.schema import CATEGORICAL_COLUMNS, LABEL
from adserve_sim.data.split import TemporalSplit
from adserve_sim.features.build import (
    OutOfFoldTargetEncoder,
    add_time_features,
    feature_columns,
)

logger = logging.getLogger(__name__)

#: MLflow experiment under which all training runs are recorded.
EXPERIMENT_NAME: str = "ctr-baseline"

#: CatBoost settings, fixed across encoding strategies so the comparison is fair.
CATBOOST_PARAMS: dict[str, object] = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "early_stopping_rounds": 50,
    "verbose": 100,
}


@dataclass(frozen=True)
class Metrics:
    """Evaluation metrics for one model on one partition.

    Attributes:
        auc: Ranking quality. Invariant to any monotone rescaling of the
            scores, which is precisely why it cannot detect miscalibration.
        log_loss: Proper scoring rule. Unlike AUC it *does* penalise wrong
            probabilities, so a gap between good AUC and poor log loss is the
            first sign of a calibration problem.
        base_rate: Observed click rate, for reference.
        mean_prediction: Mean predicted probability. A well-calibrated model
            matches ``base_rate`` closely; the gap is the crudest possible
            calibration check.
    """

    auc: float
    log_loss: float
    base_rate: float
    mean_prediction: float

    @property
    def calibration_gap(self) -> float:
        """Difference between mean prediction and observed rate."""
        return self.mean_prediction - self.base_rate


def evaluate(labels: pd.Series, predictions: np.ndarray) -> Metrics:
    """Compute evaluation metrics for a set of predictions.

    Args:
        labels: Observed binary outcomes.
        predictions: Predicted click probabilities.

    Returns:
        The computed metrics.
    """
    return Metrics(
        auc=float(roc_auc_score(labels, predictions)),
        log_loss=float(log_loss(labels, predictions)),
        base_rate=float(labels.mean()),
        mean_prediction=float(predictions.mean()),
    )


def baseline_metrics(split: TemporalSplit) -> Metrics:
    """Evaluate the trivial model that predicts the training click rate always.

    Any model that fails to beat this is not learning anything. It also gives
    the log-loss floor a real model has to improve on, which is a more honest
    reference point than comparing AUC to 0.5.

    Args:
        split: The partitioned data.

    Returns:
        Metrics for the constant predictor on the test partition.
    """
    constant = float(split.train[LABEL].mean())
    predictions = np.full(len(split.test), constant)

    return Metrics(
        auc=0.5,
        log_loss=float(log_loss(split.test[LABEL], predictions)),
        base_rate=float(split.test[LABEL].mean()),
        mean_prediction=constant,
    )


def _prepare(
    split: TemporalSplit, use_target_encoding: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build model matrices for each partition under one encoding strategy.

    The encoder is fitted on training data only. Validation and test are
    transformed with those fitted mappings, never refitted, refitting on later
    data would let information from the evaluation windows reach the model.

    Args:
        split: The partitioned data.
        use_target_encoding: Whether to apply explicit target encoding.

    Returns:
        A ``(train, validation, test, columns)`` tuple.
    """
    train = add_time_features(split.train)
    validation = add_time_features(split.validation)
    test = add_time_features(split.test)

    if use_target_encoding:
        encoder = OutOfFoldTargetEncoder()
        train = pd.concat([train, encoder.fit_transform(train)], axis=1)
        validation = pd.concat([validation, encoder.transform(validation)], axis=1)
        test = pd.concat([test, encoder.transform(test)], axis=1)

    return train, validation, test, feature_columns(use_target_encoding)


def train_model(
    split: TemporalSplit,
    use_target_encoding: bool = False,
    log_to_mlflow: bool = True,
) -> tuple[CatBoostClassifier, dict[str, Metrics]]:
    """Fit a CatBoost classifier and evaluate it on validation and test.

    Args:
        split: The partitioned data.
        use_target_encoding: If true, use the explicit out-of-fold encoder; if
            false, hand raw categoricals to CatBoost's native handling.
        log_to_mlflow: Whether to record the run. Disabled in tests.

    Returns:
        A ``(model, metrics)`` pair, where metrics is keyed by partition name.
    """
    train, validation, test, columns = _prepare(split, use_target_encoding)
    categorical = [] if use_target_encoding else list(CATEGORICAL_COLUMNS)

    train_pool = Pool(train[columns], train[LABEL], cat_features=categorical)
    validation_pool = Pool(validation[columns], validation[LABEL], cat_features=categorical)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)

    metrics = {
        "validation": evaluate(validation[LABEL], model.predict_proba(validation[columns])[:, 1]),
        "test": evaluate(test[LABEL], model.predict_proba(test[columns])[:, 1]),
    }

    strategy = "target_encoded" if use_target_encoding else "native_categorical"
    logger.info(
        "%s: test AUC %.4f, log loss %.4f", strategy, metrics["test"].auc, metrics["test"].log_loss
    )

    if log_to_mlflow:
        _record(strategy, split, metrics, model)

    return model, metrics


def _record(
    strategy: str,
    split: TemporalSplit,
    metrics: dict[str, Metrics],
    model: CatBoostClassifier,
) -> None:
    """Log one training run to MLflow.

    Args:
        strategy: Encoding strategy name, used as the run name.
        split: The partitioned data, for size and base-rate context.
        metrics: Evaluation metrics by partition.
        model: The fitted model, for the iteration count actually used.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=strategy):
        mlflow.log_params({**CATBOOST_PARAMS, "encoding": strategy})
        mlflow.log_params({f"rows_{name}": size for name, size in split.sizes.items()})
        mlflow.log_metrics({f"base_rate_{k}": v for k, v in split.base_rates.items()})
        mlflow.log_metric("best_iteration", model.get_best_iteration() or 0)

        for partition, values in metrics.items():
            mlflow.log_metrics(
                {f"{partition}_{key}": value for key, value in asdict(values).items()}
            )
            mlflow.log_metric(f"{partition}_calibration_gap", values.calibration_gap)
