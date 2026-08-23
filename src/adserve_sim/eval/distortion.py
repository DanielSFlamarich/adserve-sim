# src/adserve_sim/eval/distortion.py

"""Inject known miscalibration into well-calibrated predictions.

The baseline model in this project turned out to be well calibrated, CatBoost
fitted with ``Logloss``, a proper scoring rule, and stopped early before it could
grow overconfident. That leaves nothing to correct, and a calibration study with
no miscalibration in it measures nothing.

So the question is asked the other way round: distort a good model by a *known*
amount and measure what each amount costs downstream. That is a controlled
experiment rather than an anecdote about one unlucky model, and the resulting
cost curve does not depend on which model happened to be trained.

Distortion is applied in log-odds space rather than to probabilities directly.
Multiplying a probability pushes values outside ``[0, 1]`` and squashes the
distribution unevenly; the log-odds transform stays in range and has two
parameters with distinct, interpretable meanings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

#: Clip bound applied before the logit, since logit(0) and logit(1) are infinite.
EPSILON: float = 1e-6


@dataclass(frozen=True)
class Distortion:
    """An affine transform of log-odds: ``a * logit(p) + b``.

    Attributes:
        sharpness: Multiplies log-odds. Above 1 pushes predictions toward 0 and
            1 (overconfidence); below 1 pulls them toward the base rate
            (underconfidence). Must be positive; a negative value would invert
            the ranking, which is a different failure than miscalibration.
        shift: Added to log-odds. Positive raises every prediction, which is the
            systematic over-prediction that becomes a systematic over-bid.
    """

    sharpness: float = 1.0
    shift: float = 0.0

    def __post_init__(self) -> None:
        """Reject transforms that would reorder predictions."""
        if self.sharpness <= 0:
            raise ValueError(
                f"sharpness must be positive to preserve ranking, got {self.sharpness}"
            )

    @property
    def is_identity(self) -> bool:
        """Whether this transform leaves predictions unchanged."""
        return self.sharpness == 1.0 and self.shift == 0.0

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        """Distort probabilities, preserving their order.

        Strictly increasing for any positive ``sharpness``, so AUC and every
        other rank-based metric are unchanged. That invariance is the point:
        the distorted model is indistinguishable from the original by ranking
        metrics and differs only in the numbers it reports.

        Args:
            probabilities: Values in ``[0, 1]``.

        Returns:
            Distorted probabilities, same shape.

        Raises:
            ValueError: If any input lies outside ``[0, 1]``.
        """
        array = np.asarray(probabilities, dtype=np.float64)
        if array.size and (array.min() < 0.0 or array.max() > 1.0):
            raise ValueError("probabilities must lie in [0, 1]")

        clipped = np.clip(array, EPSILON, 1.0 - EPSILON)
        return np.asarray(expit(self.sharpness * logit(clipped) + self.shift))


#: A reference set spanning the two failure modes in both directions.
#:
#: The shift magnitude of 0.4 in log-odds is roughly a third relative change at a
#: 16% base rate - large enough to matter commercially, small enough that no
#: monitoring dashboard would flag it.
STANDARD_SCENARIOS: dict[str, Distortion] = {
    "none": Distortion(),
    "over-predict": Distortion(shift=0.4),
    "under-predict": Distortion(shift=-0.4),
    "overconfident": Distortion(sharpness=1.5),
    "underconfident": Distortion(sharpness=0.6),
}
