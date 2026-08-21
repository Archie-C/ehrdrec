from __future__ import annotations

from sklearn.metrics import (
    label_ranking_average_precision_score,
)

from ehrdrec.evaluation.contracts import EvaluationOutput
from ehrdrec.evaluation.metrics.base import Metric


class LRAP(Metric):
    def __init__(
        self,
    ) -> None:
        super().__init__(name="lrap")
    
    def compute(
        self,
        output: EvaluationOutput,
    ) -> dict[str, float]:
        value = label_ranking_average_precision_score(
            output.targets.numpy(),
            output.scores.numpy(),
        )

        return {
            "lrap": float(value),
        }