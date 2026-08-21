from __future__ import annotations

from typing import Literal

from sklearn.metrics import average_precision_score

from ehrdrec.evaluation.contracts import EvaluationOutput
from ehrdrec.evaluation.metrics.base import Metric


class PRAUC(Metric):
    def __init__(
        self,
        average: Literal[
            "samples",
            "micro",
            "macro",
            "weighted",
        ] = "samples",
    ) -> None:
        super().__init__(name="prauc")
        self.average = average

    def compute(
        self,
        output: EvaluationOutput,
    ) -> dict[str, float]:
        value = average_precision_score(
            output.targets.numpy(),
            output.scores.numpy(),
            average=self.average,
        )

        return {
            f"prauc_{self.average}": float(value),
        }