from __future__ import annotations

import torch

from ehrdrec.evaluation.contracts import EvaluationOutput
from ehrdrec.evaluation.metrics.base import Metric


class F1(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(name="f1", threshold=threshold)

    def compute(
        self,
        output: EvaluationOutput,
    ) -> dict[str, float]:
        predictions = (
            output.scores >= self.threshold
        )
        targets = output.targets.bool()

        true_positives = (
            predictions & targets
        ).sum(dim=1)

        predicted_positives = predictions.sum(dim=1)
        actual_positives = targets.sum(dim=1)

        denominator = (
            predicted_positives
            + actual_positives
        )

        f1 = torch.where(
            denominator > 0,
            (
                2.0
                * true_positives.float()
                / denominator.float()
            ),
            torch.ones_like(
                denominator,
                dtype=torch.float,
            ),
        )

        return {
            "f1": f1.mean().item(),
        }