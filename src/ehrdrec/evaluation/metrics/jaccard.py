from __future__ import annotations

import torch

from ehrdrec.evaluation.contracts import EvaluationOutput
from ehrdrec.evaluation.metrics.base import Metric


class Jaccard(Metric):
    def __init__(
        self,
        threshold: float = 0.5
    ) -> None:
        super().__init__(name="jaccard", threshold=threshold)

    def compute(
        self,
        output: EvaluationOutput,
    ) -> dict[str, float]:
        predictions = (
            output.scores >= self.threshold
        )
        targets = output.targets.bool()

        intersection = (
            predictions & targets
        ).sum(dim=1)

        union = (
            predictions | targets
        ).sum(dim=1)

        # Define two empty sets as having perfect overlap.
        jaccard = torch.where(
            union > 0,
            intersection.float() / union.float(),
            torch.ones_like(
                union,
                dtype=torch.float,
            ),
        )

        return {
            "jaccard": jaccard.mean().item(),
        }