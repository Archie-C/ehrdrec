from typing import Any

import torch
from torch import nn

from ehrdrec.contracts.models import LossOutput, ModelContext, ModelOutput
from ehrdrec.requirements.model import InputRequirement, ModelRequirement

class TorchEHRDrecModel(nn.Module):
    _inputs: set[InputRequirement] = set()
    _requirements: set[ModelRequirement] = set()

    def __init__(self, context: ModelContext):
        super().__init__()
        self.context = context

    def forward(self, batch: Any) -> ModelOutput:
        raise NotImplementedError("Subclasses must implement this method.")

    def loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossOutput:
        prediction_loss = self.context.task_loss(
            outputs=outputs,
            targets=targets,
        )

        return LossOutput(
            total=prediction_loss,
            components={
                "prediction": prediction_loss,
            },
        )

    @classmethod
    def get_inputs(cls) -> set[InputRequirement]:
        return cls._inputs

    @classmethod
    def get_requirements(cls) -> set[ModelRequirement]:
        return cls._requirements