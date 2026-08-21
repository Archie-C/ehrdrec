from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ehrdrec.tasks.base import Task
from ehrdrec.utils import Vocab

TaskLoss = Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor,
]


@dataclass(frozen=True)
class VocabSizes:
    """
    Vocabulary dimensions available to EHRDRec models.

    Fields are optional because not every task/model necessarily
    uses every vocabulary.
    """

    diagnoses: int | None = None
    procedures: int | None = None
    medications: int | None = None


@dataclass(frozen=True)
class ModelContext:
    """
    Information required to construct an EHRDRec model.

    This contains static experiment/task information only.

    It does NOT contain:
        - train/validation/test data
        - batches
        - model-specific encoded inputs

    Additional static resources such as DDI graphs or molecular
    graphs can be stored in `resources`.
    """

    vocab: VocabSizes
    
    task_loss: TaskLoss

    resources: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def from_task_output(
        cls,
        vocabs: dict[str, Vocab],
        task_loss: TaskLoss,
        resources: dict[str, Any] | None = None,
    ) -> "ModelContext":
        """
        Construct a ModelContext from task vocabularies.
        """

        return cls(
            vocab=VocabSizes(
                diagnoses=(
                    vocabs["diagnoses"].vocab_size
                    if "diagnoses" in vocabs
                    else None
                ),
                procedures=(
                    vocabs["procedures"].vocab_size
                    if "procedures" in vocabs
                    else None
                ),
                medications=(
                    vocabs["medications"].vocab_size
                    if "medications" in vocabs
                    else None
                ),
            ),
            task_loss=task_loss,
            resources=resources or {},
        )

    def get_resource(
        self,
        name: str,
    ) -> Any:
        """
        Get a required model resource.

        Raises a clear error if it is unavailable.
        """

        try:
            return self.resources[name]

        except KeyError as exc:
            raise ValueError(
                f"Required model resource "
                f"'{name}' is not available."
            ) from exc
            
@dataclass
class ModelOutput:
    scores: torch.Tensor
    
    # Anything required by model specific loss functions, for interpretability or diagnostics etc.
    auxiliary: dict[str, Any] = field(default_factory=dict)
    
@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]