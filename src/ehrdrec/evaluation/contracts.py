from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EvaluationOutput:
    scores: torch.Tensor
    targets: torch.Tensor