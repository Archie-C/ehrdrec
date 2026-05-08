from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.models.dataclasses import EvaluationResults


class BaseEvaluator(ABC):
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        metrics: list | None = None,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)
        self.test_loader = test_loader
        self.metrics = metrics or []
        self.model = model.to(self.device)

    @abstractmethod
    def run(self) -> EvaluationResults:
        pass