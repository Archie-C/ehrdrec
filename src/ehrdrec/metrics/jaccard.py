import torch
from ehrdrec.metrics.base import Metric


class Jaccard(Metric):
    def __init__(self, name: str = "Jaccard", threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold
        self.intersection = 0
        self.union = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (outputs >= self.threshold).float()

        intersection = (preds * targets).sum(dim=-1)
        union = ((preds + targets) >= 1).float().sum(dim=-1)

        self.intersection += intersection.sum().item()
        self.union += union.sum().item()

    def compute(self) -> float:
        if self.union == 0:
            return 0.0
        return self.intersection / self.union

    def reset(self) -> None:
        self.intersection = 0
        self.union = 0