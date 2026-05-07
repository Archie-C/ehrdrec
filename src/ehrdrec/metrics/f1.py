import torch
from ehrdrec.metrics.base import Metric

class F1(Metric):
    def __init__(self, name: str = "F1", threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (outputs >= self.threshold).float()

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()

    def compute(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return 0.0
        return (2 * self.tp) / denominator

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0