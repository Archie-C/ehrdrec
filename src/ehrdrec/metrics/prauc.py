import torch
from ehrdrec.metrics.base import Metric

class PRAUC(Metric):
    def __init__(self, name: str = "PRAUC"):
        super().__init__(name)
        self.all_outputs: list[torch.Tensor] = []
        self.all_targets: list[torch.Tensor] = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.all_outputs.append(outputs.cpu())
        self.all_targets.append(targets.cpu())

    def compute(self) -> float:
        outputs = torch.cat(self.all_outputs, dim=0).flatten()
        targets = torch.cat(self.all_targets, dim=0).flatten()

        # Sort by descending score
        sorted_indices = torch.argsort(outputs, descending=True)
        targets = targets[sorted_indices]

        tp = torch.cumsum(targets, dim=0)
        fp = torch.cumsum(1 - targets, dim=0)

        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / targets.sum().clamp(min=1e-8)

        # Prepend (recall=0, precision=1) for AUC calculation
        precision = torch.cat([torch.tensor([1.0]), precision])
        recall = torch.cat([torch.tensor([0.0]), recall])

        # Trapezoidal integration
        auc = torch.trapezoid(precision, recall).item()
        return abs(auc)

    def reset(self) -> None:
        self.all_outputs = []
        self.all_targets = []