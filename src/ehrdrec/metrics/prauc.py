import torch
from ehrdrec.metrics.base import Metric
from ehrdrec.utils.constants import ReservedId

# TODO: Remove UNK when scoring
class PRAUC(Metric):
    def __init__(
        self, 
        name: str = "PRAUC",
        ignore_indices: list[int] | None = None,
    ):
        super().__init__(name)
        self.all_outputs: list[torch.Tensor] = []
        self.all_targets: list[torch.Tensor] = []
        self.ignore_indices = ignore_indices if ignore_indices is not None else [ReservedId.UNK, ReservedId.PAD]

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self.ignore_indices:
            keep_mask = torch.ones(outputs.shape[-1], dtype=torch.bool, device=outputs.device)
            keep_mask[self.ignore_indices] = False

            outputs = outputs[..., keep_mask]
            targets = targets[..., keep_mask]

        self.all_outputs.append(outputs.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> float:
        outputs = torch.cat(self.all_outputs, dim=0).flatten()
        targets = torch.cat(self.all_targets, dim=0).flatten()

        if targets.sum() == 0:
            return 0.0

        sorted_indices = torch.argsort(outputs, descending=True)
        targets = targets[sorted_indices]

        tp = torch.cumsum(targets, dim=0)
        fp = torch.cumsum(1 - targets, dim=0)

        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / targets.sum().clamp(min=1e-8)

        precision = torch.cat([torch.ones(1, dtype=precision.dtype), precision])
        recall = torch.cat([torch.zeros(1, dtype=recall.dtype), recall])

        return torch.trapezoid(precision, recall).item()

    def reset(self) -> None:
        self.all_outputs = []
        self.all_targets = []