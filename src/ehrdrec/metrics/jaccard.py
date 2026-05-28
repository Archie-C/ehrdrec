import torch
from ehrdrec.metrics.base import Metric
from ehrdrec.utils.constants import ReservedId

# TODO: Remove UNK when scoring
class Jaccard(Metric):
    def __init__(
        self,
        name: str = "Jaccard",
        threshold: float = 0.5,
        ignore_indices: list[int] | None = None,
        from_logits: bool = True,
    ):
        super().__init__(name)
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_indices = ignore_indices if ignore_indices is not None else [ReservedId.UNK, ReservedId.PAD]
        self.intersection = 0
        self.union = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        x = outputs.sigmoid() if self.from_logits else outputs
        preds = (x >= self.threshold).float()

        if self.ignore_indices:
            keep_mask = torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
            keep_mask[self.ignore_indices] = False

            preds = preds[..., keep_mask]
            targets = targets[..., keep_mask]

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