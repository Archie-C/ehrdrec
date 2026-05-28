import torch
from ehrdrec.metrics.base import Metric
from ehrdrec.utils.constants import ReservedId

# TODO: Remove UNK when scoring
class F1(Metric):
    def __init__(
        self,
        name: str = "F1",
        threshold: float = 0.5,
        ignore_indices: list[int] | None = None,
        from_logits: bool = True,
    ):
        super().__init__(name)
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_indices = ignore_indices if ignore_indices is not None else [ReservedId.UNK, ReservedId.PAD]
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        x = outputs.sigmoid() if self.from_logits else outputs
        preds = (x >= self.threshold).float()

        if self.ignore_indices:
            keep_mask = torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
            keep_mask[self.ignore_indices] = False

            preds = preds[..., keep_mask]
            targets = targets[..., keep_mask]

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