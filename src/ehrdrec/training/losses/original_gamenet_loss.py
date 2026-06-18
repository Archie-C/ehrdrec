import torch
import torch.nn as nn

from ehrdrec.training.losses.base import LossFunction


class OriginalGAMENetLoss(LossFunction):
    def __init__(self, ddi_weight: float = 0.05):
        super().__init__()
        self.ddi_weight = ddi_weight
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, model_output=None, features=None, losses=None, **kwargs):
        loss = self.bce_loss_fn(predictions, targets)

        if losses is not None and "ddi_loss" in losses:
            ddi_loss = losses["ddi_loss"]
            if not isinstance(ddi_loss, torch.Tensor):
                ddi_loss = torch.as_tensor(ddi_loss, device=predictions.device)
            loss = loss + self.ddi_weight * ddi_loss.to(predictions.device)

        return loss
