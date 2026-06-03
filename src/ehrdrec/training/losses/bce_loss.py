import torch.nn as nn

from ehrdrec.training.losses.base import LossFunction

class BCELoss(LossFunction):
    def forward(self, predictions, targets, losses=None, **kwargs):
        bce_loss_fn = nn.BCEWithLogitsLoss()
        return bce_loss_fn(predictions, targets)