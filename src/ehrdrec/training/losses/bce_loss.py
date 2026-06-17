import torch.nn as nn

from ehrdrec.training.losses.base import LossFunction

class BCELoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets, model_output=None, features=None, losses=None, **kwargs):
        return self.bce_loss_fn(predictions, targets)