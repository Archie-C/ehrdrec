from abc import ABC, abstractmethod
import torch.nn as nn

class LossFunction(ABC, nn.Module):
    def __init__(self):
        super().__init__()  # initialises nn.Module

    @abstractmethod
    def forward(self, predictions, targets, model_output=None, features=None, losses=None):
        raise NotImplementedError("Subclasses must implement the forward method.")