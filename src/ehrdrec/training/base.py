from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        metrics: list | None = None,
        device: str | torch.device = "cuda",
        epochs: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or []
        self.device = torch.device(device)
        self.epochs = epochs

    @abstractmethod
    def fit(self):
        pass