from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ehrdrec.models.dataclasses import TrainingResults
from ehrdrec.training.logging import TrainerLogger


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        metrics: list | None = None,
        target_metric: str | None = None,
        higher_is_better: bool = True,
        device: str | torch.device = "cuda",
        epochs: int = 10,
        logger: TrainerLogger | None = None,
    ):
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or []
        self.target_metric = target_metric
        self.higher_is_better = higher_is_better
        self.epochs = epochs
        self.model = model.to(self.device)
        self.logger = logger

    @abstractmethod
    def fit(self) -> TrainingResults:
        pass