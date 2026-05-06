import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ehrdrec.models.dataclasses import TrainingResults
from ehrdrec.training import BaseTrainer

class Trainer(BaseTrainer):
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
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            device=device,
            epochs=epochs,
        )
    
    # TODO: Add support for metrics, logging, learning rate scheduling, early stopping, etc.
    def fit(self) -> TrainingResults:
        best_val_loss = None
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        best_train_metrics: dict[str, float] = {}
        best_val_metrics: dict[str, float] = {}

        final_train_loss = None
        final_val_loss = None

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self._train_one_epoch()

            final_train_loss = train_loss

            if self.val_loader is not None:
                val_loss, val_metrics = self._validate()
                final_val_loss = val_loss

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics

            else:
                # If no validation set, keep latest model as "best"
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_train_metrics = train_metrics

        return TrainingResults(
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_model_state=best_model_state,
            best_train_metrics=best_train_metrics,
            best_val_metrics=best_val_metrics,
            best_epoch=best_epoch,
        )

    def _train_one_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        self._reset_metrics()

        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(features)
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            self._update_metrics(outputs, targets)

        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples.")

        avg_loss = total_loss / total_samples
        metrics = self._compute_metrics()

        return avg_loss, metrics

    def _validate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        self._reset_metrics()

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                self._update_metrics(outputs, targets)

        if total_samples == 0:
            raise ValueError("Validation dataloader produced no samples.")

        avg_loss = total_loss / total_samples
        metrics = self._compute_metrics()

        return avg_loss, metrics

    def _reset_metrics(self) -> None:
        if self.metrics:
            for metric in self.metrics:
                metric.reset()

    def _update_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self.metrics:
            for metric in self.metrics:
                metric.update(outputs.detach(), targets.detach())

    def _compute_metrics(self) -> dict[str, float]:
        if not self.metrics:
            return {}

        results = {}

        for metric in self.metrics:
            name = metric.__class__.__name__
            value = metric.compute()

            if isinstance(value, torch.Tensor):
                value = value.item()

            results[name] = value

        return results