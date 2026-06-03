import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ehrdrec.models.dataclasses import TrainingResults
from ehrdrec.training import BaseTrainer
from ehrdrec.training.logging import TrainerLogger

class Trainer(BaseTrainer):
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
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            target_metric=target_metric,
            higher_is_better=higher_is_better,
            device=device,
            epochs=epochs,
            logger=logger,
        )
    
    # TODO: Add support for metrics, logging, learning rate scheduling, early stopping, etc.
    def fit(self) -> TrainingResults:
        best_val_score = None
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        best_train_metrics: dict[str, float] = {}
        best_val_metrics: dict[str, float] = {}

        final_train_loss = None
        final_val_score = None

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self._train_one_epoch()

            final_train_loss = train_loss

            if self.val_loader is not None:
                val_metrics = self._validate()
                current = val_metrics.get(self.target_metric) if self.target_metric else None
                final_val_score = current

                if current is not None:
                    improved = (
                        best_val_score is None or
                        (self.higher_is_better and current > best_val_score) or
                        (not self.higher_is_better and current < best_val_score)
                    )
                    if improved:
                        best_val_score = current
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_train_metrics = train_metrics
                        best_val_metrics = val_metrics
                        if self.logger is not None:
                            self.logger.on_best_model(epoch, best_val_score, best_model_state)
                else:
                    # no target metric, just keep latest
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics
                    if self.logger is not None:
                        self.logger.on_best_model(epoch, None, best_model_state)
                        
            if self.logger is not None:
                self.logger.on_epoch_end(epoch, train_metrics, val_metrics)

        return TrainingResults(
            final_train_loss=final_train_loss,
            final_val_score=final_val_score,
            best_val_score=best_val_score,
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
            if isinstance(features, dict):
                features = {k: v.to(self.device) for k, v in features.items()}
            else:
                features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            output = self.model(features)

            if isinstance(output, dict):
                logits = output["predictions"]
                losses = output.get("losses", None)
            else:
                logits = output
                losses = None

            # The loss function must be able to handle input losses (we can't just use the standard torch ones)
            loss = self.loss_fn(logits, targets, losses=losses)

            loss.backward()
            self.optimizer.step()

            batch_size = next(iter(features.values())).size(0) if isinstance(features, dict) else features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            self._update_metrics(logits, targets)

        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples.")

        avg_loss = total_loss / total_samples
        metrics = self._compute_metrics()

        return avg_loss, metrics

    # In validation we only care about metrics, not losses
    def _validate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()

        self._reset_metrics()

        with torch.no_grad():
            for features, targets in self.val_loader:
                if isinstance(features, dict):
                    features = {k: v.to(self.device) for k, v in features.items()}
                else:
                    features = features.to(self.device)
                targets = targets.to(self.device)

                output = self.model(features)
                logits = output["predictions"] if isinstance(output, dict) else output

                self._update_metrics(logits, targets)

        metrics = self._compute_metrics()

        return metrics

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
            name = metric.name
            value = metric.compute()

            if isinstance(value, torch.Tensor):
                value = value.item()

            results[name] = value

        return results