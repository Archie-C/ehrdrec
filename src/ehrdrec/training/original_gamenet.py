import copy
from typing import TYPE_CHECKING

import torch

from ehrdrec.models.dataclasses import TrainingResults
from ehrdrec.training import BaseTrainer
from ehrdrec.training.logging import TrainerLogger
from ehrdrec.utils.seeding import seed_everything

if TYPE_CHECKING:
    import optuna


class OriginalGAMENetTrainer(BaseTrainer):
    """
    Trainer for the upstream/original GAMENet model contract.

    The original model consumes one Python patient-history list at a time and
    returns (logits, ddi_loss) in train mode, so it cannot use the standard
    tensor/dict trainer directly.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        loss_fn=None,
        optimizer=None,
        metrics: list | None = None,
        target_metric: str | None = None,
        higher_is_better: bool = True,
        device: str | torch.device = "cuda",
        epochs: int = 10,
        logger: TrainerLogger | None = None,
        trial: "optuna.Trial | None" = None,
        seed: int | None = None,
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
        self.trial = trial
        self.seed = seed
        self._sync_original_model_device()

    def fit(self) -> TrainingResults:
        import optuna

        if self.seed is not None:
            seed_everything(self.seed)

        best_val_score = None
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        best_train_metrics: dict[str, float] = {}
        best_val_metrics: dict[str, float] = {}
        final_train_loss = None
        final_val_score = None
        history: list[dict] = []

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self._train_one_epoch()
            final_train_loss = train_loss
            val_metrics: dict[str, float] = {}

            if self.val_loader is not None:
                val_metrics = self._validate()
                current = val_metrics.get(self.target_metric) if self.target_metric else None
                final_val_score = current

                if current is not None:
                    improved = (
                        best_val_score is None
                        or (self.higher_is_better and current > best_val_score)
                        or (not self.higher_is_better and current < best_val_score)
                    )
                    if improved:
                        best_val_score = current
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_train_metrics = train_metrics
                        best_val_metrics = val_metrics
                        if self.logger is not None:
                            self.logger.on_best_model(epoch, best_val_score, best_model_state)

                    if self.trial is not None:
                        self.trial.report(current, epoch)
                        if self.trial.should_prune():
                            raise optuna.TrialPruned()
                else:
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics
                    if self.logger is not None:
                        self.logger.on_best_model(epoch, None, best_model_state)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

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
            seed=self.seed,
            history=history,
        )

    def _train_one_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.train()
        self._reset_metrics()

        total_loss = 0.0
        total_samples = 0

        for histories, targets in self.train_loader:
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            logits, ddi_loss = self._forward_histories(histories, expect_training_loss=True)
            losses = {"ddi_loss": ddi_loss} if ddi_loss is not None else None
            model_output = {"predictions": logits, "losses": losses}
            loss = self.loss_fn(
                logits,
                targets,
                model_output=model_output,
                features=histories,
                losses=losses,
            )

            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self._update_metrics(logits, targets)

        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples.")

        return total_loss / total_samples, self._compute_metrics()

    def _validate(self) -> dict[str, float]:
        self.model.eval()
        self._reset_metrics()

        with torch.no_grad():
            for histories, targets in self.val_loader:
                targets = targets.to(self.device)
                logits, _ = self._forward_histories(histories, expect_training_loss=False)
                self._update_metrics(logits, targets)

        return self._compute_metrics()

    def _forward_histories(
        self,
        histories: list,
        *,
        expect_training_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = []
        ddi_losses = []

        for history in histories:
            output = self.model(history)
            if isinstance(output, tuple):
                patient_logits, ddi_loss = output
                if ddi_loss is not None:
                    ddi_losses.append(ddi_loss)
            else:
                patient_logits = output
                ddi_loss = None

            logits.append(patient_logits)

        batch_logits = torch.cat(logits, dim=0)
        batch_ddi_loss = None

        if ddi_losses:
            batch_ddi_loss = torch.stack(
                [
                    loss if isinstance(loss, torch.Tensor) else torch.as_tensor(loss, device=self.device)
                    for loss in ddi_losses
                ]
            ).mean()
        elif expect_training_loss:
            batch_ddi_loss = torch.zeros((), device=batch_logits.device)

        return batch_logits, batch_ddi_loss

    def _sync_original_model_device(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "device"):
                module.device = self.device
            for attr_name in ("tensor_ddi_adj", "adj", "x"):
                value = getattr(module, attr_name, None)
                if isinstance(value, torch.Tensor):
                    setattr(module, attr_name, value.to(self.device))

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
            value = metric.compute()
            if isinstance(value, torch.Tensor):
                value = value.item()
            results[metric.name] = value

        return results
