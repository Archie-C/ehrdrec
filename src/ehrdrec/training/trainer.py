from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ehrdrec.contracts.models import LossOutput
from ehrdrec.data.torch import EHRBatch
from ehrdrec.evaluation import EvaluationOutput
from ehrdrec.models.base import TorchEHRDrecModel
from ehrdrec.evaluation.metrics import Metric


logger = logging.getLogger(__name__)


# =====================================================================
# Results
# =====================================================================


@dataclass(frozen=True)
class EpochResult:
    epoch: int

    train_loss: float
    train_components: dict[str, float]

    validation_loss: float | None = None
    validation_components: dict[str, float] | None = None

    metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class TrainingResult:
    epochs: list[EpochResult]
    total_steps: int
    best_epoch: int | None = None
    selected_validation_loss: float | None = None
    selected_validation_metrics: dict[str, float] | None = None
    training_time: float = 0.0
    validation_time: float = 0.0


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class TrainerConfig:
    """
    Generic training configuration.

    Model-specific hyperparameters and objectives do not belong here.
    """

    epochs: int
    device: str | torch.device | None = None

    gradient_clip_norm: float | None = None
    gradient_clip_value: float | None = None

    non_blocking_device_transfer: bool = True
    selection_metric: str | None = None
    selection_mode: str = "min"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be greater than zero."
            )

        if (
            self.gradient_clip_value is not None
            and self.gradient_clip_value <= 0
        ):
            raise ValueError(
                "gradient_clip_value must be greater than zero."
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_value is not None
        ):
            raise ValueError(
                "Only one of gradient_clip_norm and "
                "gradient_clip_value may be configured."
            )

        if self.selection_mode not in {"min", "max"}:
            raise ValueError(
                "selection_mode must be either 'min' or 'max'."
            )


# =====================================================================
# Trainer
# =====================================================================


class Trainer:
    """
    Generic PyTorch trainer for EHRDRec models.

    The trainer is deliberately unaware of:

        - the task
        - the dataset
        - model architecture
        - prediction semantics
        - metric implementation details
        - model-specific objectives

    Models receive an EHRBatch and are responsible for their forward
    computation and training objective.

    The trainer owns only the mechanics of optimization.
    """

    def __init__(
        self,
        config: TrainerConfig,
        metrics: list[Metric] | None = None,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(
            config.device
        )
        self.metrics = metrics or []

    # =================================================================
    # Public API
    # =================================================================

    def fit(
        self,
        model: TorchEHRDrecModel,
        train_loader: DataLoader,
        optimizer: Optimizer,
        validation_loader: DataLoader | None = None,
    ) -> TrainingResult:

        model.to(self.device)
        logger.info(
            "Training started: epochs=%d device=%s",
            self.config.epochs,
            self.device,
        )

        epoch_results: list[EpochResult] = []
        total_steps = 0
        training_time = 0.0
        validation_time = 0.0

        best_score: float | None = None
        best_state_dict: dict[str, Any] | None = None
        best_epoch: int | None = None
        selected_validation_loss: float | None = None
        selected_validation_metrics: dict[str, float] | None = None

        for epoch in range(
            1,
            self.config.epochs + 1,
        ):
            train_started = time.perf_counter()
            (
                train_loss,
                train_components,
                epoch_steps,
            ) = self._train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
            )
            training_time += time.perf_counter() - train_started
            total_steps += epoch_steps

            validation_loss = None
            validation_components = None
            validation_metrics = None

            if validation_loader is not None:
                validation_started = time.perf_counter()
                (
                    validation_loss,
                    validation_components,
                    evaluation_output,
                ) = self._validate_epoch(
                    model=model,
                    loader=validation_loader,
                )

                validation_metrics = {
                    name: value
                    for metric in self.metrics
                    for name, value in metric.compute(
                        evaluation_output
                    ).items()
                }
                validation_time += (
                    time.perf_counter() - validation_started
                )

            epoch_result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_components=train_components,
                validation_loss=validation_loss,
                validation_components=validation_components,
                metrics=validation_metrics,
            )
            epoch_results.append(epoch_result)

            logger.info(
                "Epoch %d/%d complete: train_loss=%.6f "
                "validation_loss=%s validation_metrics=%s",
                epoch,
                self.config.epochs,
                train_loss,
                (
                    f"{validation_loss:.6f}"
                    if validation_loss is not None
                    else "n/a"
                ),
                validation_metrics or {},
            )

            if validation_loss is not None:
                if self.config.selection_metric is None:
                    selection_score = validation_loss
                else:
                    if (
                        validation_metrics is None
                        or self.config.selection_metric
                        not in validation_metrics
                    ):
                        raise ValueError(
                            "Configured selection_metric "
                            f"{self.config.selection_metric!r} was not "
                            "produced during validation."
                        )
                    selection_score = validation_metrics[
                        self.config.selection_metric
                    ]

                improved = (
                    best_score is None
                    or (
                        self.config.selection_mode == "min"
                        and selection_score < best_score
                    )
                    or (
                        self.config.selection_mode == "max"
                        and selection_score > best_score
                    )
                )

                if improved:
                    best_score = selection_score
                    best_epoch = epoch
                    selected_validation_loss = validation_loss
                    selected_validation_metrics = dict(
                        validation_metrics or {}
                    )
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    logger.info(
                        "Selected checkpoint updated: epoch=%d "
                        "criterion=%s value=%.6f",
                        epoch,
                        self.config.selection_metric or "validation_loss",
                        selection_score,
                    )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info(
                "Restored selected validation checkpoint from epoch %d",
                best_epoch,
            )

        logger.info(
            "Training completed: steps=%d training_seconds=%.3f "
            "validation_seconds=%.3f",
            total_steps,
            training_time,
            validation_time,
        )

        return TrainingResult(
            epochs=epoch_results,
            total_steps=total_steps,
            best_epoch=best_epoch,
            selected_validation_loss=selected_validation_loss,
            selected_validation_metrics=selected_validation_metrics,
            training_time=training_time,
            validation_time=validation_time,
        )

    # =================================================================
    # Training
    # =================================================================

    def _train_epoch(
        self,
        model: TorchEHRDrecModel,
        loader: DataLoader,
        optimizer: Optimizer,
    ) -> tuple[
        float,
        dict[str, float],
        int,
    ]:
        model.train()

        total_loss = 0.0
        component_totals: dict[str, float] = defaultdict(float)

        total_examples = 0
        total_steps = 0

        for batch in loader:
            batch = self._prepare_batch(batch)

            optimizer.zero_grad(
                set_to_none=True,
            )

            model_output = model(batch)

            loss_output = model.loss(
                outputs=model_output.scores,
                targets=batch.targets,
            )

            self._validate_loss_output(
                loss_output,
            )

            loss_output.total.backward()

            self._clip_gradients(
                model,
            )

            optimizer.step()

            batch_size = self._batch_size(
                batch,
            )

            total_loss += (
                loss_output.total.detach().item()
                * batch_size
            )

            for name, value in loss_output.components.items():
                component_totals[name] += (
                    value.detach().item()
                    * batch_size
                )

            total_examples += batch_size
            total_steps += 1

        if total_steps == 0:
            raise RuntimeError(
                "Training DataLoader produced no batches."
            )

        return (
            total_loss / total_examples,
            {
                name: value / total_examples
                for name, value in component_totals.items()
            },
            total_steps,
        )

    # =================================================================
    # Validation
    # =================================================================

    @torch.inference_mode()
    def _validate_epoch(
        self,
        model: TorchEHRDrecModel,
        loader: DataLoader,
    ) -> tuple[
        float,
        dict[str, float],
        EvaluationOutput,
    ]:
        model.eval()

        total_loss = 0.0
        component_totals: dict[str, float] = defaultdict(float)

        scores: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        total_examples = 0

        for batch in loader:
            batch = self._prepare_batch(
                batch,
            )

            model_output = model(
                batch,
            )

            loss_output = model.loss(
                outputs=model_output.scores,
                targets=batch.targets,
            )

            self._validate_loss_output(
                loss_output,
            )

            batch_size = self._batch_size(
                batch,
            )

            total_loss += (
                loss_output.total.detach().item()
                * batch_size
            )

            for name, value in loss_output.components.items():
                component_totals[name] += (
                    value.detach().item()
                    * batch_size
                )

            scores.append(
                model_output.scores.detach().cpu()
            )
            targets.append(
                batch.targets.detach().cpu()
            )

            total_examples += batch_size

        if total_examples == 0:
            raise RuntimeError(
                "Validation DataLoader produced no batches."
            )

        evaluation_output = EvaluationOutput(
            scores=torch.cat(scores, dim=0),
            targets=torch.cat(targets, dim=0),
        )

        return (
            total_loss / total_examples,
            {
                name: value / total_examples
                for name, value in component_totals.items()
            },
            evaluation_output,
        )

    # =================================================================
    # Batch preparation
    # =================================================================

    def _prepare_batch(
        self,
        batch: Any,
    ) -> EHRBatch:
        """
        Validate and move an EHRBatch to the trainer device.

        Metadata deliberately remains on the CPU because it is not part
        of model computation.
        """

        if not isinstance(
            batch,
            EHRBatch,
        ):
            raise TypeError(
                "Trainer expected the DataLoader to produce "
                f"EHRBatch instances, received "
                f"{type(batch).__name__!r} instead."
            )

        return EHRBatch(
            inputs=self._move_to_device(
                batch.inputs
            ),
            targets=batch.targets.to(
                self.device,
                non_blocking=(
                    self.config
                    .non_blocking_device_transfer
                ),
            ),
            metadata=batch.metadata,
        )

    def _move_to_device(
        self,
        value: Any,
    ) -> Any:
        """
        Recursively move tensors contained in an adapted batch.

        This supports variable-length structures such as:

            List[Tensor[T_i, vocab_size]]

        as used by RETAIN.
        """

        if isinstance(
            value,
            torch.Tensor,
        ):
            return value.to(
                self.device,
                non_blocking=(
                    self.config
                    .non_blocking_device_transfer
                ),
            )

        if isinstance(
            value,
            dict,
        ):
            return {
                key: self._move_to_device(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                self._move_to_device(
                    item
                )
                for item
                in value
            ]

        if isinstance(
            value,
            tuple,
        ):
            return tuple(
                self._move_to_device(
                    item
                )
                for item
                in value
            )

        return value

    # =================================================================
    # Gradients
    # =================================================================

    def _clip_gradients(
        self,
        model: TorchEHRDrecModel,
    ) -> None:
        """
        Apply configured gradient clipping.
        """

        if (
            self.config.gradient_clip_norm
            is not None
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    self.config
                    .gradient_clip_norm
                ),
            )

        elif (
            self.config.gradient_clip_value
            is not None
        ):
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                clip_value=(
                    self.config
                    .gradient_clip_value
                ),
            )

    # =================================================================
    # Validation helpers
    # =================================================================

    def _validate_loss_output(
        self,
        loss_output: LossOutput,
    ) -> None:

        if not isinstance(
            loss_output,
            LossOutput,
        ):
            raise TypeError(
                "model.loss() must return LossOutput, "
                f"received {type(loss_output).__name__!r}."
            )

        self._validate_loss_tensor(
            loss_output.total,
            name="total",
        )

        for name, value in loss_output.components.items():
            self._validate_loss_tensor(
                value,
                name=name,
            )


    @staticmethod
    def _validate_loss_tensor(
        value: torch.Tensor,
        name: str,
    ) -> None:

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"Loss component {name!r} must be "
                "a torch.Tensor."
            )

        if value.numel() != 1:
            raise ValueError(
                f"Loss component {name!r} must be scalar, "
                f"received shape {tuple(value.shape)}."
            )

        if not torch.isfinite(value).item():
            raise FloatingPointError(
                f"Loss component {name!r} is non-finite."
            )

    @staticmethod
    def _batch_size(
        batch: EHRBatch,
    ) -> int:
        """
        Determine number of examples from the target tensor.
        """

        if batch.targets.ndim == 0:
            return 1

        return int(
            batch.targets.shape[0]
        )

    # =================================================================
    # Device
    # =================================================================

    @staticmethod
    def _resolve_device(
        requested: str | torch.device | None,
    ) -> torch.device:
        """
        Resolve the execution device.

        If no explicit device is requested, CUDA is preferred when
        available and CPU is used otherwise.
        """

        if requested is not None:
            return torch.device(
                requested
            )

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )