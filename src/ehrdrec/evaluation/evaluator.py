from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from torch.utils.data import DataLoader

from ehrdrec.data.torch import EHRBatch
from ehrdrec.evaluation.contracts import EvaluationOutput
from ehrdrec.evaluation.metrics import Metric
from ehrdrec.models.base import TorchEHRDrecModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    scores: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    example_ids: tuple[str, ...] = ()


class Evaluator:
    def __init__(
        self,
        metrics: list[Metric],
        device: str | torch.device | None = None,
        non_blocking_device_transfer: bool = True,
    ) -> None:
        self.metrics = metrics
        self.device = self._resolve_device(device)
        self.non_blocking_device_transfer = (
            non_blocking_device_transfer
        )

    @torch.inference_mode()
    def evaluate(
        self,
        model: TorchEHRDrecModel,
        loader: DataLoader,
    ) -> EvaluationResult:
        model.to(self.device)
        model.eval()

        expected_examples = None
        try:
            expected_examples = len(loader.dataset)
        except (AttributeError, TypeError):
            pass

        logger.info(
            "Evaluation started: device=%s examples=%s",
            self.device,
            expected_examples if expected_examples is not None else "unknown",
        )

        scores: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        example_ids: list[str] = []

        total_examples = 0

        for batch in loader:
            batch = self._prepare_batch(batch)
            batch_size = self._batch_size(batch)

            model_output = model(batch)

            scores.append(
                model_output.scores.detach().cpu()
            )

            targets.append(
                batch.targets.detach().cpu()
            )
            example_ids.extend(
                self._example_ids(
                    batch=batch,
                    offset=total_examples,
                    batch_size=batch_size,
                )
            )

            total_examples += batch_size

        if total_examples == 0:
            raise RuntimeError(
                "Evaluation DataLoader produced no batches."
            )

        output = EvaluationOutput(
            scores=torch.cat(scores, dim=0),
            targets=torch.cat(targets, dim=0),
        )

        metrics = {
            name: value
            for metric in self.metrics
            for name, value in metric.compute(output).items()
        }

        logger.info(
            "Evaluation completed: examples=%d metrics=%s",
            total_examples,
            metrics,
        )

        return EvaluationResult(
            metrics=metrics,
            scores=output.scores,
            targets=output.targets,
            example_ids=tuple(example_ids),
        )

    @staticmethod
    def _example_ids(
        batch: EHRBatch,
        offset: int,
        batch_size: int,
    ) -> list[str]:
        values = batch.metadata.get("EXAMPLE_ID")
        if values is not None:
            identifiers = [str(value) for value in values]
            if len(identifiers) == batch_size:
                return identifiers

        subjects = batch.metadata.get("SUBJECT_ID")
        visits = batch.metadata.get("HADM_ID")
        if subjects is not None and visits is not None:
            identifiers = [
                f"{subject}:{visit}"
                for subject, visit in zip(subjects, visits, strict=True)
            ]
            if len(identifiers) == batch_size:
                return identifiers

        return [
            f"example_{index:08d}"
            for index in range(offset, offset + batch_size)
        ]

    def _prepare_batch(
        self,
        batch: EHRBatch,
    ) -> EHRBatch:
        if not isinstance(batch, EHRBatch):
            raise TypeError(
                "Evaluator expected the DataLoader to produce "
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
                    self.non_blocking_device_transfer
                ),
            ),
            metadata=batch.metadata,
        )

    def _move_to_device(
        self,
        value,
    ):
        if isinstance(value, torch.Tensor):
            return value.to(
                self.device,
                non_blocking=(
                    self.non_blocking_device_transfer
                ),
            )

        if isinstance(value, dict):
            return {
                key: self._move_to_device(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._move_to_device(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._move_to_device(item)
                for item in value
            )

        return value

    @staticmethod
    def _batch_size(
        batch: EHRBatch,
    ) -> int:
        if batch.targets.ndim == 0:
            return 1

        return int(batch.targets.shape[0])

    @staticmethod
    def _resolve_device(
        requested: str | torch.device | None,
    ) -> torch.device:
        if requested is not None:
            return torch.device(requested)

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")