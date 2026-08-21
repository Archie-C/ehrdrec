from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import logging
from pathlib import Path
import time
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ehrdrec.contracts.experiment_output import (
    Experiment,
    ExperimentResults,
    PredictionRow,
    ReproducibilityJson,
    RunConfiguration,
    RunResults,
    RunStatus,
    RunTimes,
    TaskInformation,
    TrainedModel,
)
from ehrdrec.contracts.models import ModelContext
from ehrdrec.data import MIMIC3Loader
from ehrdrec.data.torch import EHRBatchCollator, EHRDataset
from ehrdrec.evaluation import EvaluationResult, Evaluator
from ehrdrec.evaluation.metrics import Metric
from ehrdrec.experiments.artifacts import (
    sha256_file,
    stable_fingerprint,
    summarize_experiment,
    write_experiment_artifacts,
)
from ehrdrec.experiments.reproducibility import (
    capture_dataset_information,
    capture_hardware_information,
    capture_model_information,
    capture_software_environment,
    current_command,
    resolve_callable_config,
    resolved_task_settings,
    seed_worker,
    set_seed,
)
from ehrdrec.models.base import TorchEHRDrecModel
from ehrdrec.tasks import (
    MedicationSetRecommendationAdapter,
    MedicationSetRecommendationTask,
    Task,
)
from ehrdrec.training.trainer import Trainer, TrainerConfig, TrainingResult


logger = logging.getLogger(__name__)

ModelFactory = Callable[..., TorchEHRDrecModel]
OptimizerFactory = Callable[
    [Iterable[torch.nn.Parameter]],
    Optimizer,
]
AdapterFactory = Callable[..., Any]


class ExperimentRunner:
    """Lightweight orchestration around the existing EHRDRec pipeline."""

    def __init__(
        self,
        output_root: str | Path,
        trainer_config: TrainerConfig,
        metrics: list[Metric],
        batch_size: int = 32,
        num_workers: int = 0,
        prediction_threshold: float = 0.5,
        deterministic: bool = True,
        log_level: int = logging.INFO,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        self.output_root = Path(output_root)
        self.trainer_config = trainer_config
        self.metrics = list(metrics)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prediction_threshold = prediction_threshold
        self.deterministic = deterministic
        self.log_level = log_level
        self.last_output_dir: Path | None = None

    def run(
        self,
        model: ModelFactory,
        task: Task,
        seeds: Iterable[int],
        *,
        model_config: dict[str, Any] | None = None,
        optimizer_factory: OptimizerFactory | None = None,
        loader: Any | None = None,
        adapter_factory: AdapterFactory | None = None,
        dataset_path: str | Path | None = None,
        dataset_name: str = "MIMIC-III",
        dataset_version: str | None = "1.4",
        experiment_id: str | None = None,
        model_config_source: str | Path | None = None,
    ) -> Experiment:
        seeds = [int(seed) for seed in seeds]
        if not seeds:
            raise ValueError("At least one seed is required.")

        model_config = dict(model_config or {})
        resolved_model_config = resolve_callable_config(model, model_config)
        task_settings = resolved_task_settings(task)
        started = datetime.now(timezone.utc)
        experiment_id = experiment_id or self._new_experiment_id(
            started=started,
            model=model,
            model_config=resolved_model_config,
            task=task,
            task_settings=task_settings,
        )

        output_dir = self.output_root / experiment_id
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "logs").mkdir()
        (output_dir / "models").mkdir()
        (output_dir / "history").mkdir()
        self.last_output_dir = output_dir
        log_path = output_dir / "logs" / "experiment.log"

        with self._file_logging(log_path):
            logger.info(
                "Experiment started: experiment_id=%s model=%s task=%s runs=%d",
                experiment_id,
                self._name(model),
                type(task).__name__,
                len(seeds),
            )
            try:
                experiment = self._run_experiment(
                    experiment_id=experiment_id,
                    output_dir=output_dir,
                    log_path=log_path,
                    started=started,
                    model=model,
                    model_config=model_config,
                    resolved_model_config=resolved_model_config,
                    model_config_source=model_config_source,
                    task=task,
                    task_settings=task_settings,
                    seeds=seeds,
                    optimizer_factory=optimizer_factory,
                    loader=loader,
                    adapter_factory=adapter_factory,
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                )
            except BaseException:
                logger.exception(
                    "Experiment failed before artifact completion: %s",
                    experiment_id,
                )
                raise

        return experiment

    def _run_experiment(
        self,
        *,
        experiment_id: str,
        output_dir: Path,
        log_path: Path,
        started: datetime,
        model: ModelFactory,
        model_config: dict[str, Any],
        resolved_model_config: dict[str, Any],
        model_config_source: str | Path | None,
        task: Task,
        task_settings: dict[str, Any],
        seeds: list[int],
        optimizer_factory: OptimizerFactory | None,
        loader: Any | None,
        adapter_factory: AdapterFactory | None,
        dataset_path: str | Path | None,
        dataset_name: str,
        dataset_version: str | None,
    ) -> Experiment:
        input_requirements = model.get_inputs()
        model_requirements = model.get_requirements()

        logger.info("Creating data request")
        data_request = task.get_data_request(
            input_requirements=input_requirements,
            model_requirements=model_requirements,
        )

        data_loader = loader or MIMIC3Loader()
        if dataset_path is None:
            dataset_path = task.config.get("mimic3_path")
        if dataset_path is None:
            raise ValueError(
                "dataset_path is required when it is not present as "
                "task.config['mimic3_path']."
            )

        logger.info("Data loading started")
        raw_frames = data_loader.load(
            path=dataset_path,
            request=data_request,
        )
        logger.info("Data loading completed")

        logger.info("Task preprocessing started")
        preprocess_kwargs = {
            "raw_frames": raw_frames,
            "input_requirements": input_requirements,
        }
        if "model_requirements" in inspect.signature(
            task.preprocess
        ).parameters:
            preprocess_kwargs["model_requirements"] = model_requirements
        task_output = task.preprocess(**preprocess_kwargs)
        logger.info("Task preprocessing completed")

        adapter_type = adapter_factory or self._default_adapter(task)
        logger.info("Adapter started: %s", self._name(adapter_type))
        adapter = adapter_type(
            task_output=task_output,
            input_requirements=input_requirements,
        )
        adapted = adapter.adapt()
        logger.info("Adapter completed")

        datasets = {
            "train": EHRDataset(adapted.train),
            "validation": EHRDataset(adapted.validation),
            "test": EHRDataset(adapted.test),
        }
        logger.info(
            "Dataset split sizes: train=%d validation=%d test=%d",
            len(datasets["train"]),
            len(datasets["validation"]),
            len(datasets["test"]),
        )

        dataset_information = capture_dataset_information(
            datasets=datasets,
            name=dataset_name,
            version=dataset_version,
            sources=list(raw_frames),
        )
        logger.info(
            "Cohort statistics: patients=%s visits=%s examples=%s",
            dataset_information.num_patients,
            dataset_information.num_visits,
            dataset_information.num_examples,
        )

        collator = EHRBatchCollator(
            fields=adapted.fields,
            target=adapted.target,
        )
        context = ModelContext.from_task_output(
            vocabs=task_output.vocab,
            task_loss=task.loss,
            resources=getattr(task_output, "resources", None),
        )

        run_results: dict[str, RunResults] = {}
        histories: dict[str, TrainingResult] = {}
        predictions: list[PredictionRow] = []
        trained_models: dict[str, TrainedModel] = {}

        run_configurations = [
            RunConfiguration(
                run_id=f"run_{index:03d}",
                seed=seed,
            )
            for index, seed in enumerate(seeds, start=1)
        ]

        for run_configuration in run_configurations:
            interrupted = self._run_once(
                run_configuration=run_configuration,
                output_dir=output_dir,
                model=model,
                model_config=model_config,
                optimizer_factory=optimizer_factory,
                datasets=datasets,
                collator=collator,
                context=context,
                run_results=run_results,
                histories=histories,
                predictions=predictions,
                trained_models=trained_models,
            )
            if interrupted:
                break

        results = ExperimentResults(
            experiment_id=experiment_id,
            model_name=self._name(model),
            task=type(task).__name__,
            runs=run_results,
        )
        logger.info("Creating experiment summary")
        summary = summarize_experiment(results)
        finished = datetime.now(timezone.utc)

        reproducibility = ReproducibilityJson(
            schema_version="1.0",
            experiment_id=experiment_id,
            model=capture_model_information(
                model_factory=model,
                resolved_config=resolved_model_config,
                config_source=model_config_source,
            ),
            task=TaskInformation(
                name=type(task).__name__,
                version=str(getattr(task, "version", "unknown")),
                settings=task_settings,
                fingerprint=stable_fingerprint(task_settings),
            ),
            dataset=dataset_information,
            runs=run_configurations,
            software=capture_software_environment(),
            hardware=capture_hardware_information(),
            command=current_command(),
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
        )

        experiment = Experiment(
            results=results,
            results_summary=summary,
            predictions=predictions,
            logs=str(log_path),
            reproducibility=reproducibility,
            models=trained_models,
        )

        logger.info("Writing experiment artifacts to %s", output_dir)
        write_experiment_artifacts(
            experiment=experiment,
            output_dir=output_dir,
            histories=histories,
        )
        logger.info(
            "Experiment completed: experiment_id=%s completed_runs=%d "
            "total_runs=%d",
            experiment_id,
            summary.num_runs,
            len(run_results),
        )
        return experiment

    def _run_once(
        self,
        *,
        run_configuration: RunConfiguration,
        output_dir: Path,
        model: ModelFactory,
        model_config: dict[str, Any],
        optimizer_factory: OptimizerFactory | None,
        datasets: dict[str, EHRDataset],
        collator: EHRBatchCollator,
        context: ModelContext,
        run_results: dict[str, RunResults],
        histories: dict[str, TrainingResult],
        predictions: list[PredictionRow],
        trained_models: dict[str, TrainedModel],
    ) -> bool:
        run_id = run_configuration.run_id
        seed = run_configuration.seed
        run_started = time.perf_counter()
        training_result: TrainingResult | None = None
        testing_time: float | None = None

        logger.info("Run started: run_id=%s seed=%d", run_id, seed)

        try:
            set_seed(seed, deterministic=self.deterministic)
            loaders = self._build_loaders(
                datasets=datasets,
                collator=collator,
                seed=seed,
            )

            logger.info("Model initialization: run_id=%s", run_id)
            run_model = model(
                context=context,
                **model_config,
            )
            parameter_count = sum(
                parameter.numel()
                for parameter in run_model.parameters()
            )
            logger.info(
                "Model initialized: run_id=%s parameters=%d",
                run_id,
                parameter_count,
            )

            optimizer_builder = (
                optimizer_factory
                if optimizer_factory is not None
                else lambda parameters: torch.optim.Adadelta(parameters)
            )
            optimizer = optimizer_builder(run_model.parameters())
            logger.info(
                "Optimizer initialized: run_id=%s optimizer=%s",
                run_id,
                type(optimizer).__name__,
            )

            trainer = Trainer(
                config=self.trainer_config,
                metrics=self.metrics,
            )
            training_result = trainer.fit(
                model=run_model,
                train_loader=loaders["train"],
                validation_loader=loaders["validation"],
                optimizer=optimizer,
            )
            histories[run_id] = training_result

            model_path = output_dir / "models" / f"{run_id}.pt"
            torch.save(run_model.state_dict(), model_path)
            trained_models[run_id] = TrainedModel(
                filename=str(model_path.relative_to(output_dir)),
                format="pytorch_state_dict",
                sha256=sha256_file(model_path),
            )
            logger.info(
                "Model saved: run_id=%s path=%s",
                run_id,
                model_path,
            )

            evaluator = Evaluator(
                metrics=self.metrics,
                device=trainer.device,
                non_blocking_device_transfer=(
                    self.trainer_config.non_blocking_device_transfer
                ),
            )
            logger.info("Test evaluation started: run_id=%s", run_id)
            testing_started = time.perf_counter()
            evaluation = evaluator.evaluate(
                model=run_model,
                loader=loaders["test"],
            )
            testing_time = time.perf_counter() - testing_started
            logger.info(
                "Test evaluation completed: run_id=%s metrics=%s",
                run_id,
                evaluation.metrics,
            )

            run_predictions = self._prediction_rows(
                run_id=run_id,
                evaluation=evaluation,
            )
            predictions.extend(run_predictions)
            logger.info(
                "Prediction capture completed: run_id=%s rows=%d",
                run_id,
                len(run_predictions),
            )

            run_results[run_id] = RunResults(
                seed=seed,
                metrics={
                    "validation": dict(
                        training_result.selected_validation_metrics or {}
                    ),
                    "test": dict(evaluation.metrics),
                },
                run_time=RunTimes(
                    training=training_result.training_time,
                    validation=training_result.validation_time,
                    testing=testing_time,
                    total=time.perf_counter() - run_started,
                ),
                status=RunStatus.COMPLETED,
                selected_epoch=training_result.best_epoch,
            )
            logger.info(
                "Run completed: run_id=%s seed=%d total_seconds=%.3f",
                run_id,
                seed,
                run_results[run_id].run_time.total,
            )
            return False

        except KeyboardInterrupt:
            run_results[run_id] = self._failed_run_result(
                seed=seed,
                status=RunStatus.INTERRUPTED,
                error="KeyboardInterrupt",
                run_started=run_started,
                training_result=training_result,
                testing_time=testing_time,
            )
            logger.warning(
                "Run interrupted: run_id=%s seed=%d",
                run_id,
                seed,
                exc_info=True,
            )
            return True

        except Exception as exc:
            run_results[run_id] = self._failed_run_result(
                seed=seed,
                status=RunStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                run_started=run_started,
                training_result=training_result,
                testing_time=testing_time,
            )
            logger.exception(
                "Run failed: run_id=%s seed=%d",
                run_id,
                seed,
            )
            return False

    def _build_loaders(
        self,
        datasets: dict[str, EHRDataset],
        collator: EHRBatchCollator,
        seed: int,
    ) -> dict[str, DataLoader]:
        generator = torch.Generator()
        generator.manual_seed(seed)

        common = {
            "batch_size": self.batch_size,
            "collate_fn": collator,
            "num_workers": self.num_workers,
            "worker_init_fn": seed_worker if self.num_workers else None,
        }
        return {
            "train": DataLoader(
                datasets["train"],
                shuffle=True,
                generator=generator,
                **common,
            ),
            "validation": DataLoader(
                datasets["validation"],
                shuffle=False,
                **common,
            ),
            "test": DataLoader(
                datasets["test"],
                shuffle=False,
                **common,
            ),
        }

    def _prediction_rows(
        self,
        run_id: str,
        evaluation: EvaluationResult,
    ) -> list[PredictionRow]:
        if evaluation.scores is None or evaluation.targets is None:
            raise RuntimeError(
                "Evaluator did not return scores and targets required "
                "for prediction artifacts."
            )
        if len(evaluation.example_ids) != evaluation.scores.shape[0]:
            raise RuntimeError(
                "Evaluator example ID count does not match prediction count."
            )

        predictions = evaluation.scores >= self.prediction_threshold
        rows = []

        for example_id, target, prediction, score in zip(
            evaluation.example_ids,
            evaluation.targets,
            predictions,
            evaluation.scores,
            strict=True,
        ):
            rows.append(
                PredictionRow(
                    run_id=run_id,
                    example_id=example_id,
                    ground_truth=target.nonzero(as_tuple=True)[0].tolist(),
                    prediction=prediction.nonzero(as_tuple=True)[0].tolist(),
                    scores=score.tolist(),
                )
            )

        return rows

    @staticmethod
    def _failed_run_result(
        *,
        seed: int,
        status: RunStatus,
        error: str,
        run_started: float,
        training_result: TrainingResult | None,
        testing_time: float | None,
    ) -> RunResults:
        return RunResults(
            seed=seed,
            metrics={},
            run_time=RunTimes(
                training=(
                    training_result.training_time
                    if training_result is not None
                    else None
                ),
                validation=(
                    training_result.validation_time
                    if training_result is not None
                    else None
                ),
                testing=testing_time,
                total=time.perf_counter() - run_started,
            ),
            status=status,
            selected_epoch=(
                training_result.best_epoch
                if training_result is not None
                else None
            ),
            error=error,
        )

    @staticmethod
    def _default_adapter(task: Task) -> AdapterFactory:
        if isinstance(task, MedicationSetRecommendationTask):
            return MedicationSetRecommendationAdapter
        raise ValueError(
            "adapter_factory is required for tasks without a registered "
            "default adapter."
        )

    @staticmethod
    def _name(value: object) -> str:
        return str(getattr(value, "__name__", type(value).__name__))

    @staticmethod
    def _new_experiment_id(
        *,
        started: datetime,
        model: ModelFactory,
        model_config: dict[str, Any],
        task: Task,
        task_settings: dict[str, Any],
    ) -> str:
        fingerprint = stable_fingerprint(
            {
                "model": ExperimentRunner._name(model),
                "model_config": model_config,
                "task": type(task).__name__,
                "task_settings": task_settings,
            }
        )
        timestamp = started.strftime("%Y%m%dT%H%M%S%fZ")
        return f"exp_{timestamp}_{fingerprint[:8]}"

    @contextmanager
    def _file_logging(self, log_path: Path):
        package_logger = logging.getLogger("ehrdrec")
        previous_level = package_logger.level
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(self.log_level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )
        package_logger.addHandler(handler)
        if previous_level == logging.NOTSET or previous_level > self.log_level:
            package_logger.setLevel(self.log_level)

        try:
            yield
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(previous_level)
            handler.close()
