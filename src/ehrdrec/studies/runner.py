from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any

from ehrdrec.contracts.experiment_output import (
    DatasetInformation,
    Experiment,
    RunStatus,
    TaskInformation,
)
from ehrdrec.contracts.study_output import (
    StudyExperiment,
    StudyManifest,
    StudyResults,
    StudyStatus,
    StudySummary,
)
from ehrdrec.evaluation.metrics import Metric
from ehrdrec.experiments.artifacts import stable_fingerprint
from ehrdrec.experiments.reproducibility import resolved_task_settings
from ehrdrec.experiments.runner import (
    AdapterFactory,
    ExperimentRunner,
    ModelFactory,
    OptimizerFactory,
)
from ehrdrec.requirements import InputRequirement, ModelRequirement
from ehrdrec.studies.artifacts import write_study_artifacts
from ehrdrec.tasks import Task
from ehrdrec.training import TrainerConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentDefinition:
    """The model-specific arguments for one existing Experiment execution."""

    model: ModelFactory
    seeds: tuple[int, ...] | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    optimizer_factory: OptimizerFactory | None = None
    adapter_factory: AdapterFactory | None = None
    experiment_id: str | None = None
    model_config_source: str | Path | None = None
    runner: ExperimentRunner | None = None

    def __post_init__(self) -> None:
        if self.seeds is not None:
            seeds = tuple(int(seed) for seed in self.seeds)
            if not seeds:
                raise ValueError("ExperimentDefinition seeds cannot be empty.")
            object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "model_config", dict(self.model_config))


class StudyRunner:
    """Run comparable model Experiments over one shared prepared task."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        experiment_runner: ExperimentRunner | None = None,
        trainer_config: TrainerConfig | None = None,
        metrics: list[Metric] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        prediction_threshold: float = 0.5,
        deterministic: bool = True,
        log_level: int = logging.INFO,
    ) -> None:
        if experiment_runner is None:
            if trainer_config is None or metrics is None:
                raise ValueError(
                    "Provide experiment_runner, or both trainer_config and metrics."
                )
            experiment_runner = ExperimentRunner(
                output_root=output_root,
                trainer_config=trainer_config,
                metrics=metrics,
                batch_size=batch_size,
                num_workers=num_workers,
                prediction_threshold=prediction_threshold,
                deterministic=deterministic,
                log_level=log_level,
            )

        self.output_root = Path(output_root)
        self.experiment_runner = experiment_runner
        self.log_level = log_level
        self.last_output_dir: Path | None = None

    def run(
        self,
        *,
        task: Task,
        experiments: Iterable[ExperimentDefinition],
        name: str | None = None,
        seeds: Iterable[int] | None = None,
        loader: Any | None = None,
        dataset_path: str | Path | None = None,
        dataset_name: str = "MIMIC-III",
        dataset_version: str | None = "1.4",
        study_id: str | None = None,
    ) -> StudyResults:
        definitions = list(experiments)
        if not definitions:
            raise ValueError("A Study requires at least one ExperimentDefinition.")

        common_seeds = (
            tuple(int(seed) for seed in seeds) if seeds is not None else None
        )
        if common_seeds is not None and not common_seeds:
            raise ValueError("Study seeds cannot be empty.")

        resolved = [
            (
                definition,
                definition.seeds
                if definition.seeds is not None
                else common_seeds,
            )
            for definition in definitions
        ]
        missing_seeds = [
            self._name(definition.model)
            for definition, experiment_seeds in resolved
            if experiment_seeds is None
        ]
        if missing_seeds:
            raise ValueError(
                "Seeds must be supplied by the Study or every ExperimentDefinition; "
                f"missing for {missing_seeds}."
            )

        self._validate_explicit_ids(definitions)
        task_settings = resolved_task_settings(task)
        started = datetime.now(timezone.utc)
        study_id = study_id or self._new_study_id(
            started=started,
            name=name,
            task=task,
            task_settings=task_settings,
            resolved=resolved,
        )
        output_dir = self.output_root / study_id
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "logs").mkdir()
        experiments_dir = output_dir / "experiments"
        experiments_dir.mkdir()
        self.last_output_dir = output_dir
        log_path = output_dir / "logs" / "study.log"

        task_information = TaskInformation(
            name=type(task).__name__,
            version=str(getattr(task, "version", "unknown")),
            settings=task_settings,
            fingerprint=stable_fingerprint(task_settings),
        )
        records: list[StudyExperiment] = []
        completed_experiments: dict[str, Experiment] = {}
        comparison_errors: list[str] = []
        canonical_dataset: DatasetInformation | None = None
        interrupted = False
        started_clock = time.perf_counter()

        with self._file_logging(log_path):
            logger.info(
                "Study started: study_id=%s name=%s task=%s experiments=%d runs=%d",
                study_id,
                name,
                type(task).__name__,
                len(definitions),
                sum(len(experiment_seeds or ()) for _, experiment_seeds in resolved),
            )
            self._warn_duplicate_configurations(resolved)

            try:
                self._validate_metric_definitions(definitions)
                input_requirements, model_requirements = self._requirement_union(
                    definitions
                )
                logger.info(
                    "Combined input requirements: %s",
                    self._sorted_inputs(input_requirements),
                )
                logger.info(
                    "Combined model requirements: %s",
                    sorted(requirement.name for requirement in model_requirements),
                )
                logger.info("Shared data preparation started")
                prepared = self.experiment_runner.prepare_task_data(
                    task=task,
                    input_requirements=input_requirements,
                    model_requirements=model_requirements,
                    loader=loader,
                    dataset_path=dataset_path,
                )
                logger.info("Shared data preparation completed")
            except BaseException as exc:
                status = (
                    StudyStatus.INTERRUPTED
                    if isinstance(exc, KeyboardInterrupt)
                    else StudyStatus.FAILED
                )
                logger.exception("Shared Study preparation failed")
                self._finish(
                    output_dir=output_dir,
                    study_id=study_id,
                    name=name,
                    task_information=task_information,
                    dataset=None,
                    status=status,
                    num_experiments=len(definitions),
                    records=records,
                    completed_experiments=completed_experiments,
                    comparison_errors=[
                        f"{type(exc).__name__}: {exc}"
                    ],
                    started_at=started,
                    total_wall_time=time.perf_counter() - started_clock,
                )
                raise

            for index, (definition, experiment_seeds) in enumerate(
                resolved,
                start=1,
            ):
                assert experiment_seeds is not None
                experiment_runner = definition.runner or self.experiment_runner
                before_output = experiment_runner.last_output_dir
                logger.info(
                    "Experiment %d/%d started: model=%s seeds=%s",
                    index,
                    len(definitions),
                    self._name(definition.model),
                    list(experiment_seeds),
                )

                try:
                    experiment = experiment_runner.run(
                        model=definition.model,
                        task=task,
                        seeds=experiment_seeds,
                        model_config=definition.model_config,
                        optimizer_factory=definition.optimizer_factory,
                        adapter_factory=definition.adapter_factory,
                        dataset_name=dataset_name,
                        dataset_version=dataset_version,
                        experiment_id=definition.experiment_id,
                        model_config_source=definition.model_config_source,
                        prepared_task_data=prepared,
                        output_root=experiments_dir,
                    )
                except KeyboardInterrupt as exc:
                    interrupted = True
                    records.append(
                        self._failed_record(
                            definition=definition,
                            seeds=experiment_seeds,
                            runner=experiment_runner,
                            previous_output=before_output,
                            experiments_dir=experiments_dir,
                            index=index,
                            status=StudyStatus.INTERRUPTED,
                            exc=exc,
                        )
                    )
                    logger.warning(
                        "Experiment interrupted: model=%s",
                        self._name(definition.model),
                        exc_info=True,
                    )
                    break
                except Exception as exc:
                    records.append(
                        self._failed_record(
                            definition=definition,
                            seeds=experiment_seeds,
                            runner=experiment_runner,
                            previous_output=before_output,
                            experiments_dir=experiments_dir,
                            index=index,
                            status=StudyStatus.FAILED,
                            exc=exc,
                        )
                    )
                    logger.exception(
                        "Experiment failed: model=%s",
                        self._name(definition.model),
                    )
                    continue

                experiment_id = experiment.results.experiment_id
                experiment_status = self._experiment_status(experiment)
                artifact_path = Path("experiments") / experiment_id
                records.append(
                    StudyExperiment(
                        experiment_id=experiment_id,
                        model_name=experiment.results.model_name,
                        status=experiment_status,
                        seeds=tuple(experiment_seeds),
                        artifact_path=artifact_path.as_posix(),
                        reproducibility_path=(
                            artifact_path / "reproducibility.json"
                        ).as_posix(),
                    )
                )
                completed_experiments[experiment_id] = experiment

                identity_error = self._comparison_error(
                    task_information=task_information,
                    canonical_dataset=canonical_dataset,
                    experiment=experiment,
                )
                if identity_error is not None:
                    comparison_errors.append(identity_error)
                    logger.error(identity_error)
                elif canonical_dataset is None:
                    canonical_dataset = experiment.reproducibility.dataset
                    logger.info(
                        "Shared cohort statistics: patients=%s visits=%s "
                        "examples=%s",
                        canonical_dataset.num_patients,
                        canonical_dataset.num_visits,
                        canonical_dataset.num_examples,
                    )
                    logger.info(
                        "Shared split statistics: %s",
                        {
                            split.name: split.num_examples
                            for split in canonical_dataset.splits
                        },
                    )

                logger.info(
                    "Experiment %d/%d completed: experiment_id=%s model=%s "
                    "status=%s",
                    index,
                    len(definitions),
                    experiment_id,
                    experiment.results.model_name,
                    experiment_status.value,
                )

                if any(
                    run.status is RunStatus.INTERRUPTED
                    for run in experiment.results.runs.values()
                ):
                    interrupted = True
                    break

            status = self._study_status(
                records=records,
                experiments=completed_experiments,
                comparison_errors=comparison_errors,
                interrupted=interrupted,
            )
            result = self._finish(
                output_dir=output_dir,
                study_id=study_id,
                name=name,
                task_information=task_information,
                dataset=canonical_dataset,
                status=status,
                num_experiments=len(definitions),
                records=records,
                completed_experiments=completed_experiments,
                comparison_errors=comparison_errors,
                started_at=started,
                total_wall_time=time.perf_counter() - started_clock,
            )
            logger.info(
                "Study completed: study_id=%s status=%s experiments=%d",
                study_id,
                status.value,
                len(records),
            )
            return result

    def _finish(
        self,
        *,
        output_dir: Path,
        study_id: str,
        name: str | None,
        task_information: TaskInformation,
        dataset: DatasetInformation | None,
        status: StudyStatus,
        num_experiments: int,
        records: list[StudyExperiment],
        completed_experiments: dict[str, Experiment],
        comparison_errors: list[str],
        started_at: datetime,
        total_wall_time: float,
    ) -> StudyResults:
        finished = datetime.now(timezone.utc)
        summaries = {
            experiment_id: experiment.results_summary
            for experiment_id, experiment in completed_experiments.items()
        }
        summary = StudySummary(
            study_id=study_id,
            status=status,
            num_experiments=num_experiments,
            num_completed=sum(
                record.status is StudyStatus.COMPLETED for record in records
            ),
            num_failed=sum(
                record.status is StudyStatus.FAILED for record in records
            ),
            num_successful_runs=sum(
                experiment.results_summary.num_runs
                for experiment in completed_experiments.values()
            ),
            total_wall_time=total_wall_time,
            experiments=summaries,
        )
        manifest = StudyManifest(
            schema_version="1.0",
            study_id=study_id,
            study_name=name,
            task=task_information,
            dataset=dataset,
            status=status,
            started_at=started_at.isoformat(),
            finished_at=finished.isoformat(),
            experiments=tuple(records),
            comparison_valid=not comparison_errors,
            errors=tuple(comparison_errors),
        )
        logger.info("Writing Study artifacts to %s", output_dir)
        write_study_artifacts(
            output_dir=output_dir,
            manifest=manifest,
            summary=summary,
            experiments=completed_experiments,
        )
        return StudyResults(
            study_id=study_id,
            status=status,
            experiments=summaries,
            manifest=manifest,
            summary=summary,
        )

    @staticmethod
    def _requirement_union(
        definitions: list[ExperimentDefinition],
    ) -> tuple[set[InputRequirement], set[ModelRequirement]]:
        input_requirements: set[InputRequirement] = set()
        model_requirements: set[ModelRequirement] = set()
        for definition in definitions:
            input_requirements.update(definition.model.get_inputs())
            model_requirements.update(definition.model.get_requirements())
        return input_requirements, model_requirements

    @staticmethod
    def _experiment_status(experiment: Experiment) -> StudyStatus:
        statuses = {
            run.status for run in experiment.results.runs.values()
        }
        if statuses == {RunStatus.COMPLETED}:
            return StudyStatus.COMPLETED
        if RunStatus.INTERRUPTED in statuses:
            return StudyStatus.INTERRUPTED
        if RunStatus.COMPLETED in statuses:
            return StudyStatus.PARTIALLY_COMPLETED
        return StudyStatus.FAILED

    @staticmethod
    def _study_status(
        *,
        records: list[StudyExperiment],
        experiments: dict[str, Experiment],
        comparison_errors: list[str],
        interrupted: bool,
    ) -> StudyStatus:
        if interrupted:
            return StudyStatus.INTERRUPTED
        if comparison_errors:
            return StudyStatus.FAILED
        if records and all(
            record.status is StudyStatus.COMPLETED for record in records
        ):
            return StudyStatus.COMPLETED
        if any(
            experiment.results_summary.num_runs > 0
            for experiment in experiments.values()
        ):
            return StudyStatus.PARTIALLY_COMPLETED
        return StudyStatus.FAILED

    @staticmethod
    def _comparison_error(
        *,
        task_information: TaskInformation,
        canonical_dataset: DatasetInformation | None,
        experiment: Experiment,
    ) -> str | None:
        reproducibility = experiment.reproducibility
        if reproducibility.task.fingerprint != task_information.fingerprint:
            return (
                f"Experiment {experiment.results.experiment_id} has a different "
                "task fingerprint and is not comparable."
            )
        if canonical_dataset is None:
            return None

        canonical_splits = {
            split.name: split.fingerprint
            for split in canonical_dataset.splits
        }
        experiment_splits = {
            split.name: split.fingerprint
            for split in reproducibility.dataset.splits
        }
        if (
            reproducibility.dataset.fingerprint != canonical_dataset.fingerprint
            or experiment_splits != canonical_splits
        ):
            return (
                f"Experiment {experiment.results.experiment_id} has different "
                "cohort/split fingerprints and is not comparable."
            )
        return None

    def _failed_record(
        self,
        *,
        definition: ExperimentDefinition,
        seeds: tuple[int, ...],
        runner: ExperimentRunner,
        previous_output: Path | None,
        experiments_dir: Path,
        index: int,
        status: StudyStatus,
        exc: BaseException,
    ) -> StudyExperiment:
        output = runner.last_output_dir
        has_new_output = (
            output is not None
            and output != previous_output
            and output.parent == experiments_dir
        )
        experiment_id = (
            definition.experiment_id
            or (output.name if has_new_output and output is not None else None)
            or f"unstarted_{index:03d}"
        )
        artifact_path = (
            (Path("experiments") / experiment_id).as_posix()
            if has_new_output
            else None
        )
        return StudyExperiment(
            experiment_id=experiment_id,
            model_name=self._name(definition.model),
            status=status,
            seeds=tuple(seeds),
            artifact_path=artifact_path,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _validate_metric_definitions(
        self,
        definitions: list[ExperimentDefinition],
    ) -> None:
        signatures = {
            self._metric_signature(
                definition.runner or self.experiment_runner
            )
            for definition in definitions
        }
        if len(signatures) > 1:
            raise ValueError(
                "All Study experiments must use the same metric definitions."
            )

    @staticmethod
    def _metric_signature(runner: ExperimentRunner) -> tuple[str, ...]:
        return tuple(
            f"{type(metric).__module__}.{type(metric).__qualname__}:"
            f"{stable_fingerprint(vars(metric))}"
            for metric in runner.metrics
        )

    def _warn_duplicate_configurations(
        self,
        resolved: list[
            tuple[ExperimentDefinition, tuple[int, ...] | None]
        ],
    ) -> None:
        seen: dict[str, int] = {}
        for index, (definition, experiment_seeds) in enumerate(
            resolved,
            start=1,
        ):
            fingerprint = stable_fingerprint(
                {
                    "model": self._name(definition.model),
                    "model_config": definition.model_config,
                    "seeds": experiment_seeds,
                }
            )
            if fingerprint in seen:
                logger.warning(
                    "Exact duplicate Experiment configuration at positions %d "
                    "and %d; both will run.",
                    seen[fingerprint],
                    index,
                )
            else:
                seen[fingerprint] = index

    @staticmethod
    def _validate_explicit_ids(
        definitions: list[ExperimentDefinition],
    ) -> None:
        identifiers = [
            definition.experiment_id
            for definition in definitions
            if definition.experiment_id is not None
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Explicit Experiment IDs must be unique in a Study.")

    @staticmethod
    def _sorted_inputs(
        requirements: set[InputRequirement],
    ) -> list[str]:
        return [
            (
                f"{requirement.feature.name}/"
                f"{requirement.representation.name}/"
                f"{requirement.structure.name}"
            )
            for requirement in sorted(
                requirements,
                key=lambda item: (
                    item.feature.name,
                    item.representation.name,
                    item.structure.name,
                ),
            )
        ]

    @staticmethod
    def _name(value: object) -> str:
        return str(getattr(value, "__name__", type(value).__name__))

    @classmethod
    def _new_study_id(
        cls,
        *,
        started: datetime,
        name: str | None,
        task: Task,
        task_settings: dict[str, Any],
        resolved: list[
            tuple[ExperimentDefinition, tuple[int, ...] | None]
        ],
    ) -> str:
        fingerprint = stable_fingerprint(
            {
                "name": name,
                "task": type(task).__name__,
                "task_settings": task_settings,
                "experiments": [
                    {
                        "model": cls._name(definition.model),
                        "model_config": definition.model_config,
                        "seeds": experiment_seeds,
                    }
                    for definition, experiment_seeds in resolved
                ],
            }
        )
        timestamp = started.strftime("%Y%m%dT%H%M%S%fZ")
        return f"study_{timestamp}_{fingerprint[:8]}"

    @contextmanager
    def _file_logging(self, log_path: Path):
        study_logger = logger
        previous_level = study_logger.level
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(self.log_level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )
        study_logger.addHandler(handler)
        if previous_level == logging.NOTSET or previous_level > self.log_level:
            study_logger.setLevel(self.log_level)

        try:
            yield
        finally:
            study_logger.removeHandler(handler)
            study_logger.setLevel(previous_level)
            handler.close()
