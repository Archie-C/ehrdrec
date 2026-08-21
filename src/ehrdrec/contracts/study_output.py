from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ehrdrec.contracts.experiment_output import (
    DatasetInformation,
    ExperimentSummary,
    TaskInformation,
)


class StudyStatus(Enum):
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class StudyExperiment:
    experiment_id: str
    model_name: str
    status: StudyStatus
    seeds: tuple[int, ...]
    artifact_path: str | None = None
    reproducibility_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class StudyManifest:
    schema_version: str
    study_id: str
    study_name: str | None
    task: TaskInformation
    dataset: DatasetInformation | None
    status: StudyStatus
    started_at: str
    finished_at: str
    experiments: tuple[StudyExperiment, ...]
    comparison_valid: bool = True
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudySummary:
    study_id: str
    status: StudyStatus
    num_experiments: int
    num_completed: int
    num_failed: int
    num_successful_runs: int
    total_wall_time: float
    experiments: dict[str, ExperimentSummary] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyResults:
    study_id: str
    status: StudyStatus
    experiments: dict[str, ExperimentSummary]
    manifest: StudyManifest
    summary: StudySummary
