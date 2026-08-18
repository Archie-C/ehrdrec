from enum import Enum
from typing import Any
from dataclasses import dataclass, field

class RunStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

@dataclass
class RunTimes:
    training: float | None = None
    validation: float | None = None
    testing: float | None = None
    total: float | None = None


@dataclass
class RunResults:
    seed: int
    metrics: dict[str, dict[str, float]]
    run_time: RunTimes
    status: RunStatus = RunStatus.COMPLETED

@dataclass
class ExperimentResults:
    """
    Represents the results of an experiment. Each experiment represents one model configuration evaluated on one task configuration. An experiment may contain multiple runs, typically differing only in random seed. 
    However, it is possible have multiple runs of the same model and task type, which will be stored in the runs dictionary. Each run has an ID.
    It is intentionally verbose to provide comprehensive information for analysis and debugging.
    
    Example dictionary representation:
    
    {
        "experiment_id": "exp_12345",
        "model_name": "GAMENet",
        "task": "mimiciii_medication_recommendation",
        "runs": {
            "run_1": {
            "seed": 42,
            "metrics": {
                "train": {
                "jaccard": 0.62
                },
                "validation": {
                "jaccard": 0.55
                },
                "test": {
                "jaccard": 0.53,
                "f1": 0.64,
                "prauc": 0.71,
                "ddi_rate": 0.06
                }
            },
            "run_time": {
                "training": 120.5,
                "validation": 30.2,
                "testing": 22.8,
                "total": 173.5
            }
            }
        }
    }
    
    """
    experiment_id: str
    model_name: str
    task: str
    runs: dict[str, RunResults]

@dataclass
class MetricSummary:
    mean: float
    std: float

@dataclass
class ExperimentSummary:
    """
    {
        "experiment_id": "exp_12345",
        "model_name": "GAMENet",
        "task": "mimiii_medication_recommendation",
        "num_runs": 5,
        "test_metrics": {
            "jaccard": {
            "mean": 0.531,
            "std": 0.006
            },
            "f1": {
            "mean": 0.641,
            "std": 0.004
            },
            "prauc": {
            "mean": 0.711,
            "std": 0.008
            }
        },
        "total_run_time": 874.2
    }
    """
    experiment_id: str
    model_name: str
    task: str
    num_runs: int
    test_metrics: dict[str, MetricSummary]
    total_run_time: float

# ============================================================
# PREDICTIONS
# ============================================================

@dataclass
class PredictionRow:
    run_id: str
    example_id: str
    # TODO: Revisit these as Any later. Not ideal.
    ground_truth: Any
    prediction: Any
    scores: Any | None = None

# ============================================================
# REPRODUCIBILITY
# ============================================================

@dataclass 
class SourceFile:
    """
    Snapshot of a source file used for the experiment.
    
    The contents are embedded directly into reproducibility.json so that the exact file can always be inspected.
    """
    filename: str
    sha256: str
    content: str

@dataclass
class ModelInformation:
    """
    Information describing the model implementation used in the experiment.
    """
    name: str
    source: SourceFile
    # The configuration actually used to initialise the model
    resolved_config: dict[str, Any]
    # The original config supplied by the user
    config_source: SourceFile | None = None

@dataclass
class PackageInformation:
    """
    Information describing a package
    """
    name: str
    version: str

@dataclass
class SoftwareEnvironment:
    """
    Information describing the software environment used in the experiment.
    """
    python_version: str
    ehrdrec_version: str
    
    packages: list[PackageInformation] = field(default_factory=list)
    
    operating_system: str | None = None

@dataclass
class HardwareInformation:
    """
    Hardware available during execution.
    """
    cpu: str | None = None
    cpu_count: int | None = None

    ram_gb: float | None = None

    gpu: str | None = None
    gpu_count: int | None = None

    cuda_version: str | None = None
    cudnn_version: str | None = None

@dataclass
class TaskInformation:
    """
    Complete description of the EHRDRec task used in the experiment.
    """
    name: str
    version: str

    # Fully resolved settings rather than only user-specified settings.
    settings: dict[str, Any]

    # Allows us to verify that two runs genuinely used the same task.
    fingerprint: str | None = None

@dataclass
class DataSplitInformation:
    """
    Describes one resulting dataset split.
    """
    name: str

    num_examples: int
    num_patients: int | None = None
    num_visits: int | None = None

    fingerprint: str | None = None

@dataclass
class DatasetInformation:
    """
    Describes the source dataset and exactly which data were used.
    """
    name: str
    version: str | None = None

    # E.g. MIMIC tables/files required by this task.
    sources: list[str] = field(default_factory=list)

    # Overall fingerprint of the source/processed data.
    fingerprint: str | None = None

    # Information about the final cohort.
    num_patients: int | None = None
    num_visits: int | None = None
    num_examples: int | None = None

    splits: list[DataSplitInformation] = field(default_factory=list)

@dataclass
class RunConfiguration:
    """
    Runtime-specific configuration for one repeated run.

    Settings that are identical across runs belong at the experiment level;
    only values that may differ between runs should appear here.
    """
    run_id: str
    seed: int

@dataclass
class ReproducibilityJson:
    schema_version: str
    experiment_id: str

    model: ModelInformation
    task: TaskInformation
    dataset: DatasetInformation

    runs: list[RunConfiguration]

    software: SoftwareEnvironment
    hardware: HardwareInformation

    command: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

@dataclass
class TrainedModel:
    """
    Represents the trained model artifact produced by an experiment.

    The exact serialization format is model/framework dependent.
    """
    filename: str
    format: str
    sha256: str

@dataclass
class Experiment:
    """
    Represents an experiment using the ehrdrec package.
    
    The outputs are required for each experiment and are used for reproducibility and logging purposes.
    There are 6 file outputs:
    """
    results: ExperimentResults
    results_summary: ExperimentSummary
    predictions: list[PredictionRow]
    logs: Any
    reproducibility: ReproducibilityJson
    models: dict[str, TrainedModel] # run_id -> TrainedModel