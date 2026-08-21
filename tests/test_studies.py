from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from ehrdrec.contracts.adapters import AdapterOutput, BatchTargetSpec
from ehrdrec.contracts.models import ModelOutput
from ehrdrec.contracts.study_output import StudyStatus
from ehrdrec.evaluation.metrics import Jaccard
from ehrdrec.experiments import ExperimentRunner
from ehrdrec.models.base import TorchEHRDrecModel
from ehrdrec.requirements import (
    DataRequirement,
    Feature,
    InputRequirement,
    InputStructure,
    ModelRequirement,
    Representation,
)
from ehrdrec.studies import ExperimentDefinition, StudyRunner
from ehrdrec.tasks.base import Task
from ehrdrec.training import TrainerConfig


DIAGNOSES = InputRequirement(
    feature=Feature.DIAGNOSES,
    representation=Representation.MULTI_HOT,
    structure=InputStructure.VISIT_SEQUENCE,
)
PROCEDURES = InputRequirement(
    feature=Feature.PROCEDURES,
    representation=Representation.MULTI_HOT,
    structure=InputStructure.VISIT_SEQUENCE,
)


class CountingTask(Task):
    def __init__(self) -> None:
        super().__init__()
        self.preprocess_calls = 0
        self.seen_inputs = set()
        self.seen_model_requirements = set()

    def preprocess(
        self,
        raw_frames,
        input_requirements,
        model_requirements=None,
    ):
        self.preprocess_calls += 1
        self.seen_inputs = set(input_requirements)
        self.seen_model_requirements = set(model_requirements or ())

        def frame(offset: int) -> pl.LazyFrame:
            return pl.DataFrame(
                {
                    "SUBJECT_ID": [offset + 1, offset + 2],
                    "HADM_ID": [offset + 11, offset + 12],
                    "targets": [[0], [1]],
                }
            ).lazy()

        return SimpleNamespace(
            train=frame(0),
            validation=frame(10),
            test=frame(20),
            vocab={"medications": SimpleNamespace(vocab_size=2)},
        )

    def loss(self, outputs, targets):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            outputs,
            targets,
        )


class CountingLoader:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.request = None
        self.error = error

    def load(self, path, request):
        self.calls += 1
        self.request = request
        if self.error is not None:
            raise self.error
        return {}


class RecordingAdapter:
    requirements: list[set[InputRequirement]] = []

    def __init__(self, task_output, input_requirements):
        self.task_output = task_output
        self.input_requirements = set(input_requirements)

    def adapt(self):
        type(self).requirements.append(self.input_requirements)
        return AdapterOutput(
            train=self.task_output.train,
            validation=self.task_output.validation,
            test=self.task_output.test,
            fields={},
            target=BatchTargetSpec(
                name="targets",
                representation=Representation.MULTI_HOT,
                vocab_size=2,
            ),
        )


class FailingAdapter(RecordingAdapter):
    def adapt(self):
        raise RuntimeError("intentional adaptation failure")


class ModelA(TorchEHRDrecModel):
    _inputs = {DIAGNOSES}

    def __init__(self, context, initial_value: float = 0.0):
        super().__init__(context)
        self.logits = torch.nn.Parameter(torch.full((2,), initial_value))

    def forward(self, batch):
        return ModelOutput(
            scores=self.logits.unsqueeze(0).expand(
                batch.targets.shape[0],
                -1,
            )
        )


class ModelB(ModelA):
    _inputs = {PROCEDURES}
    _requirements = {ModelRequirement.EHR_MEDICATION_GRAPH}


class CountingExperimentRunner(ExperimentRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prepare_calls = 0
        self.run_calls = 0

    def prepare_task_data(self, **kwargs):
        self.prepare_calls += 1
        return super().prepare_task_data(**kwargs)

    def run(self, *args, **kwargs):
        self.run_calls += 1
        return super().run(*args, **kwargs)


def make_runner(tmp_path: Path) -> CountingExperimentRunner:
    return CountingExperimentRunner(
        output_root=tmp_path,
        trainer_config=TrainerConfig(epochs=1, device="cpu"),
        metrics=[Jaccard()],
        batch_size=2,
    )


def definitions(
    *,
    first_adapter=RecordingAdapter,
) -> list[ExperimentDefinition]:
    return [
        ExperimentDefinition(
            model=ModelA,
            adapter_factory=first_adapter,
            experiment_id="exp_a",
            optimizer_factory=lambda parameters: torch.optim.SGD(
                parameters,
                lr=0.1,
            ),
        ),
        ExperimentDefinition(
            model=ModelB,
            adapter_factory=RecordingAdapter,
            experiment_id="exp_b",
            optimizer_factory=lambda parameters: torch.optim.SGD(
                parameters,
                lr=0.1,
            ),
        ),
    ]


def test_study_prepares_once_unions_requirements_and_reuses_experiments(
    tmp_path: Path,
) -> None:
    RecordingAdapter.requirements = []
    task = CountingTask()
    loader = CountingLoader()
    experiment_runner = make_runner(tmp_path)
    runner = StudyRunner(
        output_root=tmp_path,
        experiment_runner=experiment_runner,
    )

    result = runner.run(
        task=task,
        experiments=definitions(),
        seeds=[7, 11],
        loader=loader,
        dataset_path=tmp_path,
        dataset_name="toy",
        dataset_version="1",
        study_id="study_test",
    )

    assert result.status is StudyStatus.COMPLETED
    assert loader.calls == 1
    assert task.preprocess_calls == 1
    assert experiment_runner.prepare_calls == 1
    assert experiment_runner.run_calls == 2
    assert task.seen_inputs == {DIAGNOSES, PROCEDURES}
    assert task.seen_model_requirements == {
        ModelRequirement.EHR_MEDICATION_GRAPH
    }
    assert loader.request.requirements == frozenset(
        {
            DataRequirement.DIAGNOSES,
            DataRequirement.PROCEDURES,
            DataRequirement.MEDICATIONS,
            DataRequirement.VISIT_TIMES,
        }
    )
    assert RecordingAdapter.requirements == [
        {DIAGNOSES},
        {PROCEDURES},
    ]

    experiment_datasets = [
        json.loads(
            (
                tmp_path
                / "study_test"
                / "experiments"
                / experiment_id
                / "reproducibility.json"
            ).read_text()
        )["dataset"]
        for experiment_id in ("exp_a", "exp_b")
    ]
    assert (
        experiment_datasets[0]["fingerprint"]
        == experiment_datasets[1]["fingerprint"]
    )
    assert {
        split["name"]: split["fingerprint"]
        for split in experiment_datasets[0]["splits"]
    } == {
        split["name"]: split["fingerprint"]
        for split in experiment_datasets[1]["splits"]
    }

    study_dir = tmp_path / "study_test"
    for relative_path in (
        "study.json",
        "summary.json",
        "results.csv",
        "logs/study.log",
        "experiments/exp_a/results.json",
        "experiments/exp_b/results.json",
    ):
        assert (study_dir / relative_path).is_file()

    with (study_dir / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["experiment_id"] for row in rows} == {"exp_a", "exp_b"}
    assert {row["model"] for row in rows} == {"ModelA", "ModelB"}
    assert {row["run_id"] for row in rows} == {"run_001", "run_002"}
    assert {int(row["seed"]) for row in rows} == {7, 11}
    assert {row["status"] for row in rows} == {"completed"}
    assert all(row["jaccard"] for row in rows)

    manifest = json.loads((study_dir / "study.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["comparison_valid"] is True
    assert [
        item["artifact_path"] for item in manifest["experiments"]
    ] == ["experiments/exp_a", "experiments/exp_b"]
    assert result.summary.num_successful_runs == 4
    assert set(result.experiments) == {"exp_a", "exp_b"}


def test_failed_experiment_is_isolated_and_later_experiment_runs(
    tmp_path: Path,
) -> None:
    experiment_runner = make_runner(tmp_path)
    result = StudyRunner(
        output_root=tmp_path,
        experiment_runner=experiment_runner,
    ).run(
        task=CountingTask(),
        experiments=definitions(first_adapter=FailingAdapter),
        seeds=[3],
        loader=CountingLoader(),
        dataset_path=tmp_path,
        study_id="study_partial",
    )

    assert result.status is StudyStatus.PARTIALLY_COMPLETED
    assert [record.status for record in result.manifest.experiments] == [
        StudyStatus.FAILED,
        StudyStatus.COMPLETED,
    ]
    assert "intentional adaptation failure" in (
        result.manifest.experiments[0].error or ""
    )
    assert (
        tmp_path
        / "study_partial"
        / "experiments"
        / "exp_b"
        / "summary.json"
    ).is_file()


def test_shared_preparation_failure_writes_failed_manifest_and_runs_nothing(
    tmp_path: Path,
) -> None:
    experiment_runner = make_runner(tmp_path)
    runner = StudyRunner(
        output_root=tmp_path,
        experiment_runner=experiment_runner,
    )

    with pytest.raises(RuntimeError, match="shared loading failure"):
        runner.run(
            task=CountingTask(),
            experiments=definitions(),
            seeds=[5],
            loader=CountingLoader(RuntimeError("shared loading failure")),
            dataset_path=tmp_path,
            study_id="study_preparation_failure",
        )

    study_dir = tmp_path / "study_preparation_failure"
    manifest = json.loads((study_dir / "study.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["experiments"] == []
    assert experiment_runner.run_calls == 0
    assert list((study_dir / "experiments").iterdir()) == []
    with (study_dir / "results.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
