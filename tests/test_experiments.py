from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from ehrdrec.contracts.adapters import (
    AdapterOutput,
    BatchTargetSpec,
)
from ehrdrec.contracts.experiment_output import (
    ExperimentResults,
    RunResults,
    RunStatus,
    RunTimes,
)
from ehrdrec.contracts.models import ModelContext, ModelOutput
from ehrdrec.data.torch import EHRBatchCollator, EHRDataset
from ehrdrec.evaluation.metrics import Jaccard
from ehrdrec.experiments import (
    ExperimentRunner,
    capture_hardware_information,
    capture_software_environment,
    set_seed,
    sha256_file,
    summarize_experiment,
    to_jsonable,
)
from ehrdrec.models.base import TorchEHRDrecModel
from ehrdrec.requirements import Representation
from ehrdrec.tasks.base import Task
from ehrdrec.training import Trainer, TrainerConfig


@dataclass
class ExampleArtifact:
    status: RunStatus
    path: Path
    values: np.ndarray


class ToyTask(Task):
    def preprocess(
        self,
        raw_frames,
        input_requirements,
        model_requirements=None,
    ):
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
            vocab={
                "medications": SimpleNamespace(vocab_size=2),
            },
        )

    def loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            outputs,
            targets,
        )


class ToyLoader:
    def load(self, path, request):
        return {}


class ToyAdapter:
    def __init__(self, task_output, input_requirements):
        self.task_output = task_output

    def adapt(self) -> AdapterOutput:
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


class ToyModel(TorchEHRDrecModel):
    def __init__(self, context, initial_value: float = 0.0):
        super().__init__(context)
        self.logits = torch.nn.Parameter(
            torch.full((2,), initial_value)
        )

    def forward(self, batch):
        return ModelOutput(
            scores=self.logits.unsqueeze(0).expand(
                batch.targets.shape[0],
                -1,
            )
        )


def _results(
    statuses_and_values: list[tuple[RunStatus, float]],
) -> ExperimentResults:
    return ExperimentResults(
        experiment_id="exp_test",
        model_name="Toy",
        task="ToyTask",
        runs={
            f"run_{index:03d}": RunResults(
                seed=index,
                metrics={"test": {"jaccard": value}},
                run_time=RunTimes(total=1.0),
                status=status,
            )
            for index, (status, value) in enumerate(
                statuses_and_values,
                start=1,
            )
        },
    )


def test_seed_initialization_is_repeatable() -> None:
    set_seed(42)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(2),
    )
    set_seed(42)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(2),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_summary_uses_sample_std_and_excludes_failed_runs() -> None:
    summary = summarize_experiment(
        _results(
            [
                (RunStatus.COMPLETED, 1.0),
                (RunStatus.FAILED, 100.0),
                (RunStatus.COMPLETED, 3.0),
            ]
        )
    )

    assert summary.num_runs == 2
    assert summary.test_metrics["jaccard"].mean == 2.0
    assert summary.test_metrics["jaccard"].std == pytest.approx(2**0.5)
    assert summary.total_run_time == 2.0


def test_single_run_summary_uses_zero_std() -> None:
    summary = summarize_experiment(
        _results([(RunStatus.COMPLETED, 0.75)])
    )
    assert summary.test_metrics["jaccard"].std == 0.0


def test_artifact_serializer_handles_dataclasses_enums_paths_and_numpy(
    tmp_path: Path,
) -> None:
    converted = to_jsonable(
        ExampleArtifact(
            status=RunStatus.COMPLETED,
            path=tmp_path,
            values=np.array([1, 2]),
        )
    )

    assert converted == {
        "status": "completed",
        "path": str(tmp_path),
        "values": [1, 2],
    }
    json.dumps(converted)


def test_environment_and_cpu_only_hardware_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    software = capture_software_environment()
    hardware = capture_hardware_information()

    assert software.python_version
    assert software.ehrdrec_version
    assert hardware.cpu_count is None or hardware.cpu_count > 0
    assert hardware.gpu_count == 0
    assert hardware.gpu is None


def test_experiment_runner_writes_complete_artifact_bundle(
    tmp_path: Path,
) -> None:
    runner = ExperimentRunner(
        output_root=tmp_path,
        trainer_config=TrainerConfig(epochs=2, device="cpu"),
        metrics=[Jaccard()],
        batch_size=2,
    )

    experiment = runner.run(
        model=ToyModel,
        task=ToyTask(),
        seeds=[7, 11],
        model_config={"initial_value": 0.1},
        optimizer_factory=lambda parameters: torch.optim.SGD(
            parameters,
            lr=0.1,
        ),
        loader=ToyLoader(),
        adapter_factory=ToyAdapter,
        dataset_path=tmp_path,
        dataset_name="toy",
        dataset_version="1",
        experiment_id="exp_test",
    )

    artifact_dir = tmp_path / "exp_test"
    assert experiment.results_summary.num_runs == 2
    assert set(experiment.results.runs) == {"run_001", "run_002"}
    assert all(
        run.status is RunStatus.COMPLETED
        for run in experiment.results.runs.values()
    )
    assert all(row.run_id for row in experiment.predictions)
    assert all(row.example_id for row in experiment.predictions)

    for relative_path in (
        "results.json",
        "summary.json",
        "predictions.parquet",
        "reproducibility.json",
        "logs/experiment.log",
        "history/run_001.json",
        "history/run_002.json",
        "models/run_001.pt",
        "models/run_002.pt",
    ):
        assert (artifact_dir / relative_path).is_file()

    prediction_frame = pl.read_parquet(
        artifact_dir / "predictions.parquet"
    )
    assert prediction_frame.height == 4
    assert set(prediction_frame["run_id"]) == {"run_001", "run_002"}
    assert prediction_frame["example_id"].null_count() == 0

    for run_id, model_artifact in experiment.models.items():
        model_path = artifact_dir / model_artifact.filename
        assert model_artifact.sha256 == sha256_file(model_path)
        assert run_id in model_path.name

    reproducibility = json.loads(
        (artifact_dir / "reproducibility.json").read_text()
    )
    assert reproducibility["experiment_id"] == "exp_test"
    assert [run["seed"] for run in reproducibility["runs"]] == [7, 11]

    history = json.loads(
        (artifact_dir / "history" / "run_001.json").read_text()
    )
    assert history["best_epoch"] in {1, 2}
    assert history["training_time"] >= 0.0
    assert history["validation_time"] >= 0.0


def test_failed_run_is_recorded_and_later_seed_continues(
    tmp_path: Path,
) -> None:
    calls = 0

    def optimizer_factory(parameters):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("intentional optimizer failure")
        return torch.optim.SGD(parameters, lr=0.1)

    runner = ExperimentRunner(
        output_root=tmp_path,
        trainer_config=TrainerConfig(epochs=1, device="cpu"),
        metrics=[Jaccard()],
        batch_size=2,
    )
    experiment = runner.run(
        model=ToyModel,
        task=ToyTask(),
        seeds=[1, 2],
        optimizer_factory=optimizer_factory,
        loader=ToyLoader(),
        adapter_factory=ToyAdapter,
        dataset_path=tmp_path,
        experiment_id="exp_failure",
    )

    assert experiment.results.runs["run_001"].status is RunStatus.FAILED
    assert "intentional optimizer failure" in (
        experiment.results.runs["run_001"].error or ""
    )
    assert experiment.results.runs["run_002"].status is RunStatus.COMPLETED
    assert set(experiment.models) == {"run_002"}
    assert experiment.results_summary.num_runs == 1
    assert {
        row.run_id for row in experiment.predictions
    } == {"run_002"}


def test_trainer_fit_remains_usable_without_validation() -> None:
    task = ToyTask()
    task_output = task.preprocess({}, set())
    adapted = ToyAdapter(task_output, set()).adapt()
    collator = EHRBatchCollator(
        fields=adapted.fields,
        target=adapted.target,
    )
    loader = DataLoader(
        EHRDataset(adapted.train),
        batch_size=2,
        collate_fn=collator,
    )
    context = ModelContext.from_task_output(
        vocabs=task_output.vocab,
        task_loss=task.loss,
    )
    model = ToyModel(context)
    result = Trainer(
        TrainerConfig(epochs=1, device="cpu")
    ).fit(
        model=model,
        train_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    assert result.total_steps == 1
    assert result.best_epoch is None
    assert result.epochs[0].metrics is None
