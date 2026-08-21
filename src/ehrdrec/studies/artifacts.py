from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from ehrdrec.contracts.experiment_output import Experiment
from ehrdrec.contracts.study_output import StudyManifest, StudySummary
from ehrdrec.experiments.artifacts import write_json


def write_study_artifacts(
    *,
    output_dir: str | Path,
    manifest: StudyManifest,
    summary: StudySummary,
    experiments: Mapping[str, Experiment],
) -> None:
    """Write the lightweight Study index and flat per-run results table."""

    output_dir = Path(output_dir)
    write_json(output_dir / "study.json", manifest)
    write_json(output_dir / "summary.json", summary)
    _write_results_csv(
        path=output_dir / "results.csv",
        study_id=manifest.study_id,
        experiments=experiments,
    )


def _write_results_csv(
    *,
    path: Path,
    study_id: str,
    experiments: Mapping[str, Experiment],
) -> None:
    metric_names = sorted(
        {
            name
            for experiment in experiments.values()
            for run in experiment.results.runs.values()
            for name in run.metrics.get("test", {})
        }
    )
    columns = [
        "study_id",
        "experiment_id",
        "model",
        "run_id",
        "seed",
        "status",
        *metric_names,
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for experiment_id, experiment in experiments.items():
            for run_id, run in experiment.results.runs.items():
                row = {
                    "study_id": study_id,
                    "experiment_id": experiment_id,
                    "model": experiment.results.model_name,
                    "run_id": run_id,
                    "seed": run.seed,
                    "status": run.status.value,
                }
                row.update(run.metrics.get("test", {}))
                writer.writerow(row)
