from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import polars as pl
import torch

from ehrdrec.contracts.experiment_output import (
    Experiment,
    ExperimentResults,
    ExperimentSummary,
    MetricSummary,
    PredictionRow,
    RunStatus,
)


def to_jsonable(value: Any) -> Any:
    """Convert supported artifact values to deterministic JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Enum):
        return to_jsonable(value.value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        return {
            _json_key(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, (set, frozenset)):
        converted = [to_jsonable(item) for item in value]
        return sorted(converted, key=stable_json_dumps)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy":
        if hasattr(value, "tolist"):
            return to_jsonable(value.tolist())
        if hasattr(value, "item"):
            return to_jsonable(value.item())

    raise TypeError(
        "Unsupported experiment artifact value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _json_key(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, (str, int, float, bool, Path)):
        return str(value)
    raise TypeError(
        "Experiment artifact mappings require scalar keys, received "
        f"{type(value).__name__}."
    )


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value: Any) -> str:
    return sha256_text(stable_json_dumps(value))


def summarize_experiment(
    results: ExperimentResults,
) -> ExperimentSummary:
    """Aggregate test metrics from completed runs using sample std."""

    metric_values: dict[str, list[float]] = defaultdict(list)
    completed = []

    for run in results.runs.values():
        if run.status is not RunStatus.COMPLETED:
            continue
        completed.append(run)
        for name, value in run.metrics.get("test", {}).items():
            metric_values[name].append(float(value))

    summaries = {}
    for name, values in sorted(metric_values.items()):
        summaries[name] = MetricSummary(
            mean=statistics.fmean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
        )

    return ExperimentSummary(
        experiment_id=results.experiment_id,
        model_name=results.model_name,
        task=results.task,
        num_runs=len(completed),
        test_metrics=summaries,
        total_run_time=sum(
            run.run_time.total or 0.0
            for run in completed
        ),
    )


def write_predictions(
    path: str | Path,
    predictions: list[PredictionRow],
) -> None:
    """Write medication-set predictions as typed nested Parquet columns."""

    rows = [
        {
            "run_id": row.run_id,
            "example_id": row.example_id,
            "ground_truth": to_jsonable(row.ground_truth),
            "prediction": to_jsonable(row.prediction),
            "scores": to_jsonable(row.scores),
        }
        for row in predictions
    ]

    schema = {
        "run_id": pl.String,
        "example_id": pl.String,
        "ground_truth": pl.List(pl.Int64),
        "prediction": pl.List(pl.Int64),
        "scores": pl.List(pl.Float64),
    }
    frame = pl.DataFrame(rows, schema=schema, strict=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def write_experiment_artifacts(
    experiment: Experiment,
    output_dir: str | Path,
    histories: Mapping[str, Any],
) -> None:
    """Serialize the canonical Experiment bundle into separate artifacts."""

    output_dir = Path(output_dir)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "results.json", experiment.results)
    write_json(output_dir / "summary.json", experiment.results_summary)
    write_json(output_dir / "reproducibility.json", experiment.reproducibility)
    write_predictions(output_dir / "predictions.parquet", experiment.predictions)

    for run_id, history in sorted(histories.items()):
        write_json(history_dir / f"{run_id}.json", history)
