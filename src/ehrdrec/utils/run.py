from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna
    from ehrdrec.mappings.code_to_id.vocab import Vocab
    from ehrdrec.models.dataclasses.experiment import ExperimentConfig
    from ehrdrec.models.dataclasses.evaluating import EvaluationResults
    from ehrdrec.models.dataclasses.training import TrainingResults


def save_run(
    output_dir: str | Path,
    *,
    config: ExperimentConfig,
    training_results: TrainingResults,
    eval_results: EvaluationResults,
    study: optuna.Study | None = None,
    vocabs: dict[str, Vocab] | None = None,
) -> None:
    """Persist every artefact needed to reproduce and interpret a run.

    Directory layout::

        <output_dir>/
            experiment_config.json   — all hyperparameters
            training_results.json    — metrics, per-epoch history
            best_model.pt            — best checkpoint weights
            evaluation_results.json  — test-set metrics
            predictions.pt           — raw sigmoid outputs + targets (if collected)
            vocabs/
                <name>.json          — each Vocab passed in ``vocabs``
            study/
                best_params.json     — winning trial hyperparameters + value
                trials.csv           — full trial history
            environment.json         — Python version + installed package versions
            run_summary.json         — single top-level digest of the entire run

    Args:
        output_dir:        Destination folder (created if it does not exist).
        config:            The ``ExperimentConfig`` for this run.
        training_results:  Returned by ``Trainer.fit()``.
        eval_results:      Returned by ``Evaluator.run()``.
        study:             Optuna study (optional but strongly recommended).
        vocabs:            Mapping of name → Vocab to persist (e.g.
                           ``{"medications": medications_vocab}``).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- experiment config ---
    config.save(out / "experiment_config.json")

    # --- training results (metrics + history + checkpoint) ---
    training_results.save(out)

    # --- evaluation results (metrics + optional predictions tensor) ---
    eval_results.save(out)

    # --- vocabs ---
    if vocabs:
        vocab_dir = out / "vocabs"
        for name, vocab in vocabs.items():
            vocab.save(vocab_dir / f"{name}.json")

    # --- optuna study ---
    if study is not None:
        study_dir = out / "study"
        study_dir.mkdir(exist_ok=True)

        best = {
            "trial_number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        }
        (study_dir / "best_params.json").write_text(json.dumps(best, indent=2))

        rows = []
        for t in study.trials:
            row = {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "duration_seconds": (
                    t.duration.total_seconds() if t.duration is not None else None
                ),
            }
            row.update({f"param_{k}": v for k, v in t.params.items()})
            rows.append(row)

        if rows:
            keys = list(rows[0].keys())
            lines = [",".join(keys)]
            for row in rows:
                lines.append(",".join("" if row[k] is None else str(row[k]) for k in keys))
            (study_dir / "trials.csv").write_text("\n".join(lines))

    # --- environment snapshot ---
    env = _capture_environment()
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    # --- run summary ---
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out.resolve()),
        "config": config.to_dict(),
        "best_val_metrics": training_results.best_val_metrics,
        "best_epoch": training_results.best_epoch,
        "test_metrics": {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in eval_results.test_metrics.items()
        },
        "study": (
            {
                "n_trials": len(study.trials),
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
            }
            if study is not None
            else None
        ),
        "vocabs_saved": list(vocabs.keys()) if vocabs else [],
        "python_version": env["python_version"],
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nRun saved to {out.resolve()}/")
    print(f"  experiment_config.json  — hyperparameters")
    print(f"  training_results.json   — metrics + per-epoch history")
    print(f"  best_model.pt           — best checkpoint")
    print(f"  evaluation_results.json — test metrics")
    if eval_results.predictions is not None:
        print(f"  predictions.pt          — raw predictions + targets")
    if vocabs:
        print(f"  vocabs/                 — {', '.join(vocabs)}")
    if study is not None:
        print(f"  study/                  — best_params.json + trials.csv")
    print(f"  environment.json        — package versions")
    print(f"  run_summary.json        — top-level digest")


def _capture_environment() -> dict:
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]
        if name and version:
            packages[name] = version

    git_hash = _git_hash()

    return {
        "python_version": sys.version,
        "packages": dict(sorted(packages.items())),
        "git_commit": git_hash,
    }


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None
