from typing import Callable

import optuna
from torch.utils.data import DataLoader

from ehrdrec.models.dataclasses import TrainingResults
from ehrdrec.training.standard import Trainer


TrialFn = Callable[[optuna.Trial, DataLoader, DataLoader], Trainer]


class Tuner:
    """Hyperparameter tuner wrapping Optuna. The trial_fn receives an optuna.Trial
    and both data loaders, and must return a configured Trainer with trial= set."""

    def __init__(
        self,
        trial_fn: TrialFn,
        n_trials: int = 20,
        direction: str = "maximize",
        pruner: optuna.pruners.BasePruner | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        study_name: str | None = None,
        storage: str | None = None,
    ):
        self.trial_fn = trial_fn
        self.n_trials = n_trials
        self.direction = direction
        self.pruner = pruner or optuna.pruners.MedianPruner()
        self.sampler = sampler
        self.study_name = study_name
        self.storage = storage

    def tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[optuna.Study, TrainingResults]:
        study = optuna.create_study(
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.storage is not None,
        )

        best_results: TrainingResults | None = None

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_results

            trainer = self.trial_fn(trial, train_loader, val_loader)
            results = trainer.fit()

            if results.best_val_score is None:
                raise ValueError(
                    "Trainer produced no best_val_score. "
                    "Ensure target_metric is set and a val_loader is provided."
                )

            if (
                best_results is None
                or (self.direction == "maximize" and results.best_val_score > best_results.best_val_score)
                or (self.direction == "minimize" and results.best_val_score < best_results.best_val_score)
            ):
                best_results = results

            return results.best_val_score

        study.optimize(objective, n_trials=self.n_trials)
        return study, best_results
