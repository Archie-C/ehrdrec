from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(slots=True)
class TrainingResults:
    final_train_loss: float
    final_val_score: float | None
    best_val_score: float | None
    best_model_state: dict
    best_train_metrics: dict[str, float]
    best_val_metrics: dict[str, float]
    best_epoch: int
    seed: int | None = field(default=None)
    # per-epoch history: list of {"epoch": int, "train": {...}, "val": {...}}
    history: list[dict] = field(default_factory=list)

    def save(self, directory: str | Path) -> None:
        """Save metrics/history as JSON and model weights as a .pt file.

        Args:
            directory: Destination folder (created if it does not exist).
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        summary = {
            "final_train_loss": self.final_train_loss,
            "final_val_score": self.final_val_score,
            "best_val_score": self.best_val_score,
            "best_epoch": self.best_epoch,
            "seed": self.seed,
            "best_train_metrics": self.best_train_metrics,
            "best_val_metrics": self.best_val_metrics,
            "history": self.history,
        }
        (out / "training_results.json").write_text(json.dumps(summary, indent=2))
        torch.save(self.best_model_state, out / "best_model.pt")
