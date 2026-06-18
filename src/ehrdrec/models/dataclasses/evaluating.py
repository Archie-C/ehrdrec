from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass(slots=True)
class EvaluationResults:
    test_metrics: dict[str, float]
    # Raw sigmoid outputs and ground-truth targets, present when
    # Evaluator(save_predictions=True) is used.
    predictions: Optional[torch.Tensor] = field(default=None)
    targets: Optional[torch.Tensor] = field(default=None)

    def save(self, directory: str | Path) -> None:
        """Save metrics as JSON and (optionally) raw predictions as a .pt file.

        Args:
            directory: Destination folder (created if it does not exist).
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        (out / "evaluation_results.json").write_text(
            json.dumps(self.test_metrics, indent=2)
        )

        if self.predictions is not None and self.targets is not None:
            torch.save(
                {"predictions": self.predictions.cpu(), "targets": self.targets.cpu()},
                out / "predictions.pt",
            )
