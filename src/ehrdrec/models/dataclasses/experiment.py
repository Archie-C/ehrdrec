from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Serialisable record of every setting needed to reproduce a run.

    Common training knobs are first-class fields; anything model-specific goes
    in ``model_kwargs``.  The config can be round-tripped through JSON so you
    always know exactly what settings produced a checkpoint.
    """
    # --- data source ---
    dataset_path: str = ""
    dataset_name: str = ""          # e.g. "mimic-iii", "mimic-iv"

    # --- data processing ---
    atc_level: int = 5
    minimum_admissions: int = 2

    # --- architecture (derived from data, needed to reconstruct the model) ---
    input_size: int = 0
    output_size: int = 0

    # --- data loader ---
    batch_size: int = 32

    # --- training ---
    epochs: int = 10
    lr: float = 1e-3
    seed: int | None = None

    # --- tuning ---
    n_tuning_trials: int = 0
    tuning_epochs: int = 0
    tuning_metric: str = ""

    # --- model (arbitrary, model-specific params) ---
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # --- free-form notes ---
    notes: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
