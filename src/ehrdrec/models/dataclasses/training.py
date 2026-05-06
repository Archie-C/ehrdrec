
from dataclasses import dataclass

@dataclass(slots=True)
class TrainingResults:
    final_train_loss: float
    final_val_loss: float | None
    best_val_loss: float | None
    best_model_state: dict
    best_train_metrics: dict[str, float]
    best_val_metrics: dict[str, float]
    best_epoch: int