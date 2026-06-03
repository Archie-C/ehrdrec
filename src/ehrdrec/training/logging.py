import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

@runtime_checkable
class TrainerLogger(Protocol):
    """
    Implement this protocol to plug any logging backend into Trainer.

    Both methods should be called *after* the relevant event completes, so all
    tensors are already detached and metrics are plain Python floats.
    """
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        """Called at the end of every training epoch."""
        ...

    def on_best_model(
        self,
        epoch: int,
        score: float | None,
        state_dict: dict,
    ) -> None:
        """Called whenever a new best validation score is achieved."""
        ...
        
class ConsoleLogger:
    """Logs epoch summaries to Python's stdlib logging (default: INFO)."""
 
    def __init__(self, name: str = "trainer", level: int = logging.INFO) -> None:
        self._log = logging.getLogger(name)
        self._log.setLevel(level)
 
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        self._log.info(f"Epoch {epoch}")
        if train_metrics:
            row = "  ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            self._log.info(f"  train  {row}")
        if val_metrics:
            row = "  ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            self._log.info(f"  val    {row}")
 
    def on_best_model(
        self,
        epoch: int,
        score: float | None,
        state_dict: dict,
    ) -> None:
        score_str = f"{score:.4f}" if score is not None else "n/a"
        self._log.info(f"  ✓ new best  score={score_str}")

class TensorBoardLogger:
    """Writes scalars to a TensorBoard SummaryWriter.

    Requires ``tensorboard`` or ``tensorboardX`` to be installed.
    The writer is created lazily so import errors surface only when training
    actually starts, not at import time.
    """

    def __init__(self, log_dir: str = "runs", **writer_kwargs) -> None:
        self._log_dir = log_dir
        self._writer_kwargs = writer_kwargs
        self._writer = None  # lazy init

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                from tensorboardX import SummaryWriter  # fallback
            self._writer = SummaryWriter(self._log_dir, **self._writer_kwargs)
        return self._writer

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        writer = self._get_writer()
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, global_step=epoch)
        if val_metrics:
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, global_step=epoch)

    def on_best_model(
        self,
        epoch: int,
        score: float | None,
        state_dict: dict,
    ) -> None:
        if score is not None:
            self._get_writer().add_scalar("best/score", score, global_step=epoch)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            
class CheckpointLogger:
    """Saves the best model's state dict to disk whenever a new best is found.

    Files are named ``best_epoch_{epoch:04d}.pt``. Only the most recent best
    checkpoint is kept by default (``keep_last=True``).
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", keep_last: bool = True) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keep_last = keep_last
        self._last_path: Path | None = None

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        pass  # nothing to do here

    def on_best_model(
        self,
        epoch: int,
        score: float | None,
        state_dict: dict,
    ) -> None:
        path = self._dir / f"best_epoch_{epoch:04d}.pt"
        torch.save(state_dict, path)

        if self._keep_last and self._last_path is not None and self._last_path.exists():
            self._last_path.unlink()

        self._last_path = path
        
class CompositeLogger:
    """Broadcasts events to a list of loggers in order.

    Any exception raised by an individual logger is caught and re-raised only
    after all other loggers have had a chance to run, so a broken
    TensorBoardLogger never silently swallows a checkpoint.
    """

    def __init__(self, loggers: list[TrainerLogger]) -> None:
        self._loggers = list(loggers)

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        errors: list[Exception] = []
        for lg in self._loggers:
            try:
                lg.on_epoch_end(epoch, train_metrics, val_metrics)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise ExceptionGroup("logger errors in on_epoch_end", errors)

    def on_best_model(
        self,
        epoch: int,
        score: float | None,
        state_dict: dict,
    ) -> None:
        errors: list[Exception] = []
        for lg in self._loggers:
            try:
                lg.on_best_model(epoch, score, state_dict)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise ExceptionGroup("logger errors in on_best_model", errors)