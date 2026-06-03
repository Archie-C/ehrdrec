import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ehrdrec.models.dataclasses import TrainingResults
from ehrdrec.training import BaseTrainer
from ehrdrec.training.logging import TrainerLogger

logger = logging.getLogger(__name__)


def medication_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """
    BCE reconstruction loss for sparse medication multi-hot vectors.

    pos_weight prevents the decoder from learning the cheap all-zero solution.
    """
    weight = torch.full(
        (targets.size(-1),),
        pos_weight,
        dtype=targets.dtype,
        device=targets.device,
    )

    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=weight,
    )


def pretrain_target_space(
    model: nn.Module,
    train_loader,
    val_loader=None,
    *,
    device: str | torch.device = "cuda",
    epochs: int = 20,
    lr: float = 1e-3,
    pos_weight: float = 5.0,
) -> None:
    """
    Pretrains:
        medication multihot -> target_encoder -> z_target -> target_decoder -> medication multihot

    After this, freeze target_encoder before JEPA.
    """
    
    best_val_loss = float("inf")
    best_state = None

    model.to(device)

    # Only train target-space modules here.
    model.freeze_all()
    model.unfreeze("target_encoder", "target_decoder")

    optimizer = torch.optim.AdamW(
        list(model.target_encoder.parameters())
        + list(model.target_decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for _, targets in train_loader:
            targets = targets.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            z_target = model.target_encoder(targets)
            recon_logits = model.target_decoder(z_target)

            loss = medication_reconstruction_loss(
                recon_logits,
                targets,
                pos_weight=pos_weight,
            )

            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / max(total_samples, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for _, targets in val_loader:
                    targets = targets.to(device).float()

                    z_target = model.target_encoder(targets)
                    recon_logits = model.target_decoder(z_target)

                    loss = medication_reconstruction_loss(
                        recon_logits,
                        targets,
                        pos_weight=pos_weight,
                    )

                    batch_size = targets.size(0)
                    total_val_loss += loss.item() * batch_size
                    total_val_samples += batch_size

            val_loss = total_val_loss / max(total_val_samples, 1)

        if val_loss is None:
            print(f"[target pretrain] epoch={epoch} train_loss={train_loss:.4f}")
        else:
            print(
                f"[target pretrain] epoch={epoch} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )
            
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "target_encoder": copy.deepcopy(model.target_encoder.state_dict()),
                "target_decoder": copy.deepcopy(model.target_decoder.state_dict()),
            }
            
    if best_state is not None:
        model.target_encoder.load_state_dict(best_state["target_encoder"])
        model.target_decoder.load_state_dict(best_state["target_decoder"])

    # Freeze target space after pretraining.
    model.freeze("target_encoder", "target_decoder")

    # Unfreeze the JEPA modules needed for the next stage.
    model.unfreeze("context_encoder", "jepa_predictor")

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FreezeConfig:
    """
    Declares which model modules to freeze at the start of each training stage.
    Modules are identified by name — must match DrugJEPA.MODULE_NAMES:

        "context_encoder", "med_encoder", "jepa_predictor", "prediction_head"

    Any module not listed is left in its current state (neither frozen nor
    explicitly unfrozen).  Set unfreeze_all_before_stage=True to reset all
    parameters to trainable before applying the freeze list — useful when you
    want a clean slate at each stage boundary.

    Examples
    --------
    # Classic: freeze encoder while training head, unfreeze for finetune.
    head_freeze    = FreezeConfig(freeze=["context_encoder"], unfreeze_all_before_stage=True)
    finetune_freeze = FreezeConfig(unfreeze_all_before_stage=True)

    # Freeze JEPA modules after pretraining, never touch them again.
    head_freeze    = FreezeConfig(
        freeze=["context_encoder", "med_encoder", "jepa_predictor"],
        unfreeze_all_before_stage=False,
    )
    finetune_freeze = FreezeConfig(
        freeze=["med_encoder", "jepa_predictor"],
        unfreeze_all_before_stage=False,
    )

    # Ablation: train only the head from a frozen encoder, no finetune.
    head_freeze = FreezeConfig(freeze=["context_encoder"])
    """

    freeze: list[str] = field(default_factory=list)
    unfreeze_all_before_stage: bool = False


@dataclass
class StageConfig:
    """
    Controls epoch counts and per-stage freeze behaviour.

    Stages
    ------
    jepa      : JEPA pretraining via model.forward_jepa().
    head      : Supervised head training (context encoder frozen by default).
    finetune  : Optional supervised fine-tuning (all modules trainable by default).

    Freeze defaults match the original behaviour when FreezeConfig is omitted:
        - head stage     : context_encoder frozen
        - finetune stage : context_encoder unfrozen, everything else as-is

    Override by passing explicit FreezeConfig objects.

    Examples
    --------
    # Standard staged training (default behaviour).
    StageConfig(jepa_epochs=50, head_epochs=20, finetune_epochs=5)

    # Ablation: no pretraining, supervised only.
    StageConfig(jepa_epochs=0, head_epochs=30, finetune_epochs=10,
                head_freeze=FreezeConfig())  # nothing frozen

    # Ablation: pretrain then full finetune only (no frozen head stage).
    StageConfig(jepa_epochs=50, head_epochs=0, finetune_epochs=20,
                finetune_freeze=FreezeConfig())

    # Pretrain, freeze JEPA modules permanently, then supervised.
    StageConfig(
        jepa_epochs=50,
        head_epochs=20,
        finetune_epochs=10,
        head_freeze=FreezeConfig(
            freeze=["context_encoder", "med_encoder", "jepa_predictor"],
        ),
        finetune_freeze=FreezeConfig(
            freeze=["med_encoder", "jepa_predictor"],
            unfreeze_all_before_stage=True,
        ),
    )
    """

    jepa_epochs:     int = 10
    head_epochs:     int = 10
    finetune_epochs: int = 0

    # None = use legacy defaults (context_encoder frozen for head,
    #        context_encoder unfrozen for finetune).
    head_freeze:     FreezeConfig | None = None
    finetune_freeze: FreezeConfig | None = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class StagedJEPATrainer(BaseTrainer):
    """
    Trainer for a JEPA-style drug recommendation model.

    Training stages
    ---------------
    Stage 1 — JEPA pretraining
        clinical context -> context_encoder -> jepa_predictor -> predicted med embedding
        medication labels -> med_encoder -> true med embedding
        Loss: model.forward_jepa(features, targets)["loss"]

    Stage 2 — supervised head training
        context_encoder frozen (by default).
        clinical context -> context_encoder -> prediction_head -> logits
        Loss: self.loss_fn(logits, targets)

    Stage 3 — optional supervised fine-tuning
        context_encoder unfrozen (by default).
        Same forward pass as stage 2.

    Freeze behaviour is fully controlled via StageConfig.head_freeze and
    StageConfig.finetune_freeze.  See FreezeConfig and StageConfig docstrings.

    Expected model interface
    ------------------------
        model.forward_jepa(features, targets) -> dict containing "loss"
        model.forward_supervised(features)    -> dict containing "predictions"
        model.freeze(*names)                  -> freeze named modules
        model.unfreeze(*names)                -> unfreeze named modules
        model.freeze_all() / unfreeze_all()   -> convenience helpers
        model.frozen_modules()                -> set[str] of frozen module names

    Optimisers
    ----------
    Pass separate optimisers for each stage.  If omitted:
        jepa_optimizer     defaults to optimizer
        head_optimizer     defaults to optimizer
        finetune_optimizer defaults to head_optimizer
    """

    def __init__(
        self,
        model:              nn.Module,
        train_loader:       DataLoader,
        val_loader:         DataLoader | None = None,
        loss_fn:            nn.Module | Callable | None = None,
        optimizer:          Optimizer | None = None,
        metrics:            list | None = None,
        target_metric:      str | None = None,
        higher_is_better:   bool = True,
        device:             str | torch.device = "cuda",
        logger:             TrainerLogger | None = None,
        stage_config:       StageConfig | None = None,
        jepa_optimizer:     Optimizer | None = None,
        head_optimizer:     Optimizer | None = None,
        finetune_optimizer: Optimizer | None = None,
    ):
        stage_config = stage_config or StageConfig()

        total_epochs = (
            stage_config.jepa_epochs
            + stage_config.head_epochs
            + stage_config.finetune_epochs
        )

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            target_metric=target_metric,
            higher_is_better=higher_is_better,
            device=device,
            epochs=total_epochs,
            logger=logger,
        )

        self.stage_config = stage_config

        self.jepa_optimizer     = jepa_optimizer or optimizer
        self.head_optimizer     = head_optimizer or optimizer
        self.finetune_optimizer = finetune_optimizer or self.head_optimizer

        if self.jepa_optimizer is None and stage_config.jepa_epochs > 0:
            raise ValueError("jepa_epochs > 0 requires jepa_optimizer or optimizer.")

        if self.head_optimizer is None and stage_config.head_epochs > 0:
            raise ValueError("head_epochs > 0 requires head_optimizer or optimizer.")

        if self.loss_fn is None and (
            stage_config.head_epochs > 0 or stage_config.finetune_epochs > 0
        ):
            raise ValueError("loss_fn is required for supervised head/fine-tune stages.")

        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self) -> TrainingResults:
        global_epoch = 0
        jepa_metrics: dict[str, float] = {}

        # ---- Stage 1: JEPA pretraining --------------------------------
        if self.stage_config.jepa_epochs > 0:
            self._require_method("forward_jepa")

            for _ in range(self.stage_config.jepa_epochs):
                global_epoch += 1
                jepa_metrics = self._train_jepa_one_epoch()

                if self.logger is not None:
                    self.logger.on_epoch_end(global_epoch, jepa_metrics, {})

        best_result: TrainingResults | None = None

        # ---- Stage 2: supervised head training ------------------------
        if self.stage_config.head_epochs > 0:
            self._apply_freeze_config(
                config=self.stage_config.head_freeze,
                legacy_fn=self._legacy_freeze_for_head,
            )

            best_result = self._fit_supervised_stage(
                stage_name="head_train",
                epochs=self.stage_config.head_epochs,
                optimizer=self.head_optimizer,
                global_epoch_start=global_epoch,
            )

            global_epoch += self.stage_config.head_epochs

        # ---- Stage 3: fine-tuning -------------------------------------
        if self.stage_config.finetune_epochs > 0:
            self._apply_freeze_config(
                config=self.stage_config.finetune_freeze,
                legacy_fn=self._legacy_freeze_for_finetune,
            )

            finetune_result = self._fit_supervised_stage(
                stage_name="finetune",
                epochs=self.stage_config.finetune_epochs,
                optimizer=self.finetune_optimizer,
                global_epoch_start=global_epoch,
            )

            best_result = self._select_better_result(best_result, finetune_result)

        if best_result is not None:
            return best_result

        # JEPA-only fallback.
        return TrainingResults(
            final_train_loss=jepa_metrics.get("jepa_loss"),
            final_val_score=None,
            best_val_score=None,
            best_model_state=copy.deepcopy(self.model.state_dict()),
            best_train_metrics=jepa_metrics,
            best_val_metrics={},
            best_epoch=self.stage_config.jepa_epochs,
        )

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _apply_freeze_config(
        self,
        config: FreezeConfig | None,
        legacy_fn: Callable,
    ) -> None:
        """
        Apply a FreezeConfig, or fall back to legacy behaviour if None.
        Logs the resulting frozen module set at DEBUG level.
        """
        if config is None:
            legacy_fn()
        else:
            if config.unfreeze_all_before_stage:
                if hasattr(self.model, "unfreeze_all"):
                    self.model.unfreeze_all()
                else:
                    for p in self.model.parameters():
                        p.requires_grad = True

            if config.freeze:
                if hasattr(self.model, "freeze"):
                    self.model.freeze(*config.freeze)
                else:
                    # Fallback: freeze by attribute name.
                    for name in config.freeze:
                        module = getattr(self.model, name, None)
                        if module is None:
                            raise AttributeError(
                                f"Model has no attribute '{name}' to freeze."
                            )
                        for p in module.parameters():
                            p.requires_grad = False

        # Log current freeze state if the model supports it.
        if hasattr(self.model, "frozen_modules"):
            frozen = self.model.frozen_modules()
            logging.getLogger(__name__).debug(
                "Freeze state: %s",
                frozen if frozen else "all trainable",
            )

    def _legacy_freeze_for_head(self) -> None:
        """Original behaviour: freeze context_encoder before head stage."""
        if hasattr(self.model, "freeze_context_encoder"):
            self.model.freeze_context_encoder()
        elif hasattr(self.model, "context_encoder"):
            for p in self.model.context_encoder.parameters():
                p.requires_grad = False
        else:
            raise AttributeError(
                "Model must implement freeze_context_encoder() or expose context_encoder."
            )

    def _legacy_freeze_for_finetune(self) -> None:
        """Original behaviour: unfreeze context_encoder before finetune stage."""
        if hasattr(self.model, "unfreeze_context_encoder"):
            self.model.unfreeze_context_encoder()
        elif hasattr(self.model, "context_encoder"):
            for p in self.model.context_encoder.parameters():
                p.requires_grad = True
        else:
            raise AttributeError(
                "Model must implement unfreeze_context_encoder() or expose context_encoder."
            )

    # ------------------------------------------------------------------
    # JEPA training
    # ------------------------------------------------------------------

    def _train_jepa_one_epoch(self) -> dict[str, float]:
        self.model.train()

        totals: dict[str, float] = {
            "jepa_loss":     0.0,
            "jepa_sim_loss": 0.0,
            "jepa_var_loss": 0.0,
            "jepa_cov_loss": 0.0,
            "z_pred_std":    0.0,
            "z_med_std":     0.0,
            "z_alignment":   0.0,
            "z_pred_std_max": 0.0,
            "z_pred_std_min": 0.0,
        }
        total_samples = 0

        for features, targets in self.train_loader:
            features = self._move_to_device(features)
            targets  = self._move_to_device(targets)

            self.jepa_optimizer.zero_grad(set_to_none=True)

            output = self.model.forward_jepa(features, targets)
            if not isinstance(output, dict) or "loss" not in output:
                raise ValueError(
                    "model.forward_jepa() must return a dict containing 'loss'."
                )

            loss = output["loss"]
            loss.backward()
            self.jepa_optimizer.step()

            batch_size = self._batch_size(features)
            total_samples += batch_size

            totals["jepa_loss"] += loss.item() * batch_size

            for key in (
                "jepa_sim_loss", "jepa_var_loss", "jepa_cov_loss",
                "z_pred_std", "z_med_std", "z_alignment",
                "z_pred_std_max", "z_pred_std_min",
            ):
                val = output.get(key)
                if val is not None:
                    totals[key] += (
                        val.item() if isinstance(val, torch.Tensor) else float(val)
                    ) * batch_size

        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples.")

        return {k: v / total_samples for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Supervised training
    # ------------------------------------------------------------------

    def _fit_supervised_stage(
        self,
        stage_name: str,
        epochs: int,
        optimizer: Optimizer,
        global_epoch_start: int = 0,
    ) -> TrainingResults:
        best_val_score   = None
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_epoch       = 0
        best_train_metrics: dict[str, float] = {}
        best_val_metrics:   dict[str, float] = {}

        final_train_loss = None
        final_val_score  = None

        for local_epoch in range(1, epochs + 1):
            global_epoch = global_epoch_start + local_epoch

            train_loss, train_metrics = self._train_supervised_one_epoch(optimizer)
            final_train_loss = train_loss
            train_metrics = {"loss": train_loss, **train_metrics}

            val_metrics: dict[str, float] = {}

            if self.val_loader is not None:
                val_metrics = self._validate()
                current     = val_metrics.get(self.target_metric) if self.target_metric else None
                final_val_score = current

                if current is not None:
                    improved = (
                        best_val_score is None
                        or (self.higher_is_better     and current > best_val_score)
                        or (not self.higher_is_better and current < best_val_score)
                    )
                    if improved:
                        best_val_score   = current
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        best_epoch       = global_epoch
                        best_train_metrics = train_metrics
                        best_val_metrics   = val_metrics

                        if self.logger is not None:
                            self.logger.on_best_model(
                                global_epoch, best_val_score, best_model_state,
                            )
                else:
                    # No target metric: keep latest.
                    best_model_state   = copy.deepcopy(self.model.state_dict())
                    best_epoch         = global_epoch
                    best_train_metrics = train_metrics
                    best_val_metrics   = val_metrics

                    if self.logger is not None:
                        self.logger.on_best_model(global_epoch, None, best_model_state)
            else:
                best_model_state   = copy.deepcopy(self.model.state_dict())
                best_epoch         = global_epoch
                best_train_metrics = train_metrics

            if self.logger is not None:
                self.logger.on_epoch_end(global_epoch, train_metrics, val_metrics)

        return TrainingResults(
            final_train_loss=final_train_loss,
            final_val_score=final_val_score,
            best_val_score=best_val_score,
            best_model_state=best_model_state,
            best_train_metrics=best_train_metrics,
            best_val_metrics=best_val_metrics,
            best_epoch=best_epoch,
        )

    def _train_supervised_one_epoch(
        self,
        optimizer: Optimizer,
    ) -> tuple[float, dict[str, float]]:
        self.model.train()

        total_loss   = 0.0
        total_samples = 0

        self._reset_metrics()

        for features, targets in self.train_loader:
            features = self._move_to_device(features)
            targets  = self._move_to_device(targets)

            optimizer.zero_grad(set_to_none=True)

            output  = self._forward_supervised(features)
            logits  = output["predictions"]
            losses  = output.get("losses", None)

            try:
                loss = self.loss_fn(logits, targets, losses=losses)
            except TypeError:
                loss = self.loss_fn(logits, targets)

            loss.backward()
            optimizer.step()

            batch_size    = self._batch_size(features)
            total_loss   += loss.item() * batch_size
            total_samples += batch_size

            self._update_metrics(logits, targets)

        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples.")

        return total_loss / total_samples, self._compute_metrics()

    def _validate(self) -> dict[str, float]:
        self.model.eval()
        self._reset_metrics()

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = self._move_to_device(features)
                targets  = self._move_to_device(targets)

                output = self._forward_supervised(features)
                self._update_metrics(output["predictions"], targets)

        return self._compute_metrics()

    # ------------------------------------------------------------------
    # Utilities (unchanged from original)
    # ------------------------------------------------------------------

    def _forward_supervised(self, features: Any) -> dict[str, torch.Tensor]:
        if hasattr(self.model, "forward_supervised"):
            output = self.model.forward_supervised(features)
        else:
            output = self.model(features)

        if isinstance(output, dict):
            if "predictions" not in output:
                raise ValueError(
                    "Supervised forward must return a dict containing 'predictions'."
                )
            return output

        return {"predictions": output}

    def _move_to_device(self, x: Any) -> Any:
        if isinstance(x, dict):
            return {k: self._move_to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._move_to_device(v) for v in x)
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return x

    def _batch_size(self, features: Any) -> int:
        if isinstance(features, dict):
            first = next(v for v in features.values() if isinstance(v, torch.Tensor))
            return first.size(0)
        if isinstance(features, (list, tuple)):
            first = next(v for v in features if isinstance(v, torch.Tensor))
            return first.size(0)
        if isinstance(features, torch.Tensor):
            return features.size(0)
        raise TypeError(
            "Could not infer batch size. Expected Tensor, dict, list, or tuple."
        )

    def _reset_metrics(self) -> None:
        if self.metrics:
            for m in self.metrics:
                m.reset()

    def _update_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self.metrics:
            for m in self.metrics:
                m.update(outputs.detach(), targets.detach())

    def _compute_metrics(self) -> dict[str, float]:
        if not self.metrics:
            return {}
        results = {}
        for m in self.metrics:
            value = m.compute()
            results[m.name] = value.item() if isinstance(value, torch.Tensor) else value
        return results

    def _require_method(self, method_name: str) -> None:
        if not hasattr(self.model, method_name):
            raise AttributeError(f"Model must implement {method_name}().")

    def _select_better_result(
        self,
        a: TrainingResults | None,
        b: TrainingResults,
    ) -> TrainingResults:
        if a is None:
            return b
        if self.target_metric is None:
            return b

        a_score = a.best_val_score
        b_score = b.best_val_score

        if a_score is None:
            return b
        if b_score is None:
            return a

        return b if (
            b_score > a_score if self.higher_is_better else b_score < a_score
        ) else a