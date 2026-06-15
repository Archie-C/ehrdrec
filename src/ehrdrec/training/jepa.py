import copy
from dataclasses import field, dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ehrdrec.training.standard import Trainer
from ehrdrec.training.logging import TrainerLogger
from ehrdrec.models.dataclasses import TrainingResults

@dataclass
class FreezeConfig:
    while_pretraining_medication_encoder: list[str] = field(default_factory=list)
    while_pretraining_jepa: list[str] = field(default_factory=list)
    while_training_target_decoder: list[str] = field(default_factory=list)
    while_fine_tuning: list[str] = field(default_factory=list)


class StagedJepaTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        metrics: list | None = None,
        target_metric: str | None = None,
        higher_is_better: bool = True,
        device: str | torch.device = "cuda",
        epochs: int = 10,
        logger: TrainerLogger | None = None,
        targets_for_medication_encoder: list[str] | None = None,
        freeze_config: FreezeConfig | None = None,
        med_encoder_epochs: int | None = None,
        jepa_epochs: int | None = None,
        head_epochs: int | None = None,
        finetune_epochs: int | None = None,
        med_encoder_optimizer: Optimizer | None = None,
        jepa_optimizer: Optimizer | None = None,
        head_optimizer: Optimizer | None = None,
        finetune_optimizer: Optimizer | None = None,
    ):
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
            epochs=epochs,
            logger=logger,
        )
        self.targets_for_medication_encoder = targets_for_medication_encoder
        self.freeze_config = freeze_config or FreezeConfig()
        self._med_encoder_epochs  = med_encoder_epochs  if med_encoder_epochs  is not None else epochs
        self._jepa_epochs         = jepa_epochs         if jepa_epochs         is not None else epochs
        self._head_epochs         = head_epochs         if head_epochs         is not None else epochs
        self._finetune_epochs     = finetune_epochs     if finetune_epochs     is not None else epochs
        self._med_encoder_optimizer = med_encoder_optimizer or optimizer
        self._jepa_optimizer        = jepa_optimizer        or optimizer
        self._head_optimizer        = head_optimizer        or optimizer
        self._finetune_optimizer    = finetune_optimizer    or optimizer
        
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        self._reset_metrics()

        with torch.no_grad():
            for features, atcs in self.val_loader:
                features = features.to(self.device)
                atc5 = atcs["atc5"].to(self.device) if isinstance(atcs, dict) else atcs.to(self.device)

                logits = self.model(features)
                self._update_metrics(logits, atc5)

        return self._compute_metrics()

    def fit(self) -> TrainingResults:
        self._reset_results()
        self._pretrain_medication_encoder()
        self._pretrain_jepa()
        self._train_target_decoder()
        self._fine_tune()
        return self.results
    
    def _reset_results(self) -> TrainingResults:
        self.results = TrainingResults(
            best_epoch=0,
            best_model_state=copy.deepcopy(self.model.state_dict()),
            best_train_metrics={},
            best_val_metrics={},
            final_train_loss=None,
            final_val_score=None,
            best_val_score=None,
        )
        
    # The idea here is the build a good medication encoder first
    # We aim to reconstruct all the ATC levels given the multihot vector of ATC level 5 codes
    # This should hopefully help the model learn a heirarchical representation of the medications
    def _pretrain_medication_encoder(self):
        self.optimizer = self._med_encoder_optimizer
        for epoch in range(1, self._med_encoder_epochs + 1):
            train_loss = self._pretrain_medication_encoder_one_epoch()
            print(f"Pretraining medication encoder - Epoch {epoch}/{self._med_encoder_epochs}, Loss: {train_loss:.4f}")
    
    def _pretrain_medication_encoder_one_epoch(self):
        self.model.train()
        self.model.unfreeze_all()
        self.model.freeze(self.freeze_config.while_pretraining_medication_encoder)
        
        total_loss = 0.0
        total_samples = 0
        
        for _, atcs in self.train_loader:
            if isinstance(atcs, dict):
                atcs = {k: v.to(self.device) for k, v in atcs.items()}
            else:
                atcs = atcs.to(self.device)
                
            self.optimizer.zero_grad(set_to_none=True)
            
            loss = self.model.forward_medication_encoder(atcs)
            loss.backward()
            self.optimizer.step()
            
            batch_size = next(iter(atcs.values())).size(0) if isinstance(atcs, dict) else atcs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples

    # The next step is to pretrain the JEPA itself, using the frozen medication encoder
    # Here we map features to the target embedding space and try to reconstruct the target embedding from the context embedding
    # This should hopefully help the model learn to extract useful information from the features and encode it in the embedding space
    def _pretrain_jepa(self):
        self.optimizer = self._jepa_optimizer
        for epoch in range(1, self._jepa_epochs + 1):
            train_loss = self._pretrain_jepa_one_epoch()
            print(f"Pretraining JEPA - Epoch {epoch}/{self._jepa_epochs}, Loss: {train_loss:.4f}")
    
    def _pretrain_jepa_one_epoch(self):
        self.model.train()
        self.model.unfreeze_all()
        self.model.freeze(self.freeze_config.while_pretraining_jepa)
        
        total_loss = 0.0
        total_samples = 0
        
        for features, atcs in self.train_loader:
            features = features.to(self.device)
            atc5 = atcs["atc5"].to(self.device) if isinstance(atcs, dict) else atcs.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            loss = self.model.forward_jepa(features, atc5)
            loss.backward()
            self.optimizer.step()
            
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples
    
    # The next step is to train the target decoder, using the outputs from the context encoder and maybe predictor layer
    def _train_target_decoder(self):
        self.optimizer = self._head_optimizer
        for epoch in range(1, self._head_epochs + 1):
            train_loss, train_metrics = self._train_target_decoder_one_epoch()
            self.results.final_train_loss = train_loss
            
            if self.val_loader is not None:
                val_metrics = self._validate()
                current = val_metrics.get(self.target_metric) if self.target_metric else None
                self.results.final_val_score = current
                
                if current is not None:
                    improved = (
                        self.results.best_val_score is None or
                        (self.higher_is_better and current > self.results.best_val_score) or
                        (not self.higher_is_better and current < self.results.best_val_score)
                    )
                    if improved:
                        self.results.best_val_score = current
                        self.results.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.results.best_epoch = epoch
                        self.results.best_train_metrics = train_metrics
                        self.results.best_val_metrics = val_metrics
                        if self.logger is not None:
                            self.logger.on_best_model(epoch, current, self.results.best_model_state)
                else:
                    # no target metric, just keep latest
                    self.results.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.results.best_epoch = epoch
                    self.results.best_train_metrics = train_metrics
                    self.results.best_val_metrics = val_metrics
                    if self.logger is not None:
                        self.logger.on_best_model(epoch, None, self.results.best_model_state)
            
            if self.logger is not None:
                self.logger.on_epoch_end(epoch, train_metrics, val_metrics)
    
    def _train_target_decoder_one_epoch(self):
        self.model.train()
        self.model.unfreeze_all()
        self.model.freeze(self.freeze_config.while_training_target_decoder)
        
        total_loss = 0.0
        total_samples = 0
        
        for features, atcs in self.train_loader:
            features = features.to(self.device)
            atc5 = atcs["atc5"].to(self.device) if isinstance(atcs, dict) else atcs.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            logits = self.model.forward_target_decoder(features, atc5)
            loss = self.loss_fn(logits, atc5, losses=None)
            loss.backward()
            self.optimizer.step()
            
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            self._update_metrics(logits, atc5)
            
        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples.")
        
        avg_loss = total_loss / total_samples
        metrics = self._compute_metrics()

        return avg_loss, metrics
    
    
    def _fine_tune(self):
        self.optimizer = self._finetune_optimizer
        for epoch in range(1, self._finetune_epochs + 1):
            train_loss, train_metrics = self._fine_tune_one_epoch()
            self.results.final_train_loss = train_loss
            
            if self.val_loader is not None:
                val_metrics = self._validate()
                current = val_metrics.get(self.target_metric) if self.target_metric else None
                self.results.final_val_score = current
                
                if current is not None:
                    improved = (
                        self.results.best_val_score is None or
                        (self.higher_is_better and current > self.results.best_val_score) or
                        (not self.higher_is_better and current < self.results.best_val_score)
                    )
                    if improved:
                        self.results.best_val_score = current
                        self.results.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.results.best_epoch = epoch
                        self.results.best_train_metrics = train_metrics
                        self.results.best_val_metrics = val_metrics
                        if self.logger is not None:
                            self.logger.on_best_model(epoch, current, self.results.best_model_state)
                else:
                    # no target metric, just keep latest
                    self.results.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.results.best_epoch = epoch
                    self.results.best_train_metrics = train_metrics
                    self.results.best_val_metrics = val_metrics
                    if self.logger is not None:
                        self.logger.on_best_model(epoch, None, self.results.best_model_state)
            
            if self.logger is not None:
                self.logger.on_epoch_end(epoch, train_metrics, val_metrics)
    
    def _fine_tune_one_epoch(self):
        self.model.train()
        self.model.unfreeze_all()
        self.model.freeze(self.freeze_config.while_fine_tuning)
        
        total_loss = 0.0
        total_samples = 0
        
        for features, atcs in self.train_loader:
            features = features.to(self.device)
            atc5 = atcs["atc5"].to(self.device) if isinstance(atcs, dict) else atcs.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            logits = self.model(features)
            loss = self.loss_fn(logits, atc5, losses=None)
            loss.backward()
            self.optimizer.step()
            
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            self._update_metrics(logits, atc5)
            
        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples.")
        
        avg_loss = total_loss / total_samples
        metrics = self._compute_metrics()

        return avg_loss, metrics