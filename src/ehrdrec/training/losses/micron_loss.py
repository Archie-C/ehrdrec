import torch.nn as nn
import torch

from ehrdrec.training.losses.base import LossFunction

class MicronLoss(LossFunction):
    def __init__(self, alpha=0.75, multi_weight=5e-2, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.multi_weight = multi_weight
        self.eps = eps
        
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.mlm_loss_fn = nn.MultiLabelMarginLoss()
        
        self.sample_counter = 0
        self.mean_loss = None
        self.register_buffer("weight", torch.ones(4) / 4)
        
    def _multihot_to_margin_target(self, y: torch.Tensor) -> torch.Tensor:
        """
        Converts multi-hot [B, C] targets into MultiLabelMarginLoss target format [B, C].
        Positive class indices first, remaining positions filled with -1.
        """
        B, C = y.shape
        out = torch.full((B, C), -1, dtype=torch.long, device=y.device)

        for i in range(B):
            pos = torch.nonzero(y[i] > 0.5, as_tuple=False).flatten()
            out[i, :pos.numel()] = pos

        return out
    
    def forward(self, predictions, targets, model_output=None, features=None, losses=None, **kwargs):
        predictions_last = model_output["predictions_last"]
        targets_last = features["medication_history"][:, -1, :]
        
        reconstruction_loss = losses["reconstruction_loss"]
        ddi_loss = losses["ddi_loss"]
        

        loss_bce = (
            self.alpha * self.bce_loss_fn(predictions, targets)
            + (1 - self.alpha) * self.bce_loss_fn(predictions_last, targets_last)
        )
        
        targets_margin = self._multihot_to_margin_target(targets)
        targets_last_margin = self._multihot_to_margin_target(targets_last)
        
        loss_mlm = (
            self.alpha * self.mlm_loss_fn(predictions, targets_margin)
            + (1 - self.alpha) * self.mlm_loss_fn(predictions_last, targets_last_margin)
        )
        
        current_loss = torch.stack([
            loss_bce.detach(),
            loss_mlm.detach(),
            ddi_loss.detach(),
            reconstruction_loss.detach()
        ])
        
        if self.mean_loss is None:
            self.mean_loss = current_loss.clone()
        else:
            ratio = (current_loss - self.mean_loss) / (self.mean_loss + self.eps)
            instant_weight = torch.softmax(ratio, dim=0).to(predictions.device)
            
            self.weight = self.weight.to(predictions.device)
            self.weight = 0.75 * instant_weight + 0.25 * self.weight
            
            self.mean_loss = (
                self.mean_loss.to(predictions.device) * self.sample_counter + current_loss
            ) / (self.sample_counter + 1)
        self.sample_counter += 1
        
        lambda1, lambda2, lambda3, lambda4 = self.weight
        
        # This is currently disabled as I need a way to figure this out without making it very slow
        # if current_ddi_rate >= self.ddi_threshold:
        #     loss = lambda1 * loss_bce + lambda2 * loss_mlm + lambda3 * ddi_loss + lambda4 * reconstruction_loss
        # else:
        #     loss = lambda1 * loss_bce + lambda2 * loss_mlm + lambda4 * reconstruction_loss
        loss = lambda1 * loss_bce + lambda2 * loss_mlm + lambda4 * reconstruction_loss
        return loss