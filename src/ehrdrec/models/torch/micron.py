import torch
import torch.nn as nn

class Micron(nn.Module):
    def __init__(
        self, 
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        ddi_adjacency_matrix: torch.Tensor,
        embedding_dim: int = 128,
        dropout: float = 0.5,
        return_losses: bool = False,
    ) -> None:
        super(Micron, self).__init__()
        
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.return_losses = return_losses
        
        self.register_buffer(
            "ddi_adjacency_matrix",
            ddi_adjacency_matrix.float(),
        )
        
        self.diagnoses_encoder = nn.Linear(n_diagnoses, embedding_dim)
        self.procedures_encoder = nn.Linear(n_procedures, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        self.health_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim)
        )
        
        self.prescription_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, n_medications)
        )
        
        self.init_weights()
        
    def init_weights(self):
        init_range = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def _health_rep(self, x_diag, x_proc):
        diag = self.dropout(self.diagnoses_encoder(x_diag))
        proc = self.dropout(self.procedures_encoder(x_proc))
        patient_rep = torch.cat([diag, proc], dim=-1)      # (batch, visits, 2*dim)
        return self.health_net(patient_rep)                # (batch, visits, dim)

    def forward(self, x):
        x_diag = x["diagnoses"]                  # (batch, visits, n_diagnoses)
        x_proc = x["procedures"]                 # (batch, visits, n_procedures)
        lengths = x["lengths"].to(x_diag.device) # (batch,)

        batch_size = x_diag.size(0)
        
        # health representation for every visit
        health = self._health_rep(x_diag, x_proc)          # (batch, visits, dim)

        # gather each patient's true last and second-to-last visit via lengths
        last_idx = (lengths - 1).clamp(min=0)
        prev_idx = (lengths - 2).clamp(min=0)
        has_prev = (lengths >= 2)
        
        b = torch.arange(batch_size, device=x_diag.device)
        health_rep = health[b, last_idx]
        health_rep_last = health[b, prev_idx]
        
        # if patient has no previous visit, zero out the second-to-last visit representation so it doesn't contribute to the prescription
        health_rep_last = health_rep_last * has_prev.unsqueeze(-1).float()
        
        health_residual_rep = health_rep - health_rep_last
        
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)
        
        if not self.return_losses or not self.training:
            return { "predictions": drug_rep, "losses": None }
        
        reconstruction_loss = torch.mean(
            torch.pow(
                torch.sigmoid(drug_rep) - torch.sigmoid(drug_rep_last + drug_residual_rep), 2
            )
        )
        
        neg_pred_prob = torch.sigmoid(drug_rep)
        pairwise = neg_pred_prob.unsqueeze(2) * neg_pred_prob.unsqueeze(1)
        ddi_loss = (
            pairwise.mul(self.ddi_adjacency_matrix).sum(dim=(1, 2))
            / self.ddi_adjacency_matrix.shape[0]
        ).mean()
        
        return {
            "predictions": drug_rep,
            "losses" : {
                "reconstruction_loss": reconstruction_loss,
                "ddi_loss": ddi_loss,
            }
        }
        
        