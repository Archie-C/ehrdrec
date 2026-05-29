import torch
import torch.nn as nn
from ehrdrec.models.utils import GCN


class GameNetFast(nn.Module):
    def __init__(
        self,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        medication_adjacency_matrix: torch.Tensor,
        ddi_adjacency_matrix: torch.Tensor,
        diagnoses_embedding_dim: int = 128,
        procedures_embedding_dim: int = 128,
        hidden_dim: int = 128,
        query_dim: int = 128,
    ):
        super().__init__()

        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.beta = nn.Parameter(torch.FloatTensor(1))

        self.register_buffer(
            "medication_adjacency_matrix",
            medication_adjacency_matrix.float(),
        )
        self.register_buffer(
            "ddi_adjacency_matrix",
            ddi_adjacency_matrix.float(),
        )

        self.diagnoses_encoder = nn.Linear(n_diagnoses, diagnoses_embedding_dim)
        self.procedures_encoder = nn.Linear(n_procedures, procedures_embedding_dim)

        self.diagnoses_rnn = nn.GRU(
            input_size=diagnoses_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.procedures_rnn = nn.GRU(
            input_size=procedures_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.query_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, query_dim),
        )

        self.ehr_gcn = GCN(
            n_nodes=n_medications,
            embed_dim=query_dim,
        )

        self.ddi_gcn = GCN(
            n_nodes=n_medications,
            embed_dim=query_dim,
        )

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(query_dim * 3, query_dim * 2),
            nn.ReLU(),
            nn.Linear(query_dim * 2, n_medications),
        )
        
        self.init_weights()
    
    def init_weights(self):
        self.beta.data.uniform_(-0.1, 0.1)
        
    def forward(self, x):
        x_diag = x["diagnoses"]                  # (batch, visits, n_diagnoses)
        x_proc = x["procedures"]                 # (batch, visits, n_procedures)
        medication_history = x["medication_history"]  # (batch, visits, n_medications)
        lengths = x["lengths"].to(x_diag.device)       # (batch,)

        encoded_diagnoses = self.diagnoses_encoder(x_diag)
        encoded_procedures = self.procedures_encoder(x_proc)

        diag_out, _ = self.diagnoses_rnn(encoded_diagnoses)
        proc_out, _ = self.procedures_rnn(encoded_procedures)

        patient_repr = torch.cat([diag_out, proc_out], dim=-1)

        queries = self.query_layer(patient_repr)  # (batch, visits, query_dim)

        batch_size = queries.size(0)
        batch_idx = torch.arange(batch_size, device=queries.device)
        current_idx = lengths - 1

        q_t = queries[batch_idx, current_idx]  # (batch, query_dim)

        z_ehr = self.ehr_gcn(self.medication_adjacency_matrix)
        z_ddi = self.ddi_gcn(self.ddi_adjacency_matrix)

        memory_bank = z_ehr - self.beta * z_ddi  # (n_medications, query_dim)

        # Memory Bank read
        a_c = torch.softmax(q_t @ memory_bank.T, dim=-1)  # (batch, n_medications)
        o_b = a_c @ memory_bank                           # (batch, query_dim)

        # Dynamic Memory read
        o_d = o_b.clone()
        for b, length in enumerate(lengths):
            if length <= 1:
                continue
            keys = queries[b, :length - 1, :]
            values = medication_history[b, :length - 1, :]
            a_s = torch.softmax(keys @ q_t[b], dim=0)
            a_m = values.T @ a_s
            o_d[b] = a_m @ memory_bank                      # (query_dim,)

        response = torch.cat([q_t, o_b, o_d], dim=-1)

        logits = self.output_layer(response)

        return logits