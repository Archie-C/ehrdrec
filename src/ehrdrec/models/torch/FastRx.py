import torch
import torch.nn as nn

from einops import rearrange, repeat
from ehrdrec.models.utils import GCN

class FastFormer(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        decode_dim: int = 16
    ) -> None:
        super(FastFormer, self).__init__()
        
        self.to_qkv     = nn.Linear(dim, decode_dim * 3, bias=False)
        self.weight_q   = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k   = nn.Linear(dim, decode_dim, bias=False)
        self.weight_v   = nn.Linear(dim, decode_dim, bias=False)
        self.weight_r   = nn.Linear(decode_dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5
    
    def forward(self, x, mask=None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape
        
        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b n ()')
        
        alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim=-1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)
        
        repeat_global_query = repeat(global_query, 'b d -> b copy d', copy=n)
        p = repeat_global_query * key
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim=-1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)
        
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result

class FastRx(nn.Module):
    def __init__(
        self,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        medication_adjacency_matrix: torch.Tensor,
        ddi_adjacency_matrix: torch.Tensor,
        embedding_dim: int = 256,
        embedding_dim_fastformer: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super(FastRx, self).__init__()
        
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        
        self.register_buffer(
            "medication_adjacency_matrix",
            medication_adjacency_matrix.float(),
        )
        self.register_buffer(
            "ddi_adjacency_matrix",
            ddi_adjacency_matrix.float(),
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.fastformer = FastFormer(dim=embedding_dim_fastformer * 2, decode_dim=embedding_dim)
        
        self.ehr_gcn = GCN(n_nodes=n_medications, embed_dim=embedding_dim, dropout=dropout)
        self.ddi_gcn = GCN(n_nodes=n_medications, embed_dim=embedding_dim, dropout=dropout)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        
        self.diag_proj = nn.Linear(n_diagnoses, embedding_dim_fastformer)
        self.proc_proj = nn.Linear(n_procedures, embedding_dim_fastformer)
        
        self.cnn1d = nn.Sequential(
            nn.Conv1d(embedding_dim_fastformer, embedding_dim_fastformer, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, n_medications)
        )
    
    def forward(self, x):
        lengths = x["lengths"].to(x["diagnoses"].device)
        batch_size, seq_len, _ = x["diagnoses"].shape
        
        # --- patient health representation ---
        diagnosis_embeddings = self.dropout(self.diag_proj(x["diagnoses"]))    # (batch, seq, emb_dim_ff)
        procedure_embeddings = self.dropout(self.proc_proj(x["procedures"]))   # (batch, seq, emb_dim_ff)
        
        # CNN expects (batch, channels, seq)
        diagnosis_embeddings = self.cnn1d(diagnosis_embeddings.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, seq, emb_dim_ff)
        procedure_embeddings = self.cnn1d(procedure_embeddings.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, seq, emb_dim_ff)

        combined_visit_embeddings = torch.cat([diagnosis_embeddings, procedure_embeddings], dim=-1)  # (batch, seq, emb_dim_ff*2)
        
        # True for real visits, False for padding
        visit_padding_mask = torch.arange(seq_len, device=x["diagnoses"].device).unsqueeze(0) < lengths.unsqueeze(1)  # (batch, seq)

        contextual_visit_features = self.fastformer(combined_visit_embeddings, visit_padding_mask)  # (batch, seq, emb_dim)
        
        # --- extract current visit query (last real visit per patient) ---
        current_visit_idx = (lengths - 1).clamp(min=0)                                                        # (batch,)
        current_visit_query = contextual_visit_features[torch.arange(batch_size), current_visit_idx]           # (batch, emb_dim)
        
        # --- graph memory: drug knowledge base ---
        drug_knowledge_memory = self.ehr_gcn(self.medication_adjacency_matrix) - self.ddi_gcn(self.ddi_adjacency_matrix) * self.inter                                  # (drug_vocab, emb_dim)

        global_drug_attention = torch.softmax(torch.mm(current_visit_query, drug_knowledge_memory.t()), dim=-1)   # (batch, drug_vocab)
        global_drug_context = torch.mm(global_drug_attention, drug_knowledge_memory)                           # (batch, emb_dim)
        
        
        # --- history fact: attend over previous visits ---
        patient_has_history = lengths > 1                                                                      # (batch,)

        if patient_has_history.any():
            raw_history_attention_scores = torch.bmm(
                current_visit_query.unsqueeze(1),                    # (batch, 1, emb_dim)
                contextual_visit_features.permute(0, 2, 1)          # (batch, emb_dim, seq)
            ).squeeze(1)                                             # (batch, seq)

            # mask out padding and the current visit itself
            history_visit_mask = torch.arange(seq_len, device=x["diagnoses"].device).unsqueeze(0) < (lengths - 1).unsqueeze(1)  # (batch, seq)
            masked_history_attention = raw_history_attention_scores.masked_fill(~history_visit_mask, float('-inf'))
            history_visit_attention = torch.softmax(masked_history_attention, dim=-1)
            history_visit_attention = torch.nan_to_num(history_visit_attention)                                # handle patients with no history

            history_weighted_medications = torch.bmm(
                history_visit_attention.unsqueeze(1),                # (batch, 1, seq)
                x["medication_history"]                              # (batch, seq, drug_vocab)
            ).squeeze(1)                                             # (batch, drug_vocab)

            history_drug_context = torch.mm(history_weighted_medications, drug_knowledge_memory)               # (batch, emb_dim)
            history_drug_context = torch.where(
                patient_has_history.unsqueeze(1),
                history_drug_context,
                global_drug_context,                                 # fall back to global context for first visits
            )
        else:
            history_drug_context = global_drug_context

        # --- predict medication recommendations ---
        combined_context = torch.cat([current_visit_query, global_drug_context, history_drug_context], dim=-1) # (batch, emb_dim*3)
        medication_logits = self.output(combined_context)
        
        return {"predictions": medication_logits}