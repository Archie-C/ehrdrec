import torch
import math
import torch.nn as nn
import numpy as np

from torch.nn.parameter import Parameter
import torch.nn.functional as F

import einops
from einops import rearrange


def _adjacency_to_numpy(adjacency):
    if isinstance(adjacency, torch.Tensor):
        return adjacency.detach().cpu().float().numpy()
    return np.asarray(adjacency, dtype=np.float32)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(
        self, 
        voc_size, 
        emb_dim, 
        adj, 
        device=torch.device('cpu:0')
    ):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        # x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=1000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.attention = MultiheadAttention(hidden_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x

        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(residual + x)

        residual = x

        x = self.fc(x)
        x = self.dropout2(x)
        x = self.norm2(residual + x)

        return x

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        x = torch.matmul(attention_weights, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        x = self.fc(x)

        return x

class Fastformer(nn.Module):
    def __init__(self, dim = 3, decode_dim = 16):
        super(Fastformer, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias = False)
        self.weight_q = nn.Linear(dim, decode_dim, bias = False)
        self.weight_k = nn.Linear(dim, decode_dim, bias = False)
        self.weight_v = nn.Linear(dim, decode_dim, bias = False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias = False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask = None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b n ()')

        # Caculate the global query
        alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim = 1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy = n)
        p = repeat_global_query * key
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim = 1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)

        # key-value
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
        ehr_adjacency_matrix: torch.Tensor | None = None,
        ddi_adjacency_matrix: torch.Tensor | None = None,
        medication_adjacency_matrix: torch.Tensor | None = None,
        embedding_dim: int = 128,
        embedding_dim_fastformer: int = 128,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(FastRx, self).__init__()
        if ehr_adjacency_matrix is None:
            ehr_adjacency_matrix = medication_adjacency_matrix
        if ehr_adjacency_matrix is None:
            raise ValueError("FastRx requires an EHR medication adjacency matrix.")
        if ddi_adjacency_matrix is None:
            raise ValueError("FastRx requires a DDI adjacency matrix.")

        ehr_adjacency_matrix = _adjacency_to_numpy(ehr_adjacency_matrix)
        ddi_adjacency_matrix = _adjacency_to_numpy(ddi_adjacency_matrix)

        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.ehr_adjacency_matrix = ehr_adjacency_matrix
        self.ddi_adjacency_matrix = ddi_adjacency_matrix
        self.embedding_dim = embedding_dim
        self.embedding_dim_fastformer = embedding_dim_fastformer
        self.dropout = dropout
        self.device = device
        
        self.fastformer = Fastformer(dim = 2 * self.embedding_dim_fastformer, decode_dim = self.embedding_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.ehr_gcn = GCN(
            voc_size=self.n_medications,
            adj=self.ehr_adjacency_matrix,
            emb_dim=self.embedding_dim,
            device=self.device
        )
        self.ddi_gcn = GCN(
            voc_size=self.n_medications,
            adj=self.ddi_adjacency_matrix,
            emb_dim=self.embedding_dim,
            device=self.device
        )
        self.register_buffer("tensor_ddi_adj", torch.as_tensor(self.ddi_adjacency_matrix, dtype=torch.float32))
        
        self.inter = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.inter, 0.5)
        self.embedding = nn.Embedding(self.n_diagnoses + self.n_procedures, self.embedding_dim_fastformer)
        
        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.output = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.n_medications),
        )
    
    def forward(
        self, 
        features
    ):
        diag_seq = []
        proc_seq = []

        def to_long_tensor(x):
            return torch.as_tensor(x, dtype=torch.long, device=self.device)

        def mean_code_embedding(code_ids, *, offset: int = 0):
            code_ids = to_long_tensor(code_ids)
            if code_ids.numel() == 0:
                return torch.zeros(1, 1, self.embedding_dim_fastformer, device=self.device)
            emb = self.embedding(code_ids.add(offset).unsqueeze(dim=0))
            emb = self.dropout_layer(emb)
            return emb.mean(dim=1).unsqueeze(dim=0)  # (1, 1, dim)

        diagnoses_input = features["diagnoses"]
        procedures_input = features["procedures"]
        medications_input = features["medications"]
        
        for diag_codes in diagnoses_input:
            diag_emb = mean_code_embedding(diag_codes)
            diag_seq.append(diag_emb)

        for proc_codes in procedures_input:
            proc_emb = mean_code_embedding(proc_codes, offset=self.n_diagnoses)
            proc_seq.append(proc_emb)

        # (1, seq_len, embedding_dim)
        diagnoses = torch.cat(diag_seq, dim=1)
        procedures = torch.cat(proc_seq, dim=1)

        # CNN over visit sequence
        diagnoses = self.cnn1d(diagnoses.permute(1, 0, 2))
        procedures = self.cnn1d(procedures.permute(1, 0, 2))

        diagnoses = diagnoses.permute(1, 0, 2)
        procedures = procedures.permute(1, 0, 2)

        # (1, seq_len, embedding_dim * 2)
        patient_representation = torch.cat((diagnoses, procedures), dim=-1)

        seq_len = patient_representation.size(1)
        mask = torch.ones(1, seq_len, dtype=torch.bool, device=self.device)

        feat = self.fastformer(patient_representation, mask).squeeze(dim=0)

        # current visit representation
        query = feat[-1:]  # (1, dim)

        # graph memory module
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter

        # dynamic patient history memory
        if len(diagnoses_input) > 1:
            history_keys = feat[:-1]

            history_values = torch.zeros(
                len(diagnoses_input) - 1,
                self.n_medications,
                device=self.device
            )

            for idx, meds in enumerate(medications_input[:-1]):
                meds = to_long_tensor(meds)
                history_values[idx, meds] = 1.0

        # read from global memory bank
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)
        fact1 = torch.mm(key_weights1, drug_memory)

        # read from dynamic patient history memory
        if len(diagnoses_input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=-1)
            weighted_values = torch.mm(visit_weight, history_values)
            fact2 = torch.mm(weighted_values, drug_memory)
        else:
            fact2 = fact1

        # final prediction
        result = self.output(torch.cat([query, fact1, fact2], dim=-1))

        # DDI loss
        if self.training:
            neg_pred_prob = torch.sigmoid(result)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

            return {
                "predictions": result,
                "losses": {
                    "ddi_loss": batch_neg,
                }
            }
        else:
            return {
                "predictions": result,
            }
