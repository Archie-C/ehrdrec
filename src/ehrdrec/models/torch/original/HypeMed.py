import torch
import torch.nn as nn
import faiss
import torch.nn.functional as F
import polars as pl
import os
import torch.sparse as tsp
from typing import Optional
from pathlib import Path

from torch.optim import AdamW
from tqdm.auto import trange

from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax

from info_nce import InfoNCE

from ehrdrec.utils.constants import ReservedId

class EHRMemoryAttention(nn.Module):
    """
    A simple attention + FFN block using a transformer-style architecture.
    """
    def __init__(self, embedding_dim, n_heads, dropout, top_n=10, act=nn.LeakyReLU):
        super(EHRMemoryAttention, self).__init__()
        self.visit_mem_attn = nn.MultiheadAttention(
            # embed_dim=embedding_dim * 3,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        d_model = embedding_dim
        dim_feedforward = embedding_dim
        self.top_n = top_n


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = act()

        self.res = None

    
    def neighbour_search(self, visit_rep, E_mem_patient_rep, k=10):
        d = visit_rep.shape[1]
        k = min(k, E_mem_patient_rep.shape[0])
        index = faiss.IndexFlatL2(d)
        memory = E_mem_patient_rep.detach().float().cpu().numpy()
        queries = visit_rep.detach().float().cpu().numpy()
        index.add(memory)
        distances, indices = index.search(queries, k)
        indices = torch.as_tensor(indices, dtype=torch.long, device=visit_rep.device)
        return distances, indices

    def forward(self, visit_rep, E_mem_patient_rep, E_mem_med_rep):
        """

        Args:
            visit_rep: Representation of the current visit.
            E_mem: Cluster centers obtained from EHR hyperedges, representing typical cases.

        Returns:

        """
        x = visit_rep.unsqueeze(1)  # Adjust x to a matching 3D tensor.
        k = E_mem_patient_rep
        v = E_mem_med_rep
        D, I = self.neighbour_search(visit_rep, E_mem_patient_rep, k=self.top_n)
        k = E_mem_patient_rep[I, :]
        v = E_mem_med_rep[I, :]
        
        # print(x.shape, k.shape, v.shape)
        
        x = self.norm1(x + self._att_block(x, k, v, attn_mask=None))
        x = self.norm2(x + self._ff_block(x))
        # x = self._att_block(x, k, v)
        # print(x.shape)
        return x.squeeze(1)  # Restore x to its original 2D shape.


    def _att_block(self, q, k, v, attn_mask=None):
        x, attn = self.visit_mem_attn(q, k, v,
                           need_weights=True, attn_mask=attn_mask)
        # x = x.squeeze(1)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class HistoryAttention(nn.Module):
    """
    A simple attention + FFN block using a transformer-style architecture.
    """
    def __init__(self, embedding_dim, n_heads, dropout, act=nn.LeakyReLU):
        super(HistoryAttention, self).__init__()
        self.visit_mem_attn = nn.MultiheadAttention(
            # embed_dim=embedding_dim * 3,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Implementation of Feedforward model
        # d_model = embedding_dim * 3
        # dim_feedforward = embedding_dim * 3
        d_model = embedding_dim
        dim_feedforward = embedding_dim
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = act()

    def forward(self, x, attn_mask):
        """

        Args:
            visit_rep: Representation of the current visit.
            E_mem: Cluster centers obtained from EHR hyperedges, representing typical cases.

        Returns:

        """
        x = self.norm1(x + self._att_block(x, x, x, attn_mask))
        x = self.norm2(x + self._ff_block(x))
        # x = self._att_block(x, k, v)

        return x

    # self-attention block
    def _att_block(self, x, k, v, attn_mask):
        x, attn = self.visit_mem_attn(x, k, v, attn_mask=attn_mask,
                           need_weights=True)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class FeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, dropout, act):
        super(FeedForwardLayer, self).__init__()
        # Feed Forward block.
        self.ff_linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ff_linear2 = nn.Linear(embed_dim * 2, embed_dim)
        self.act_fn_ff = act()
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def reset_parameters(self):
        self.ff_linear1.reset_parameters()
        self.ff_linear2.reset_parameters()

    def forward(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

class Node2EdgeAggregator(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, act=nn.LeakyReLU):
        super(Node2EdgeAggregator, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self._ff_block = FeedForwardLayer(embed_dim, dropout, act)
    def forward(self, x):
        """

        Args:
            x: bsz, max_size, dim

        Returns:

        """
        # First compute the hyperedge representation by averaging, then concatenate the node and hyperedge representations to compute attention.
        bsz, max_size, dim = x.shape
        hyperedge_attr = x.mean(1, keepdim=True)   # bsz, 1, dim
        hyperedge_attr = self.norm1(self._sa_block(hyperedge_attr, x, x) + hyperedge_attr)
        out = self.norm2(self._ff_block(hyperedge_attr) + hyperedge_attr)
        return out

    def _sa_block(self, q, k, v):
        attn_visit, attn = self.mha(
            query=q,
            key=k,
            value=v,
            need_weights=True
        )
        return attn_visit

class HypeMed(nn.Module):
    def __init__(
        self,
        n_diagnoses: int, 
        n_procedures: int,
        n_medications: int,
        embedding_dim: int, 
        number_of_heads: int,
        number_of_ehr_edges: int,
        top_n: int,
        device: torch.device,
        X_hat,
        E_mem,
        ddi_adjacency_matrix: torch.Tensor,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super(HypeMed, self).__init__()
        
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.embedding_dim = embedding_dim
        self.number_of_heads = number_of_heads
        self.number_of_ehr_edges = number_of_ehr_edges
        self.n_ehr_edges = number_of_ehr_edges
        self.name_lst = ["diagnoses", "procedures", "medications"]
        self.dropout = dropout
        self.device = device
        
        self.activation = nn.Silu if activation == "silu" else nn.LeakyReLU
        
        self.X_hat = nn.ModuleDict({
            "diagnoses": nn.Embedding(n_diagnoses, embedding_dim).from_pretrained(X_hat["diagnoses"], freeze=False, padding_idx=ReservedId.PAD),
            "procedures": nn.Embedding(n_procedures, embedding_dim).from_pretrained(X_hat["procedures"], freeze=False, padding_idx=ReservedId.PAD),
            "medications": nn.Embedding(n_medications, embedding_dim).from_pretrained(X_hat["medications"], freeze=False, padding_idx=ReservedId.PAD)
        })
        
        self.E_mem = nn.ModuleDict({
            "diagnoses": nn.Embedding(n_diagnoses, embedding_dim).from_pretrained(E_mem["diagnoses"], freeze=False),
            "procedures": nn.Embedding(n_procedures, embedding_dim).from_pretrained(E_mem["procedures"], freeze=False),
            "medications": nn.Embedding(n_medications, embedding_dim).from_pretrained(E_mem["medications"], freeze=False)
        })
        
        self.ddi_adjacency_matrix = ddi_adjacency_matrix.to(self.device)
        self.tensor_ddi_adj = self.ddi_adjacency_matrix
        self.embedding_norm = nn.ModuleDict({
            "diagnoses": nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Dropout(dropout)
            ),
            "procedures": nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Dropout(dropout)
            ),
            "medications": nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Dropout(dropout)
            )
        })
        
        self.node2edge_agg = nn.ModuleDict({
            "diagnoses": Node2EdgeAggregator(embedding_dim, number_of_heads, dropout, self.activation),
            "procedures": Node2EdgeAggregator(embedding_dim, number_of_heads, dropout, self.activation),
            "medications": Node2EdgeAggregator(embedding_dim, number_of_heads, dropout, self.activation)
        })
        
        self.none_mlp = nn.Sequential(
            nn.LayerNorm(3 * embedding_dim),
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            self.activation(),
        )
        
        self.patient_projection = nn.Linear(embedding_dim, embedding_dim)
        self.medication_projection = nn.Linear(embedding_dim, embedding_dim)
        self.proj_patient = self.patient_projection
        self.proj_med = self.medication_projection
        
        self.patient_level_dp_attention = HistoryAttention(embedding_dim, number_of_heads, dropout, self.activation)
        self.patient_level_mh_attention = HistoryAttention(embedding_dim, number_of_heads, dropout, self.activation)
        self.patient_level_dp_attn = self.patient_level_dp_attention
        self.patient_level_mh_attn = self.patient_level_mh_attention
        
        self.ehr_level_attention = EHRMemoryAttention(embedding_dim, number_of_heads, dropout, top_n=top_n, act=self.activation)
        self.memory_context_attention = HistoryAttention(embedding_dim, number_of_heads, dropout, self.activation)
        self.ehr_level_attn = self.ehr_level_attention
        self.mem_context_attn = self.memory_context_attention
        
        self.fusion_prediction_normalisation = nn.LayerNorm(self.n_medications)
        self.fusion_pred_norm = self.fusion_prediction_normalisation
        self.pred_bias = nn.Parameter(torch.zeros(self.n_medications), requires_grad=True)
        
        self.cat_ln = nn.LayerNorm(2 * embedding_dim)
        self.gate_control = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2),
            nn.Dropout(dropout),
        )
        
        self.info_nce_loss = InfoNCE(reduction='mean')
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def node2edge(self, entity_seq_embed):
        """

        Args:
            entity_seq_embed: (bsz, max_vist, max_size, dim)
            records: (bsz, max_vist, max_size)

        Returns:

        """
        visit_seq_embed = {}
        for n in self.name_lst:
            # Flatten the data first: bsz, max_vist, max_size, dim.
            seq_embed = entity_seq_embed[n]
            bsz, max_vist, max_size, dim = seq_embed.shape
            seq_embed = seq_embed.reshape(bsz * max_vist, max_size, dim)
            visit_seq_embed[n] = self.node2edge_agg[n](seq_embed).reshape(bsz, max_vist, dim)
        return visit_seq_embed

    
    def orthogonality_loss_cosine(self, emb1, emb2):
        # Compute the cosine similarity between two embedding vectors.
        cosine_similarity = F.cosine_similarity(emb1, emb2, dim=1)
        # Since we want the vectors to be orthogonal (cosine similarity of 0), directly compute the squared cosine similarity.
        loss = torch.mean(cosine_similarity ** 2)
        return loss

    def forward(self, features):
        if "records" in features:
            records = features["records"]
            masks = features["masks"]
            true_visit_idx = features["true_visit_idx"]
            visit2edge_idx = features["visit2edge_idx"]
        else:
            records = {
                "diagnoses": features["diagnoses"],
                "procedures": features["procedures"],
                "medications": features["medications"],
            }
            masks = {"attn_mask": features["attn_mask"]}
            true_visit_idx = features["true_visit_idx"]
            visit2edge_idx = features["visit2edge_idx"]
        
        assert len(visit2edge_idx) == true_visit_idx.sum().item()
        X_hat = self.X_hat
        E_mem = {
            'diagnoses': self.E_mem['diagnoses'](torch.arange(self.n_ehr_edges).to(self.device)),
            'procedures': self.E_mem['procedures'](torch.arange(self.n_ehr_edges).to(self.device)),
            'medications': self.E_mem['medications'](torch.arange(self.n_ehr_edges).to(self.device))
        }

        # Parse sequence data.
        entity_seq_embed = {}  # (bsz, max_vist, max_size, dim)
        for n in self.name_lst:
            entity_seq_embed[n] = self.embedding_norm[n](X_hat[n](records[n]))

        # Start with visit-level data representations.
        # Use multi-head attention.
        visit_seq_embed = self.node2edge(entity_seq_embed)  # bsz, max_visit, dim

        # Pull medications out separately as medication history.
        med_history = visit_seq_embed['medications']  # Intended as a pure medication-history decoder.
        # Note that medications at the final time step are not visible and can only be used as supervision.
        batch_size, max_visit, dim = med_history.shape
        pad_head_med_history = torch.zeros(batch_size, 1, dim, dtype=med_history.dtype, device=med_history.device)
        med_history = torch.cat([pad_head_med_history, med_history], dim=1)[:, :-1, :]  # Shifted here.


        # Context first, then memory.
        # Concatenate diagnoses, procedures, and historical medications here.
        # Add causal relationships:
        # diag ---> proc     last_med
        #  |          |          |
        #  |--->med<--|----------|
        # visit_rep = visit_seq_embed['diagnoses'] + visit_seq_embed['procedures']
        diag_rep = visit_seq_embed['diagnoses']
        proc_rep = visit_seq_embed['procedures']
        dp_rep = visit_seq_embed['diagnoses'] + visit_seq_embed['procedures']
        dp_rep = dp_rep.reshape(batch_size * max_visit, dim)
        dp_rep = dp_rep[true_visit_idx]
        # Compute representations that include contextual information.
        # attn_mask = masks['attn_mask'].repeat(self.n_heads, 1, 1)
        attn_mask = masks['attn_mask']
        attn_mask = attn_mask[true_visit_idx][:, true_visit_idx]
        assert attn_mask.shape == (dp_rep.shape[0], dp_rep.shape[0])
        patient_level_dp_rep = self.patient_level_dp_attn(dp_rep, attn_mask)
        # patient_level_dp_rep = dp_rep

        mh_rep = med_history
        mh_rep = mh_rep.reshape(batch_size * max_visit, dim)
        mh_rep = mh_rep[true_visit_idx]
        patient_level_mh_rep = self.patient_level_mh_attn(mh_rep, attn_mask)
        # patient_level_mh_rep = mh_rep
        patient_level_rep = patient_level_dp_rep + patient_level_mh_rep
        #
        # med_history = med_history.reshape(batch_size * max_visit, dim)
        # med_history = med_history[true_visit_idx]  # Keep only non-empty visits.

        # EHR-level
        currect_case_rep = visit_seq_embed['diagnoses'] + visit_seq_embed['procedures']
        currect_case_rep = currect_case_rep.reshape(batch_size * max_visit, dim)
        currect_case_rep = currect_case_rep[true_visit_idx]  # Keep only non-empty visits.

        E_mem_case_rep = E_mem['diagnoses'] + E_mem['procedures']
        E_mem_med_rep = E_mem['medications']
        ehr_level_rep = self.ehr_level_attn(
            currect_case_rep, E_mem_case_rep, E_mem_med_rep
        )
        ehr_level_rep = self.mem_context_attn(ehr_level_rep, attn_mask)

        if self.training:
            med_rep = visit_seq_embed['medications'].reshape(batch_size * max_visit, dim)[true_visit_idx]

            # Alignment is done here during training, so the tensors need to be aligned here.
            patient_edge_in_batch = E_mem_case_rep[visit2edge_idx]
            m_edge_in_batch = E_mem_med_rep[visit2edge_idx]
            proj_patient = self.proj_patient(currect_case_rep)
            proj_patient_edge = self.proj_patient(patient_edge_in_batch)
            proj_med = self.proj_med(med_rep)
            proj_med_edge = self.proj_med(m_edge_in_batch)
            dp_ssl_loss = self.info_nce_loss(proj_patient, proj_patient_edge)
            m_ssl_loss = self.info_nce_loss(proj_med, proj_med_edge)
            ssl_loss = dp_ssl_loss + m_ssl_loss

        cat_rep = torch.cat([
            patient_level_rep,
            ehr_level_rep],
            -1)
        cat_rep = self.cat_ln(cat_rep)
        patient_level_rep, ehr_level_rep = torch.split(cat_rep, dim, dim=-1)

        cat_rep = torch.cat([
            patient_level_rep.unsqueeze(-1),
            ehr_level_rep.unsqueeze(-1)],
            -1)

        gate = self.gate_control(cat_rep.reshape(-1, 2 * patient_level_rep.shape[-1])).reshape(-1, 1, 2)
        assert len(gate.shape) == 3 and gate.shape[-1] == 2
        gate = torch.softmax(gate, -1)

        fusion_rep = (gate * cat_rep).sum(-1)

        med_rep = X_hat['medications'](torch.arange(self.n_medications, dtype=torch.long, device=self.device))

        fusion_output = torch.matmul(fusion_rep, med_rep.T) + self.pred_bias
        output = self.fusion_pred_norm(fusion_output)

        if self.training:
            orthogonality_loss = self.orthogonality_loss_cosine(patient_level_rep, ehr_level_rep)
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.unsqueeze(-1)
            neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (true visit num, voc_size, voc_size)

            # loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).unsqueeze(-1)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

            return {
                "predictions": output,
                "losses": {
                    "ddi_loss": batch_neg,
                    "ssl_loss": ssl_loss,
                    "orthogonality_loss": orthogonality_loss,
                    "gate": gate
                }
            }

        return {
            "predictions": output
        }


def construct_graphs(
    train_dataframe: pl.DataFrame | pl.LazyFrame,
    n_diagnoses: int,
    n_procedures: int,
    n_medications: int,
    *,
    diagnosis_col: str = "diagnosis_ids",
    procedure_col: str = "procedure_ids",
    medication_col: str = "atc_ids",
    medication_multihot_col: str = "medication_multihot",
) -> dict[str, torch.Tensor]:
    """
    Construct diagnosis, procedure, and medication hypergraphs from a processed
    training frame. Each visit is one hyperedge; code IDs in that visit connect
    to the visit's hyperedge column.

    The standard multihot preprocessor keeps diagnoses/procedures as sparse ID
    lists and may keep medications either as ``atc_ids`` or only as a dense
    ``medication_multihot`` vector. Both medication formats are supported.
    """
    if isinstance(train_dataframe, pl.LazyFrame):
        train_dataframe = train_dataframe.collect()

    num_nodes = {
        "diagnoses": n_diagnoses,
        "procedures": n_procedures,
        "medications": n_medications,
    }
    columns = {
        "diagnoses": diagnosis_col,
        "procedures": procedure_col,
        "medications": medication_col,
    }

    available_columns = set(train_dataframe.columns)
    missing_columns = [
        column
        for name, column in columns.items()
        if name != "medications" and column not in available_columns
    ]
    if missing_columns:
        raise ValueError(
            "train_dataframe is missing required encoded code columns: "
            + ", ".join(missing_columns)
        )

    medication_has_ids = medication_col in available_columns
    medication_has_multihot = medication_multihot_col in available_columns
    if not medication_has_ids and not medication_has_multihot:
        raise ValueError(
            "train_dataframe must include either "
            f"{medication_col!r} or {medication_multihot_col!r}."
        )

    coo_entries = {name: [] for name in num_nodes}
    selected_columns = [diagnosis_col, procedure_col]
    selected_columns.append(medication_col if medication_has_ids else medication_multihot_col)

    for visit_num, row in enumerate(train_dataframe.select(selected_columns).iter_rows(named=True)):
        visit_codes = {
            "diagnoses": row[diagnosis_col],
            "procedures": row[procedure_col],
        }
        if medication_has_ids:
            visit_codes["medications"] = row[medication_col]
        else:
            visit_codes["medications"] = [
                idx for idx, value in enumerate(row[medication_multihot_col]) if value
            ]

        for name, codes in visit_codes.items():
            if codes is None:
                continue
            for code in codes:
                if code is None:
                    continue
                code = int(code)
                if code == ReservedId.PAD:
                    continue
                if code < 0 or code >= num_nodes[name]:
                    raise ValueError(
                        f"{name} code id {code} is outside the valid range "
                        f"[0, {num_nodes[name]})."
                    )
                coo_entries[name].append((code, visit_num))

    visit_count = train_dataframe.height
    hypergraphs = {}
    for name, entries in coo_entries.items():
        if entries:
            indices = torch.tensor(entries, dtype=torch.long).T.contiguous()
            values = torch.ones(indices.shape[1], dtype=torch.float32)
        else:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)

        hypergraphs[name] = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(num_nodes[name], visit_count),
        ).coalesce()

    return hypergraphs


def drop_incidence(hyperedge_index: Tensor, drop_rate: float) -> Tensor:
    if drop_rate <= 0 or hyperedge_index.numel() == 0:
        return hyperedge_index
    keep_mask = torch.rand(hyperedge_index.size(1), device=hyperedge_index.device) >= drop_rate
    if not bool(keep_mask.any()):
        keep_mask[torch.randint(0, keep_mask.numel(), (1,), device=keep_mask.device)] = True
    return hyperedge_index[:, keep_mask]


def drop_features(features: Tensor, drop_rate: float) -> Tensor:
    if drop_rate <= 0:
        return features
    keep_mask = torch.rand_like(features) >= drop_rate
    return features * keep_mask.to(features.dtype)


def valid_node_edge_mask(
    hyperedge_index: Tensor,
    num_nodes: int,
    num_edges: int,
) -> tuple[Tensor, Tensor]:
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=hyperedge_index.device)
    edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=hyperedge_index.device)
    if hyperedge_index.numel() > 0:
        node_mask[hyperedge_index[0]] = True
        edge_mask[hyperedge_index[1]] = True
    return node_mask, edge_mask


def hyperedge_index_masking(
    hyperedge_index: Tensor,
    num_nodes: int,
    num_edges: int,
    node_mask: Tensor | None = None,
    edge_mask: Tensor | None = None,
) -> Tensor:
    keep_mask = torch.ones(hyperedge_index.size(1), dtype=torch.bool, device=hyperedge_index.device)
    if node_mask is not None:
        keep_mask &= node_mask[hyperedge_index[0]]
    if edge_mask is not None:
        keep_mask &= edge_mask[hyperedge_index[1]]

    masked_index = hyperedge_index[:, keep_mask].clone()
    if edge_mask is not None and masked_index.numel() > 0:
        edge_remap = torch.full((num_edges,), -1, dtype=torch.long, device=hyperedge_index.device)
        edge_remap[edge_mask] = torch.arange(int(edge_mask.sum()), device=hyperedge_index.device)
        masked_index[1] = edge_remap[masked_index[1]]
    return masked_index


class FeatureEncoder(nn.Module):
    def __init__(self, H, idx2word, se_dim, pe_dim, ke_dim, cache_dir, device, name):
        super().__init__()
        num_nodes = H.shape[0]
        self.pe = nn.Parameter(torch.empty(num_nodes, pe_dim, device=device))
        self.se = nn.Parameter(torch.empty(num_nodes, se_dim, device=device))
        self.ke = nn.Parameter(torch.empty(num_nodes, ke_dim, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.pe)
        nn.init.xavier_uniform_(self.se)
        nn.init.xavier_uniform_(self.ke)

    def forward(self):
        return {"pe": self.pe, "se": self.se, "ke": self.ke}, None

# -----------------------------
# Pretraining embeddings
# -----------------------------

class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
          hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
          hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Hyperedge feature matrix
                in :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0


        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))

        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias


        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

class MPNN(nn.Module):
    """
    local MPNN part
    """

    def __init__(self, embed_dim, n_heads, dropout):
        super(MPNN, self).__init__()
        self.conv = HypergraphConv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            use_attention=True,
            heads=n_heads, dropout=dropout,
            concat=False
        )
        # 超边特征计算,对所有节点做线性变换然后求和,加上对超边特征变换求和
        # 原始GatedGCN中分别对头尾节点使用不同的变换,这里统一成相同的
        self.node_ffn = nn.Linear(embed_dim, embed_dim)
        self.edge_ffn = nn.Linear(embed_dim, embed_dim)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.node_ffn.reset_parameters()
        self.edge_ffn.reset_parameters()

    def compute_edge_feat(self, X, E, H):
        """
        获取超边的表示，通过聚合当前超边下所有item的embedding
        实际上就是乘以H(n_edges, n_items)
        Args:
            X: 节点特征
            E: 边特征
            H:

        Returns:

        """

        # embed: n_items, dim
        n_items, n_edges = H.shape
        if H.is_sparse:
            norm_factor = (tsp.sum(H, dim=0) ** -1).to_dense().reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            X_trans = self.node_ffn(X)
            # E计算:变换后的节点聚合,原始E变换,原始E
            agg_edge_feat = norm_factor * tsp.mm(H.T, X_trans)
            E_res = agg_edge_feat + self.edge_ffn(E) + E
        else:
            norm_factor = (torch.sum(H, dim=0) ** -1).reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            X_trans = self.node_ffn(X)
            # E计算:变换后的节点聚合,原始E变换,原始E
            agg_edge_feat = norm_factor * tsp.mm(H.T, X_trans)
            E_res = agg_edge_feat + self.edge_ffn(E) + E

        return E_res

    def forward(self, X, E, H, edge_weight):
        adj_index = H.indices()
        E_res = self.compute_edge_feat(X, E, H)
        X_res = self.conv(X, adj_index, hyperedge_attr=E_res, hyperedge_weight=edge_weight)
        return X_res, E_res


class GlobalAttention(nn.Module):
    """
    global attention part
    """

    def __init__(self, embed_dim, n_heads, dropout):
        super(GlobalAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def reset_parameters(self):
        self.self_attn._reset_parameters()

    def forward(self, X, ke_bias=None):
        X_attn = self._sa_block(X, ke_bias, None)  # 这里attn_mask如果是float类型,可以直接加到attn_weights上面
        return X_attn

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        # Requires PyTorch v1.11+ to support `average_attn_weights=False`
        # option to return attention weights of individual heads.
        x, A = self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=True,
                              average_attn_weights=False)
        # self.attn_weights = A.detach().cpu()
        return x

class HGTEncoderLayer(nn.Module):
    """
    参考GraphGPS,变成超图
    """

    def __init__(self, embed_dim, n_heads, dropout, act=nn.LeakyReLU):
        super(HGTEncoderLayer, self).__init__()
        self.MPNN_layer = MPNN(embed_dim, n_heads, dropout)
        self.global_att = GlobalAttention(embed_dim, n_heads, dropout)

        self.local_ln = nn.LayerNorm(embed_dim)
        self.local_dropout = nn.Dropout(dropout)
        self.global_ln = nn.LayerNorm(embed_dim)
        self.global_dropout = nn.Dropout(dropout)

        self.node_ff_block = FeedForwardLayer(embed_dim, dropout, act)
        self.node_norm = nn.LayerNorm(embed_dim)

        self.edge_norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.MPNN_layer.reset_parameters()
        self.global_att.reset_parameters()
        self.local_ln.reset_parameters()
        self.global_ln.reset_parameters()
        self.node_ff_block.reset_parameters()
        self.node_norm.reset_parameters()
        self.edge_norm.reset_parameters()

    def forward(self, X, E, H, edge_weight, ke_bias):
        """

        Args:
            X: 节点特征
            E: 边特征
            H: 邻接矩阵
            edge_weight: 超边权重
        Returns:

        """
        X_M_hat, E_res_hat = self.MPNN_layer(X, E, H, edge_weight)
        E_res = self.edge_norm(E_res_hat)
        X_M = self.local_ln(self.local_dropout(X_M_hat) + X)

        X_T_hat = self.global_att(X, ke_bias)
        X_T = self.global_ln(self.global_dropout(X_T_hat) + X)

        X_res = self.node_norm(self.node_ff_block(X_M + X_T) + X_M + X_T)
        return X_res, E_res

class HGTEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, n_layers, H, idx2word, cache_dir, device, name):
        super(HGTEncoder, self).__init__()
        self.n_layers = n_layers
        self.name = name
        cache_dir = os.path.join(cache_dir, name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.feature_encoder = FeatureEncoder(
            H=H,
            idx2word=idx2word,
            se_dim=embed_dim,
            pe_dim=embed_dim,
            ke_dim=embed_dim,
            cache_dir=cache_dir,
            device=device,
            name=name,
        )

        self.encoders = nn.ModuleList()
        for i in range(n_layers):
            self.encoders.append(
                HGTEncoderLayer(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    dropout=dropout
                )
            )

        self.node_norm = nn.LayerNorm(embed_dim)
        self.edge_norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.feature_encoder.reset_parameters()
        for layer in self.encoders:
            layer.reset_parameters()
            self.node_norm.reset_parameters()
            self.edge_norm.reset_parameters()

    def forward(self, X, E, H, edge_weight=None):
        """

        Args:
            X: 节点特征
            E: 边特征
            H: 邻接矩阵
            edge_weight: 超边权重

        Returns:

        """
        side_encodings, ke_bias = self.feature_encoder()
        pe_encoding, se_encoding, ke_encoding = side_encodings['pe'], side_encodings['se'], side_encodings['ke']
        X = X + pe_encoding + se_encoding + ke_encoding
        X_lst = [X]
        E_lst = [E]
        for i in range(self.n_layers):
            layer = self.encoders[i]
            X, E = layer(X, E, H, edge_weight, ke_bias)
            X_lst.append(X)
            E_lst.append(E)
        X = sum(X_lst) / (self.n_layers + 1)
        E = sum(E_lst) / (self.n_layers + 1)

        X = self.node_norm(X)
        E = self.edge_norm(E)

        return X, E

class TriCL(nn.Module):
    def __init__(self, encoder, embedding_dim, proj_dim: int, num_nodes, num_edges, device):
        super(TriCL, self).__init__()
        self.device = device
        self.encoder = encoder

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_dim = embedding_dim
        self.edge_dim = embedding_dim

        self.node_embedding = nn.Embedding(self.num_nodes, self.node_dim)
        self.edge_embedding = nn.Embedding(self.num_edges + self.num_nodes, self.edge_dim)  # 加上自连边

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)

        self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.disc.reset_parameters()

    def get_features(self, adj):
        node_idx = torch.arange(self.num_nodes, device=self.device)
        node_features = self.node_embedding(node_idx)

        edge_idx = torch.arange(self.num_edges + self.num_nodes, device=self.device)
        agg_edge_feat = self.get_hyperedge_representation(node_features, adj)

        edge_features = self.edge_embedding(edge_idx) + torch.cat([agg_edge_feat, node_features], dim=0)
        return node_features, edge_features

    @staticmethod
    def get_hyperedge_representation(embed, adj):
        """
        获取超边的表示，通过聚合当前超边下所有item的embedding
        实际上就是乘以H(n_edges, n_items)
        Args:
            embed:
            adj:

        Returns:

        """

        # embed: n_items, dim
        n_items, n_edges = adj.shape
        if adj.is_sparse:
            norm_factor = (tsp.sum(adj, dim=0) ** -1).to_dense().reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            E = norm_factor * tsp.mm(adj.T, embed)
        else:
            norm_factor = (torch.sum(adj, dim=0) ** -1).reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            E = norm_factor * torch.mm(adj.T, embed)

        return E

    def forward(self, x: Tensor, y: Tensor, hyperedge_index: Tensor):
        """

        Args:
            x: 节点特征
            y: 边特征
            hyperedge_index:

        Returns:

        """
        num_nodes, num_edges = self.num_nodes, self.num_edges

        # if num_nodes is None:
        #     num_nodes = int(hyperedge_index[0].max()) + 1
        # if num_edges is None:
        #     num_edges = int(hyperedge_index[1].max()) + 1

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        H = torch.sparse_coo_tensor(
            indices=self_loop_hyperedge_index,
            values=torch.ones_like(self_loop_hyperedge_index[0, :]),
            size=(num_nodes, num_edges + num_nodes)
        ).coalesce().float()
        n, e = self.encoder(x, y, H)
        return n, e[:num_edges]

    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                         num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)

    def f(self, x, tau):
        return torch.exp(x / tau)

    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))

    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int],
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float,
                        batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                        mean: bool = True):
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float,
                         batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                         mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss

    def membership_level_loss(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                              batch_size: Optional[int] = None, mean: bool = True):
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]
        if batch_size is None:
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)

            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
        else:
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)

            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]

                pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)
                neg_n = self.f(
                    self.disc_similarity(n[hyperedge_index[:, mask][0]], e_perm[hyperedge_index[:, mask][1]]), tau)
                neg_e = self.f(
                    self.disc_similarity(n_perm[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)

                aggr_pos.append(pos)
                aggr_neg_n.append(neg_n)
                aggr_neg_e.append(neg_e)
            aggr_pos = torch.concat(aggr_pos)
            aggr_neg_n = torch.concat(aggr_neg_n)
            aggr_neg_e = torch.concat(aggr_neg_e)

            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean() if mean else loss.sum()
        return loss

class HypeMedPretrainer(nn.Module):
    def __init__(
        self,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        num_edges: int,
        adjacency_dict: dict[str, torch.Tensor],
        pretrain_epochs: int,
        pretrain_learning_rate: float,
        pretrain_weight_decay: float,
        drop_incidence_rate: float,
        drop_feature_rate: float,
        tau_n: float,
        tau_g: float,
        tau_m: float,
        tau_c: float,
        batch_size_1: int,
        batch_size_2: int,
        w_g: float,
        w_m: float,
        embedding_dim: int = 128,
        projection_dim: int | None = None,
        number_of_heads: int = 4,
        dropout: float = 0.1,
        number_of_layers: int = 2,
        device: torch.device | str | None = None,
        idx2word_dict: dict[str, dict[int, str]] | None = None,
        cache_dir: str | os.PathLike = "/tmp/hypemed_pretrain",
    ) -> None:
        super().__init__()
        self.name_lst = ["diagnoses", "procedures", "medications"]
        self.num_dict = {
            "diagnoses": n_diagnoses,
            "procedures": n_procedures,
            "medications": n_medications,
        }
        self.num_edges = num_edges
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.pretrain_epochs = pretrain_epochs
        self.num_negs = None
        self.params = {
            "drop_incidence_rate": drop_incidence_rate,
            "drop_feature_rate": drop_feature_rate,
            "tau_n": tau_n,
            "tau_g": tau_g,
            "tau_m": tau_m,
            "tau_c": tau_c,
            "batch_size_1": batch_size_1,
            "batch_size_2": batch_size_2,
            "w_g": w_g,
            "w_m": w_m,
        }
        projection_dim = projection_dim or embedding_dim
        idx2word_dict = idx2word_dict or {name: {} for name in self.name_lst}
        cache_dir = Path(cache_dir)

        self._adjacency = {
            name: adjacency_dict[name].coalesce().to(self.device)
            for name in self.name_lst
        }
        self.model_dict = nn.ModuleDict({
            name: self.build_model(
                num_nodes=self.num_dict[name],
                num_edges=self.num_edges,
                adj=self._adjacency[name],
                idx2word=idx2word_dict.get(name, {}),
                cache_dir=cache_dir,
                device=self.device,
                name=name,
                embedding_dim=embedding_dim,
                projection_dim=projection_dim,
                number_of_heads=number_of_heads,
                dropout=dropout,
                number_of_layers=number_of_layers,
            )
            for name in self.name_lst
        })
        self.optimizer_dict = {
            name: self.build_optimizer(
                self.model_dict[name],
                lr=pretrain_learning_rate,
                weight_decay=pretrain_weight_decay,
            )
            for name in self.name_lst
        }

    @staticmethod
    def build_model(
        *,
        num_nodes: int,
        num_edges: int,
        adj: torch.Tensor,
        idx2word,
        cache_dir: Path,
        device: torch.device,
        name: str,
        embedding_dim: int,
        projection_dim: int,
        number_of_heads: int,
        dropout: float,
        number_of_layers: int,
    ) -> TriCL:
        encoder = HGTEncoder(
            embed_dim=embedding_dim,
            n_heads=number_of_heads,
            dropout=dropout,
            n_layers=number_of_layers,
            H=adj,
            idx2word=idx2word,
            cache_dir=str(cache_dir),
            device=device,
            name=name,
        ).to(device)
        return TriCL(
            encoder=encoder,
            embedding_dim=embedding_dim,
            proj_dim=projection_dim,
            num_nodes=num_nodes,
            num_edges=num_edges,
            device=device,
        ).to(device)

    @staticmethod
    def build_optimizer(model: nn.Module, *, lr: float, weight_decay: float) -> AdamW:
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    @staticmethod
    def get_raw_node_edge_representation(model: TriCL, adj: torch.Tensor) -> tuple[Tensor, Tensor]:
        return model.get_features(adj)

    def single_domain_step(
        self,
        model: TriCL,
        optimizer: AdamW,
        adj: torch.Tensor,
        num_nodes: int,
    ) -> float:
        hyperedge_index = adj.indices()
        num_edges = self.num_edges
        params = self.params

        model.train()
        optimizer.zero_grad(set_to_none=True)

        hyperedge_index1 = drop_incidence(hyperedge_index, params["drop_incidence_rate"])
        hyperedge_index2 = drop_incidence(hyperedge_index, params["drop_incidence_rate"])
        node_features, edge_features = model.get_features(adj)
        x1 = drop_features(node_features, params["drop_feature_rate"])
        x2 = drop_features(node_features, params["drop_feature_rate"])
        y1 = drop_features(edge_features, params["drop_feature_rate"])
        y2 = drop_features(edge_features, params["drop_feature_rate"])

        node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
        node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
        node_mask = node_mask1 & node_mask2
        edge_mask = edge_mask1 & edge_mask2

        n1, e1 = model(x1, y1, hyperedge_index1)
        n2, e2 = model(x2, y2, hyperedge_index2)
        n1, n2 = model.node_projection(n1), model.node_projection(n2)
        e1, e2 = model.edge_projection(e1), model.edge_projection(e2)

        loss_n = model.node_level_loss(
            n1[node_mask],
            n2[node_mask],
            params["tau_n"],
            batch_size=params["batch_size_1"],
            num_negs=self.num_negs,
        ) if bool(node_mask.any()) else n1.sum() * 0
        loss_g = model.group_level_loss(
            e1[edge_mask],
            e2[edge_mask],
            params["tau_g"],
            batch_size=params["batch_size_1"],
            num_negs=self.num_negs,
        ) if bool(edge_mask.any()) else e1.sum() * 0

        masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask1)
        masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask2)
        loss_m1 = model.membership_level_loss(
            n1,
            e2[edge_mask2],
            masked_index2,
            params["tau_m"],
            batch_size=params["batch_size_2"],
        ) if masked_index2.numel() else n1.sum() * 0
        loss_m2 = model.membership_level_loss(
            n2,
            e1[edge_mask1],
            masked_index1,
            params["tau_m"],
            batch_size=params["batch_size_2"],
        ) if masked_index1.numel() else n2.sum() * 0
        loss_m = (loss_m1 + loss_m2) * 0.5

        loss = loss_n + params["w_g"] * loss_g + params["w_m"] * loss_m
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    def cross_domain_step(self, edges_dict: dict[str, Tensor], tau: float) -> Tensor:
        diagnoses = F.normalize(edges_dict["diagnoses"], dim=-1)
        procedures = F.normalize(edges_dict["procedures"], dim=-1)
        medications = F.normalize(edges_dict["medications"], dim=-1)
        proc_diag_sim = torch.exp(torch.mm(procedures, diagnoses.t()) / tau)
        med_diag_sim = torch.exp(torch.mm(medications, diagnoses.t()) / tau)
        loss = -torch.log(proc_diag_sim.diag() / proc_diag_sim.sum(1))
        loss = loss - torch.log(med_diag_sim.diag() / med_diag_sim.sum(1))
        return loss.mean()

    def pretrain(self) -> dict[str, list[float]]:
        history = {name: [] for name in self.name_lst}
        for _ in trange(self.pretrain_epochs, desc="HypeMed pretraining"):
            for name in self.name_lst:
                loss = self.single_domain_step(
                    self.model_dict[name],
                    self.optimizer_dict[name],
                    self._adjacency[name],
                    self.num_dict[name],
                )
                history[name].append(loss)
        return history

    def get_encoded_embedding(self, model: TriCL, adj: torch.Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            model.eval()
            node_features, edge_features = model.get_features(adj)
            hyperedge_index = adj.indices()
            x_hat, e_hat = model(node_features, edge_features, hyperedge_index)
        return {"X": x_hat, "E": e_hat}

    def get_encoded_embeddings(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        encoded = {
            name: self.get_encoded_embedding(self.model_dict[name], self._adjacency[name])
            for name in self.name_lst
        }
        x_hat = {name: encoded[name]["X"] for name in self.name_lst}
        e_mem = {name: encoded[name]["E"] for name in self.name_lst}
        return x_hat, e_mem

