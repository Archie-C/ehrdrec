import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter

from collections import namedtuple
from einops import rearrange, repeat
from torch import einsum
from ehrdrec.utils.constants import ReservedId
from torchtyping import patch_typeguard, TensorType
from typeguard import typechecked
from typing import Optional, Tuple
from x_transformers.x_transformers import (
    apply_rotary_pos_emb, default, exists, FeedForward, RMSNorm, RotaryEmbedding
)

patch_typeguard()

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


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates + seq_masks
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output

class PositionEmbedding(nn.Module):
    """
    We assume that the sequence length is less than 512.
    """
    def __init__(self, emb_size, max_length=512):
        super(PositionEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding_layer = nn.Embedding(max_length, emb_size)

    def forward(self, batch_size, seq_length, device):
        assert(seq_length <= self.max_length)
        ids = torch.arange(0, seq_length).long().to(torch.device(device))
        ids = ids.unsqueeze(0).repeat(batch_size, 1)
        emb = self.embedding_layer(ids)
        return emb

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attn_mask=None):
        Q = self.fc_q(Q) 
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        
        attn_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        if attn_mask is not None:
            attn_mask = attn_mask.view_as(attn_score)
            if attn_mask.dtype == torch.bool:
                attn_score.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_score += attn_mask
        

        A = torch.softmax(attn_score, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, input):
        X, attn_mask = input
        return self.mab(X, X, attn_mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.FloatTensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.num_inds = num_inds

    def forward(self, input):
        X, attn_mask = input
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attn_mask) 
        attn_mask = attn_mask.transpose(-2, -1)
        return self.mab1(X, H, attn_mask)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim)) # [1, K, dim]
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X) #[batch_size*visit_len, K, dim]

SeqTensor = TensorType['batch', 'seq_len', 'token_dim']
StateTensor = TensorType['batch', 'state_len', 'state_dim']

# constants

DEFAULT_DIM_HEAD = 64
MIN_DIM_HEAD = 64 #32

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t: SeqTensor, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

@typechecked
class RecurrentStateGate(nn.Module):
    """Poor man's LSTM
    """

    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x: SeqTensor, state: StateTensor) -> StateTensor:
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)


class Attention(nn.Module):
    """Shamelessly copied from github.com/lucidrains/RETRO-pytorch
    """
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        null_kv = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # allowing for attending to nothing (null function)
        # and to save attention from breaking if all retrieved chunks are padded out
        self.null_kv = nn.Parameter(torch.randn(2, inner_dim)) if null_kv else None

    def forward(self, x, mask = None, context = None, pos_emb = None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        x = self.norm(x)
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * scale

        # apply relative positional encoding (rotary embeddings)
        if exists(pos_emb):
            if isinstance(pos_emb, tuple) and len(pos_emb) == 2 and not torch.is_tensor(pos_emb[1]):
                pos_emb = pos_emb[0]
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
            # print(f'q: {q.shape}\nq_pos_emb: {q_pos_emb.shape}')
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # add null key / values
        if exists(self.null_kv):
            nk, nv = self.null_kv.unbind(dim = 0)
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h = h), (nk, nv))
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        # derive query key similarities
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking
        mask_value = -torch.finfo(sim.dtype).max

        
        if exists(mask):
            # Add the triangular mask manually to avoid information leakage.
            tril_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1]).view(1,1,mask.shape[-1], mask.shape[-1])).bool().to(mask.device)
            sim = sim.masked_fill(~tril_mask, mask_value)
            if exists(self.null_kv):
                mask = F.pad(mask, (1, 0), value = True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # combine heads linear out
        return self.to_out(out), None


@typechecked
class BlockRecurrentAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_state: int,
        dim_head: int = DEFAULT_DIM_HEAD,
        state_len: int = 512,
        heads: int = 8,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.dim = dim
        self.dim_state = dim_state

        self.heads = heads
        self.causal = True
        self.state_len = state_len
        rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        
        self.input_self_attn = Attention(dim, heads = heads, causal = True, **attn_kwargs)
        self.state_self_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.input_state_cross_attn = Attention(dim, heads = heads, causal = False, **attn_kwargs)
        self.state_input_cross_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.proj_gate = RecurrentStateGate(dim)
        self.ff_gate = RecurrentStateGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias = False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias = False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)

    def forward(
        self,
        x: SeqTensor,
        state: Optional[StateTensor] = None,
        mask = None,
        state_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None
    ) -> Tuple[SeqTensor, StateTensor]:
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = torch.zeros((batch, self.state_len, self.dim_state)).to(x.device)
        self_attn_pos_emb = self.rotary_pos_emb(torch.arange(seq_len, device=device))
        state_pos_emb = self.rotary_pos_emb(torch.arange(state.shape[-2], device=device))
        input_attn, _ = self.input_self_attn(x, mask = mask, pos_emb = self_attn_pos_emb)
        state_attn, _ = self.state_self_attn(state, mask = state_mask, pos_emb = state_pos_emb)

        # This actually is different from how it is implemented in the paper, because the Keys and Values aren't shared
        # between the cross attention and self-attention. I'll implement that later, this is faster for now.
        input_as_q_cross_attn, _ = self.input_state_cross_attn(x, context = state, mask = mask) # , context_mask = state_mask # [batch, seq_len, hd]
        state_as_q_cross_attn, _ = self.state_input_cross_attn(state, context = x, mask = state_mask) # , context_mask = mask  # [batch, state_len, hd]

        projected_input = self.input_proj(torch.cat((input_as_q_cross_attn, input_attn), dim=2)) # torch.concat
        projected_state = self.state_proj(torch.cat((state_as_q_cross_attn, state_attn), dim=2)) # torch.concat

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state


class SHAPE(nn.Module):
    def __init__(
        self, 
        n_diagnoses: int, 
        n_procedures: int,
        n_medications: int,
        ehr_adjacency_matrix: torch.Tensor,
        ddi_adjacency_matrix: torch.Tensor, 
        ddi_mask_H, 
        embedding_dim: int = 128, 
        hidden_dim: int = 128,
        device=torch.device('cpu:0'), 
        num_inds=32, 
        num_heads=2, 
        ln=False, 
        isab_num=2, 
        kgloss_alpha=0.001
    ):
        super(SHAPE, self).__init__()
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        
        self.ehr_adjacency_matrix = ehr_adjacency_matrix
        self.ddi_adjacency_matrix = ddi_adjacency_matrix    
        
        self.emb_dim = embedding_dim
        self.device = device
        self.nhead = num_heads
        
        self.MED_PAD_TOKEN = ReservedId.PAD  
        self.DIAG_PAD_TOKEN = ReservedId.PAD
        self.PROC_PAD_TOKEN = ReservedId.PAD

        self.isab_num = isab_num
        self.num_inds = num_inds

        num_outputs = self.n_medications
        dim_output = 1

        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        # dig_num * emb_dim
        self.diag_embedding = nn.Sequential(
            nn.Embedding(self.n_diagnoses+3, embedding_dim, self.DIAG_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # proc_num * emb_dim
        self.proc_embedding = nn.Sequential(
            nn.Embedding(self.n_procedures+3, embedding_dim, self.PROC_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            # Add padding_idx so padding maps to the zero vector.
            nn.Embedding(self.n_medications+3, embedding_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # Set-encoder module.
        self.isab = ISAB(embedding_dim, hidden_dim, num_heads, num_inds, ln=ln)
        self.diag_enc = nn.Sequential(
                ISAB(embedding_dim, hidden_dim, num_heads, num_inds, ln=ln),
                ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln))
        self.proc_enc = nn.Sequential(
                ISAB(embedding_dim, hidden_dim, num_heads, num_inds, ln=ln),
                ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln))
        self.med_enc = nn.Sequential(
                ISAB(embedding_dim, hidden_dim, num_heads, num_inds, ln=ln),
                ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln))
        
        # Use a recurrent transformer to encode visit-level temporal information.
        self.dim_hidden = hidden_dim
        self.recurrent_attn = BlockRecurrentAttention(hidden_dim*3, hidden_dim*3)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(hidden_dim*3, self.n_medications)
        
        self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(embedding_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(
            ddi_mask_H.shape[1], self.n_medications, False)        

        # Add the DDI graph.
        self.gcn =  GCN(voc_size=self.n_medications, emb_dim=embedding_dim, ehr_adj=self.ehr_adjacency_matrix, ddi_adj=self.ddi_adjacency_matrix, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.kgloss_alpha = kgloss_alpha
    
    def set_encoder(self, input, attn_mask, rep=2):
        attn_mask = attn_mask.unsqueeze(2).repeat(1,1,self.num_inds, 1)
        for i in range(rep):
            input = self.isab([input, attn_mask])
        return input


    def forward(self, features):
        diseases = features["diseases"]
        procedures = features["procedures"]
        medications = features["medications"]
        d_mask_matrix = features["d_mask_matrix"]
        p_mask_matrix = features["p_mask_matrix"]
        m_mask_matrix = features["m_mask_matrix"]
        seq_length = features["seq_length"]

        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        # 1. First compute code-level embeddings.
        diag_emb = self.diag_embedding(diseases) # [batch_size, diag_code_len, emb]
        proc_emb = self.proc_embedding(procedures) # [batch_size, proc_code_len, emb]
        
        # 2. Medications need an extra padded visit record. Build new_medication
        #    to represent the previous visit, then concatenate it with medication
        #    records from timesteps [0, t-1].
        new_medication = torch.full((batch_size, 1, max_med_num), self.MED_PAD_TOKEN).to(self.device)
        new_medication = torch.cat([new_medication, medications[:, :-1, :]], dim=1) # new_medication.shape=[b,max_seq_len, max_med_num]
        # Shift m_mask_matrix in the same way.
        new_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(self.device) # Use a large negative value so softmax assigns no probability mass.
        new_m_mask = torch.cat([new_m_mask, m_mask_matrix[:, :-1, :]], dim=1)
        med_emb = self.med_embedding(new_medication)

        # # 3. Encode at the code level with set encoders.
        # # 3.1 Diagnosis set encoders.
        d_enc_mask_matrix = d_mask_matrix.view(batch_size*max_seq_length, max_diag_num).unsqueeze(1).repeat(1, self.nhead, 1) # [batch_size*visit_num, nhead, diag_len]
        diag_enc_input = diag_emb.view(batch_size*max_seq_length, max_diag_num, -1)
        diag_encode = self.set_encoder(diag_enc_input, d_enc_mask_matrix) # [batch_size*visit_len, diag_len, hdm]
        # 3.2 Procedure set encoders.
        p_enc_mask_matrix = p_mask_matrix.view(batch_size*max_seq_length, max_proc_num).unsqueeze(1).repeat(1, self.nhead, 1) # [batch_size*visit_num, nhead, proc_len]
        proc_enc_input = proc_emb.view(batch_size*max_seq_length, max_proc_num, -1)
        proc_encode = self.set_encoder(proc_enc_input, p_enc_mask_matrix)  # [batch_size*visit_len, proc_len, hdm]
        # 3.3 Medication set encoders.
        m_enc_mask_matrix = new_m_mask.view(batch_size*max_seq_length, max_med_num).unsqueeze(1).repeat(1, self.nhead, 1) # [batch_size*visit_len, nhead, med_len]
        
        # 3.4. Get representations from the EHR graph and DDI graph.
        ehr_embedding, ddi_embedding = self.gcn() # [vocab_size, hdm]
        drug_memory = ehr_embedding - ddi_embedding * self.inter # Use the co-occurrence graph minus the DDI graph representation.
        drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float() # Special tokens.
        drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)# [vocab_size, hdm]

        # # 3.5 Directly add the two medication-code representations.
        # med_memory_emb = drug_memory[new_medication] # [batch_size, max_seq_length, med_code_len, hdm]
        # med_emb = med_emb + med_memory_emb
        
        m_enc_input = med_emb.view(batch_size*max_seq_length, max_med_num, -1)
        med_encode = self.set_encoder(m_enc_input, m_enc_mask_matrix) # [batch_size, max_seq_length, med_code_len, hdm]
        
        # 4. Aggregate the three code types separately and convert them to visit level.
        diag_enc = torch.sum(diag_encode, dim=1).view(batch_size, max_seq_length, -1)
        proc_enc = torch.sum(proc_encode, dim=1).view(batch_size, max_seq_length, -1)
        med_enc = torch.sum(med_encode, dim=1).view(batch_size, max_seq_length, -1)
        visit_enc = torch.cat([diag_enc, proc_enc, med_enc], dim=-1) # [batch_size, max_seq_length, 3*hdm]

        # 5. Pass visit-level representations through the recurrent transformer.
        visit_mask = torch.full((batch_size, max_seq_length), 0).to(self.device)
        state = torch.zeros((batch_size, max_seq_length, self.dim_hidden*3)).to(self.device)
        for i, v_l in enumerate(seq_length):
            visit_mask[i, :v_l] = 1
        output, state = self.recurrent_attn(visit_enc, state, mask=visit_mask.bool()) # 
        # output, state = self.gru(visit_enc)
        sequence_output = self.output_layer(output) # [batch_size, max_seq_length, vocab_size]
        sequence_output = sequence_output * visit_mask.unsqueeze(-1)
        last_visit_idx = (seq_length.to(sequence_output.device) - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=sequence_output.device)
        decoder_output = sequence_output[batch_idx, last_visit_idx] # [batch_size, vocab_size]

        # 6. Compute DDIs for the predicted medication set and constrain it with the known DDI matrix.
        sigmoid_output = torch.sigmoid(decoder_output)
        sigmoid_output_ddi = sigmoid_output.unsqueeze(2) * sigmoid_output.unsqueeze(1) # [batch_size, vocab_size, vocab_size]
        kg_ddi = torch.as_tensor(
            self.ddi_adjacency_matrix,
            dtype=sigmoid_output.dtype,
            device=sigmoid_output.device,
        ).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, vocab_size, vocab_size]
        kg_ddi_score = 0.001 * self.kgloss_alpha * torch.sum(kg_ddi * sigmoid_output_ddi, dim=[-1,-2]).mean()

        return {
            "predictions": decoder_output,
            "sequence_predictions": sequence_output,
            "losses": {
                "ddi_loss": kg_ddi_score,
            },
        }
