import math
import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn.parameter import Parameter


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

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        if isinstance(ehr_adj, torch.Tensor):
            ehr_adj = ehr_adj.detach().float().cpu().numpy()
        if isinstance(ddi_adj, torch.Tensor):
            ddi_adj = ddi_adj.detach().float().cpu().numpy()
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

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        # Q = main_feat
        # K = other_feat
        # Attn = torch.nn.functional.cosine_similarity(Q, K)
        # Attn[Attn < 0] = -1e9

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        # Attn = torch.softmax(Attn, dim=-1)

        return Attn

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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, src_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if src_mask is not None:
            A = A.masked_fill(src_mask < -1e8, -1e9)
        A = self.softmax(A)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, src_mask=None):
        return self.mab(X, X, src_mask)


class Encoder_SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(Encoder_SAB, self).__init__()
        self.sab1 = SAB(dim_in, dim_in, dim_out, num_heads)
        self.sab2 = SAB(dim_in, dim_in, dim_out, num_heads)

    def forward(self, X, src_mask=None):
        return self.sab2(self.sab1(X, src_mask), src_mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        self.sab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, src_mask=None):
        X = self.sab(X, X, src_mask)
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, src_mask)

class MedTransformerDecoder_all(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5) -> None:
        super(MedTransformerDecoder_all, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2m_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead

    def forward(self, input_med, input_disease_embdding=None, input_proc_embedding=None):
        if input_disease_embdding is None:
            input_disease_embdding = input_proc_embedding
        if input_proc_embedding is None:
            input_proc_embedding = input_disease_embdding

        x = input_med
        x = self.norm1(x + self.self_block(x, attn_mask=None))
        x = self.norm2(x + self._m2d_mha_block(x, input_disease_embdding, attn_mask=None)
                         + self._m2p_mha_block(x, input_proc_embedding, attn_mask=None))
        x = self.norm3(x + self._ff_block(x))

        return x

    def self_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)

    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)

    def _m2m_mha_block(self, x, mem, attn_mask):
        x = self.m2m_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)


    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class SSPNet(nn.Module):
    def __init__(
        self,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        ehr_adjacency_matrix: torch.Tensor,
        ddi_adjacency_matrix: torch.Tensor,
        embedding_dim: int = 128,
        use_embeddings: bool = True,
        number_of_heads: int = 4,
        device: torch.device = torch.device("cpu"),
        dropout: float = 0.5,
    ) -> None:
        super(SSPNet, self).__init__()
        self.device = device
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.embedding_dim = embedding_dim
        self.use_embeddings = use_embeddings

        # Initialize the EHR and DDI adjacency matrices
        self.ehr_adjacency_matrix = ehr_adjacency_matrix.to(self.device)
        self.ddi_adjacency_matrix = ddi_adjacency_matrix.to(self.device)

        self.score_extractor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.gcn = GCN(voc_size=n_medications, emb_dim=embedding_dim, ehr_adj=ehr_adjacency_matrix, ddi_adj=ddi_adjacency_matrix, device=device)
        self.med_embedding = nn.Sequential(
            nn.Embedding(n_medications, embedding_dim),
            nn.Dropout(dropout),
        )
        self.diag_embedding = nn.Sequential(
            nn.Embedding(n_diagnoses, embedding_dim),
            nn.Dropout(dropout),
        )
        self.proc_embedding = nn.Sequential(
            nn.Embedding(n_procedures, embedding_dim),
            nn.Dropout(dropout),
        )
        
        self.diag_encoder = Encoder_SAB(dim_in=embedding_dim, dim_out=embedding_dim, num_heads=number_of_heads)
        self.proc_encoder = Encoder_SAB(dim_in=embedding_dim, dim_out=embedding_dim, num_heads=number_of_heads)
        self.decoder = MedTransformerDecoder_all(d_model=embedding_dim, nhead=number_of_heads)
        self.pma_d = PMA(dim=embedding_dim, num_heads=number_of_heads)
        self.pma_p = PMA(dim=embedding_dim, num_heads=number_of_heads)
        self.aggregator = AdjAttenAgger(Qdim=embedding_dim, Kdim=embedding_dim, mid_dim=embedding_dim)
        self.W_z = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.inter = Parameter(torch.ones(1), requires_grad=True)
        self.garm = Parameter(torch.ones(1), requires_grad=True)
        self.W_visit = nn.Linear(embedding_dim * 2, embedding_dim)
        self.seq_encoders = nn.ModuleList([
            nn.GRU(embedding_dim, embedding_dim, batch_first=True),
            nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        ])
        self.out_visit = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, features):
        diagnoses = features["diagnoses"]                  # [1, visit_len, max_diag]
        procedures = features["procedures"]                # [1, visit_len, max_proc]
        medication_history = features["medication_history"] # [1, visit_len, n_medications] multi-hot

        ##################### medication representation #####################

        possible_meds = torch.arange(
            self.n_medications,
            device=self.device
        ).long()

        med_emb = self.med_embedding(possible_meds)

        ehr_embedding, ddi_embedding = self.gcn()

        # Original uses minus here
        med_ehr_ddi = ehr_embedding - self.inter * ddi_embedding

        med_repr = med_emb + med_ehr_ddi

        #####################################################################

        ##################### current patient representation ################

        visit_len = diagnoses.size(1)

        # original uses patient_data[-1]
        current_diag = diagnoses[:, -1, :]
        current_proc = procedures[:, -1, :]

        d_emb = self.diag_embedding(current_diag)
        p_emb = self.proc_embedding(current_proc)

        d_repr = self.diag_encoder(d_emb)
        p_repr = self.proc_encoder(p_emb)

        #####################################################################

        ##################### personalised medication representation ########

        if visit_len > 1:
            d_history_all = []
            p_history_all = []

            # encode every visit separately, like original
            for adm in range(visit_len):
                diag_ids = diagnoses[:, adm, :]
                proc_ids = procedures[:, adm, :]

                d_emb_h = self.diag_embedding(diag_ids)
                p_emb_h = self.proc_embedding(proc_ids)

                d_repr_h = self.diag_encoder(d_emb_h)
                p_repr_h = self.proc_encoder(p_emb_h)

                d_history_all.append(d_repr_h)
                p_history_all.append(p_repr_h)

            d_history_pma = []
            p_history_pma = []
            d_p_history_pma = []

            for i in range(len(d_history_all)):
                d_temp = self.pma_d(d_history_all[i]).squeeze(0)
                p_temp = self.pma_p(p_history_all[i]).squeeze(0)

                d_p_temp = self.W_visit(torch.cat([d_temp, p_temp], dim=-1))

                d_p_history_pma.append(d_p_temp.squeeze(0))
                d_history_pma.append(d_temp)
                p_history_pma.append(p_temp)

            d_p_history_pma = torch.stack(d_p_history_pma, dim=0)

            # original: RNN across visits, not inside each visit
            d_history_pma = torch.stack(d_history_pma, dim=0)
            p_history_pma = torch.stack(p_history_pma, dim=0)

            output1, hidden1 = self.seq_encoders[0](d_history_pma)
            output2, hidden2 = self.seq_encoders[1](p_history_pma)

            output_d_p = self.out_visit(torch.cat([output1, output2], dim=-1))

            output_d_p = output_d_p.squeeze(1)

            # attention over visits
            score_c = self.aggregator(output_d_p[-1], output_d_p)
            score_c = torch.softmax(score_c, dim=-1)
            score_c = score_c.squeeze(0)

            # previous medications only, to avoid target leakage
            m_h = torch.zeros(
                1,
                self.n_medications,
                device=self.device
            )

            for i in range(visit_len - 1):
                m_emb_h = medication_history[:, i, :].float()
                m_h += m_emb_h * score_c[i]

            m_c = torch.ones(
                self.n_medications,
                device=self.device
            )

            m_weight = m_c.unsqueeze(0) + m_h * self.garm

            med_repr = med_repr * m_weight.t()

        #####################################################################

        ##################### medication prediction #########################

        hidden = self.decoder(
            med_repr.unsqueeze(0),
            d_repr,
            p_repr
        )

        hidden = hidden.squeeze(0)

        score = self.score_extractor(hidden).t()

        #####################################################################

        ##################### DDI penalty ###################################
        if self.training:

            neg_pred_prob = torch.sigmoid(score)

            neg_pred_prob = torch.matmul(
                neg_pred_prob.t(),
                neg_pred_prob
            )

            batch_neg = 0.0005 * neg_pred_prob.mul(self.ddi_adjacency_matrix).sum()
        
        return {
            "predictions": score,
            "losses": {
                "ddi_loss": batch_neg
            } if self.training else None
        }