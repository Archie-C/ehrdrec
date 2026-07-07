import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
            self.register_parameter("bias", None)
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
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        if isinstance(adj, torch.Tensor):
            adj = adj.detach().float().cpu().numpy()
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


class Aggregation(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(Aggregation, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU()
        )
        
        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs):
        gates = self.gate_layer(self.h1(seqs))
        output = F.sigmoid(gates)

        return output

class RASNet(nn.Module):
    
    def __init__(
        self,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        ehr_adjacency_matrix: torch.Tensor,
        ddi_adjacency_matrix: torch.Tensor,
        ddi_mask_H: torch.Tensor,
        fai,
        dropout: float = 0.5,
        embedding_dim: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(RASNet, self).__init__()
        self.fai = fai
        self.device = device
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.embedding_dim = embedding_dim

        # Initialize the EHR and DDI adjacency matrices
        self.ehr_adjacency_matrix = ehr_adjacency_matrix.to(self.device)
        self.ddi_adjacency_matrix = ddi_adjacency_matrix.to(self.device)
        self.ddi_mask_H = ddi_mask_H.to(self.device)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_diagnoses, embedding_dim),
            nn.Embedding(n_procedures, embedding_dim),
            nn.Embedding(n_medications, embedding_dim)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoders = nn.ModuleList([
            nn.GRU(embedding_dim, embedding_dim, batch_first=True),
            nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        ])
        
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear( 2 * embedding_dim, embedding_dim),
        )
        
        self.poly = Aggregation(embedding_dim * 2)
        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, n_medications),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.ehr_gcn = GCN(voc_size=n_medications, emb_dim=embedding_dim, adj=self.ehr_adjacency_matrix, device=self.device)
        self.ddi_gcn = GCN(voc_size=n_medications, emb_dim=embedding_dim, adj=self.ddi_adjacency_matrix, device=self.device)
        self.register_buffer("tensor_ddi_adj", self.ddi_adjacency_matrix.float())
        self.inter1 = Parameter(torch.ones(1), requires_grad=True)
    
    def forward(self, features):
        preser = []
        def sum_embedding(embedding):
            return embedding.sum(dim=0).view(1, 1, -1)
        
        diagnoses_list = features["diagnoses"]
        procedures_list = features["procedures"]
        
        diagnoses_emb_list = torch.empty(1, 0, self.embedding_dim, device=self.device)
        procedures_emb_list = torch.empty(1, 0, self.embedding_dim, device=self.device)
        
        for diagnoses in diagnoses_list:
            diagnoses_emb = self.embeddings[0](diagnoses)
            diagnoses_emb = self.dropout(diagnoses_emb)
            diagnoses_emb = sum_embedding(diagnoses_emb)
            diagnoses_emb_list = torch.cat([diagnoses_emb_list, diagnoses_emb], dim=1)
        
        for procedures in procedures_list:
            procedures_emb = self.embeddings[1](procedures)
            procedures_emb = self.dropout(procedures_emb)
            procedures_emb = sum_embedding(procedures_emb)
            procedures_emb_list = torch.cat([procedures_emb_list, procedures_emb], dim=1)
            
        if diagnoses_emb_list.size(1) >= 2:
            patient_representation = torch.cat([diagnoses_emb_list, procedures_emb_list], dim=-1).squeeze(dim=0)
            cur_query = patient_representation[-1:, :]
            poly_cur = self.poly(cur_query)
            for i in range(len(patient_representation) - 1):
                poly_history = self.poly(patient_representation[i:i + 1])
                s = abs(poly_cur - poly_history)
                if s.item() <= self.fai:
                    preser.append(i)
            
            if not preser:
                diagnoses_emb_list = diagnoses_emb_list[:, -1:, :]
                procedures_emb_list = procedures_emb_list[:, -1:, :]
            else:
                preser.append(len(patient_representation) - 1)
                diagnoses_emb_list = torch.cat([diagnoses_emb_list[:, i:i+1, :] for i in preser], dim=1)
                procedures_emb_list = torch.cat([procedures_emb_list[:, i:i+1, :] for i in preser], dim=1)
        
        o1, h1, = self.encoders[0](diagnoses_emb_list)
        o2, h2, = self.encoders[1](procedures_emb_list)
        
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)
        
        query = self.query(patient_representations)[-1:, :]
        safe_gcn = self.ehr_gcn() - self.inter1 * self.ddi_gcn()
        
        med_base = self.embeddings[2](torch.arange(self.n_medications, device=self.device))
        drug_gcn = safe_gcn + med_base
        
        key_weights1 = torch.softmax(torch.mm(query , drug_gcn.t()), dim=-1)  # (1, size)
        med_result = torch.mm(key_weights1, drug_gcn)
        
        final_representations = torch.cat([self.layer_norm(query), med_result], dim=-1)
        result = self.classification(final_representations)
        
        if self.training:
            neg_pred_prob = torch.sigmoid(result)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob

            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
            return {
                "predictions": result,
                "losses": {
                    "ddi_loss": batch_neg
                }
            }

        return {
            "predictions": result
        }
        