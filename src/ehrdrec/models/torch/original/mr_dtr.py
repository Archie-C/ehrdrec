import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

class GraphConvolution(torch.nn.Module):
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

    def forward(self, node_features, adjacency_matrix):
        support = torch.mm(node_features, self.weight)
        output = torch.mm(adjacency_matrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GCN(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        ehr_adjacency_matrix: torch.Tensor, 
        ddi_adjacency_matrix: torch.Tensor, 
        device=torch.device('cpu:0')
    ):
        super(GCN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device

        normalized_ehr_adj = self.normalize(ehr_adjacency_matrix + torch.eye(ehr_adjacency_matrix.shape[0]))
        normalized_ddi_adj = self.normalize(ddi_adjacency_matrix + torch.eye(ehr_adjacency_matrix.shape[0]))

        self.ehr_adj = torch.FloatTensor(normalized_ehr_adj).to(device)
        self.ddi_adjacency_matrix = torch.FloatTensor(normalized_ddi_adj).to(device)
        self.identity_features = torch.eye(vocab_size).to(device)

        self.shared_input_layer = GraphConvolution(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.ehr_output_layer = GraphConvolution(embedding_dim, embedding_dim)
        self.ddi_output_layer = GraphConvolution(embedding_dim, embedding_dim)

    def forward(self):
        ehr_node_embedding = self.shared_input_layer(self.identity_features, self.ehr_adj)
        ddi_node_embedding = self.shared_input_layer(self.identity_features, self.ddi_adjacency_matrix)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.ehr_output_layer(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.ddi_output_layer(ddi_node_embedding, self.ddi_adjacency_matrix)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, adjacency_matrix):
        """Row-normalize sparse matrix"""
        rowsum = np.array(adjacency_matrix.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        normalized_adjacency_matrix = r_mat_inv.dot(adjacency_matrix)
        return normalized_adjacency_matrix

class PeriodicTimeEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super(PeriodicTimeEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale_factor = (1 / (embedding_dim // 2)) ** 0.5

        self.frequency_weights = nn.Parameter(torch.randn(1, embedding_dim // 2))
        self.phase_offsets = nn.Parameter(torch.randn(1, embedding_dim // 2))

    def forward(self, relative_time: torch.Tensor):
        """

        :param relative_time: shape (batch_size, temporal_feature_dimension) or (batch_size, max_neighbors_num, temporal_feature_dimension)
               input_time_dim = 1 since the feature denotes relative time (scalar)
        :return:
            time_encoding, shape (batch_size, embedding_dim) or (batch_size, max_neighbors_num, embedding_dim)
        """
        # print(relative_time.shape)
        # cos_encoding, shape (batch_size, embedding_dim // 2) or (batch_size, max_neighbors_num, embedding_dim // 2)
        cos_encoding = torch.cos(torch.matmul(relative_time, self.frequency_weights) + self.phase_offsets)
        # sin_encoding, shape (batch_size, embedding_dim // 2) or (batch_size, max_neighbors_num, embedding_dim // 2)
        sin_encoding = torch.sin(torch.matmul(relative_time, self.frequency_weights) + self.phase_offsets)

        # time_encoding, shape (batch_size, embedding_dim) or (batch_size, max_neighbors_num, embedding_dim)
        time_encoding = self.scale_factor * torch.cat([cos_encoding, sin_encoding], dim=-1)

        return time_encoding

class ContextSelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(ContextSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, neighbor_context, patient_context, medication_memory):

        stacked_inputs = torch.stack([neighbor_context, patient_context, medication_memory], dim=1) # shape: (med_num, 3, embedding_dim)
        

        queries = self.query(stacked_inputs) # shape: (med_num, 3, embedding_dim)
        keys = self.key(stacked_inputs) # shape: (med_num, 3, embedding_dim)
        values = self.value(stacked_inputs) # shape: (med_num, 3, embedding_dim)
        

        neighbor_attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embedding_dim ** 0.5) # shape: (med_num, 3, 3)
        attention_weights = self.softmax(neighbor_attention_scores) # shape: (med_num, 3, 3)
        

        weighted_values = torch.matmul(attention_weights, values) # shape: (med_num, 3, embedding_dim)
        

        output = torch.sum(weighted_values, dim=1) # shape: (med_num, embedding_dim)
        
        return output

class MRDTR(nn.Module):
    def __init__(
        self, 
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        n_patients: int,
        embedding_dim: int = 128,
        embedding_dropout: float = 0.1,
        temporal_attention_dropout: float = 0.1,
        temporal_information_importance: float = 0.5,
        ehr_adjacency_matrix: torch.Tensor | None = None,
        ddi_adjacency_matrix: torch.Tensor | None = None,
        device: torch.device | None = None,
        hop_num: int = 3,
        temporal_feature_dim: int = 1        
    ):
        super(MRDTR, self).__init__()
        
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.n_patients = n_patients
        self.embedding_dim = embedding_dim
        self.embedding_dropout_rate = embedding_dropout
        self.temporal_attention_dropout_rate = temporal_attention_dropout
        self.temporal_information_importance = temporal_information_importance
        self.ehr_adjacency_matrix = ehr_adjacency_matrix
        self.ddi_adjacency_matrix = ddi_adjacency_matrix
        self.device = device
        self.hop_num = hop_num
        self.temporal_feature_dim = temporal_feature_dim
        
        self.patient_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
        )
        
        self.medication_gcn = GCN(
            vocab_size=self.n_medications,
            embedding_dim=self.embedding_dim,
            ehr_adjacency_matrix=self.ehr_adjacency_matrix,
            ddi_adjacency_matrix=self.ddi_adjacency_matrix,
            device=self.device
        )
        
        self.diagnosis_embedding = nn.Sequential(
            nn.Embedding(self.n_diagnoses, self.embedding_dim),
            nn.Dropout(p=self.embedding_dropout_rate)
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(self.n_procedures, self.embedding_dim),
            nn.Dropout(p=self.embedding_dropout_rate)
        )
        self.medication_embedding = nn.Sequential(
            nn.Embedding(self.n_medications, self.embedding_dim),
            nn.Dropout(p=self.embedding_dropout_rate)
        )
        
        self.patient_embedding = nn.Embedding(self.n_patients, self.embedding_dim)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.embedding_dropout_layer = nn.Dropout(p=self.embedding_dropout_rate)
        self.temporal_attention_dropout_layer = nn.Dropout(p=self.temporal_attention_dropout_rate)
        
        self.num_attention_heads = 2
        diagnosis_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_attention_heads,
            batch_first=True,
            dropout=0.2,
        )
        procedure_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_attention_heads,
            batch_first=True,
            dropout=0.2,
        )
        self.diagnosis_transformer_encoder = nn.TransformerEncoder(
            diagnosis_encoder_layer,
            num_layers=1,
        )
        self.procedure_transformer_encoder = nn.TransformerEncoder(
            procedure_encoder_layer,
            num_layers=1,
        )

        self.periodic_time_encoder = PeriodicTimeEncoder(embedding_dim=self.embedding_dim)
        
        self.neighbor_context_projection = nn.Linear(self.hop_num * self.embedding_dim, self.embedding_dim)
        self.context_attention = ContextSelfAttention(embedding_dim=self.embedding_dim)
        
        self.output_projection = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, 1)
        )
        
        self.ddi_memory_weight = Parameter(torch.FloatTensor(1))
        self.medication_memory_weight = Parameter(torch.FloatTensor(1))
        self.neighbor_context_weight = Parameter(torch.FloatTensor(1))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1

        self.diagnosis_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.procedure_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.medication_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.patient_embedding.weight.data.uniform_(-initrange, initrange)
    
    def forward(
        self,
        hop_node_indices: list, 
        hop_node_temporal_features: list,
        central_node_temporal_feature,
        diagnosis_code_lists: list,
        procedure_code_lists: list,
    ):
        query_embeddings = self.medication_embedding(
            torch.LongTensor([i for i in range(self.n_medications)]).to(self.device)
        )
        
        hop_context_embeddings = []
        
        diagnosis_features = torch.stack([
            torch.sum(
                self.diagnosis_embedding(torch.LongTensor(ids).to(self.device)),
                dim=0,
            )
            for ids in diagnosis_code_lists
        ])
        procedure_features = torch.stack([
            torch.sum(
                self.procedure_embedding(torch.LongTensor(ids).to(self.device)),
                dim=0,
            )
            for ids in procedure_code_lists
        ])
        diagnosis_features = diagnosis_features[-1]
        procedure_features = procedure_features[-1]

        patient_context = torch.cat([diagnosis_features, procedure_features], dim=-1)
        patient_context = self.patient_projection(patient_context)
        patient_context = patient_context * query_embeddings
        
        ehr_embedding, ddi_embedding = self.medication_gcn()
        medication_memory = ehr_embedding - ddi_embedding * self.ddi_memory_weight
        
        central_node_temporal_embedding = self.periodic_time_encoder(torch.Tensor([[central_node_temporal_feature]]).to(self.device))
        
        for hop_index in range(len(hop_node_indices)):
            
            current_hop_node_indices, current_hop_temporal_features = hop_node_indices[hop_index], hop_node_temporal_features[hop_index]
            
            if hop_index % 2 == 0:
                if hop_index == 0:
                    continue
                else:
                    hop_node_embeddings = self.patient_embedding(torch.LongTensor([current_hop_node_indices]).to(self.device))
            else:
                hop_diagnosis_embeddings = self.diagnosis_embedding(torch.LongTensor([current_hop_node_indices[0]]).to(self.device))
                hop_procedure_embeddings = self.procedure_embedding(torch.LongTensor([current_hop_node_indices[1]]).to(self.device))
                hop_medication_embeddings = self.medication_embedding(torch.LongTensor([current_hop_node_indices[2]]).to(self.device))
                hop_node_embeddings = torch.cat([hop_diagnosis_embeddings, hop_procedure_embeddings, hop_medication_embeddings], dim=1)
            
            hop_node_embeddings = self.embedding_dropout_layer(hop_node_embeddings)
            
            neighbor_attention = torch.einsum('if,bnf->bin', query_embeddings, hop_node_embeddings)
            neighbor_attention = self.leaky_relu(neighbor_attention)
            neighbor_attention_scores = F.softmax(neighbor_attention, dim=-1)
            hop_context = torch.bmm(neighbor_attention_scores, hop_node_embeddings)
            hop_context_embeddings.append(hop_context)
            
            hop_temporal_embeddings = self.periodic_time_encoder(torch.Tensor([current_hop_temporal_features]).unsqueeze(dim=-1).to(self.device))
            
            temporal_attention = torch.einsum('bif,bnf->bin', torch.stack([central_node_temporal_embedding for _ in range(self.n_medications)], dim=1), hop_temporal_embeddings)
            temporal_attention = self.temporal_attention_dropout_layer(temporal_attention)
            
            
        hop_context_embeddings = self.embedding_dropout_layer(torch.stack(hop_context_embeddings, dim=2))
        neighbor_context = self.neighbor_context_projection(hop_context_embeddings.flatten(start_dim=2))
        combined_context = self.neighbor_context_weight * neighbor_context.squeeze(dim=0) + patient_context + self.medication_memory_weight * medication_memory
        
        medication_logits = self.output_projection(combined_context).t()
        negative_prediction_probabilities = torch.sigmoid(medication_logits)
        negative_prediction_probabilities = torch.matmul(negative_prediction_probabilities.t(), negative_prediction_probabilities)
        ddi_adjacency_matrix = self.ddi_adjacency_matrix.to(
            device=negative_prediction_probabilities.device,
            dtype=negative_prediction_probabilities.dtype,
        )
        ddi_loss = 0.0005 * negative_prediction_probabilities.mul(ddi_adjacency_matrix).sum()
        return {
            "predictions": medication_logits,
            "losses" : {
                "ddi_loss": ddi_loss,
            }
        }
            
            