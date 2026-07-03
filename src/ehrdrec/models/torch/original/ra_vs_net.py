import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .SetTransformer import ScaledDotProductAttention, SAB, MAB
from .homo_relation_graph import homo_relation_graph

import math

from torch.nn.parameter import Parameter


class Aggregation(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(Aggregation, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU()
        )

        self.gate_layer = nn.Linear(64, 1)

    def forward(self, sequence_embeddings):
        gates = self.gate_layer(self.h1(sequence_embeddings))
        gate_probabilities = F.sigmoid(gates)

        return gate_probabilities

class CausaltyReview(nn.Module):
    def __init__(self, casual_graph, num_med):
        super(CausaltyReview, self).__init__()

        self.num_med = num_med
        self.c1 = casual_graph
        diag_to_med_high_threshold = casual_graph.get_threshold_effect(0.97, "Diag", "Med")
        diag_to_med_low_threshold = casual_graph.get_threshold_effect(0.90, "Diag", "Med")
        proc_to_med_high_threshold = casual_graph.get_threshold_effect(0.97, "Proc", "Med")
        proc_to_med_low_threshold = casual_graph.get_threshold_effect(0.90, "Proc", "Med")
        symptom_to_med_high_threshold = casual_graph.get_threshold_effect(0.97, "Sym", "Med")
        symptom_to_med_low_threshold = casual_graph.get_threshold_effect(0.90, "Sym", "Med")
        self.c1_high_limit = nn.Parameter(torch.tensor([
            diag_to_med_high_threshold,
            proc_to_med_high_threshold,
            symptom_to_med_high_threshold,
        ]))  # 选用的97%
        self.c1_low_limit = nn.Parameter(torch.tensor([
            diag_to_med_low_threshold,
            proc_to_med_low_threshold,
            symptom_to_med_low_threshold,
        ]))  # 选用的90%
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.01))
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, predicted_probabilities, diagnoses, procedures, symptoms):
        reviewed_probabilities = predicted_probabilities.clone()

        for medication_index in range(self.num_med):
            max_diag_med_effect = 0.0
            max_proc_med_effect = 0.0
            max_symptom_med_effect = 0.0
            for diagnosis_code in diagnoses:
                diag_med_effect = self.c1.get_effect(diagnosis_code, medication_index, "Diag", "Med")
                max_diag_med_effect = max(max_diag_med_effect, diag_med_effect)
            for procedure_code in procedures:
                proc_med_effect = self.c1.get_effect(procedure_code, medication_index, "Proc", "Med")
                max_proc_med_effect = max(max_proc_med_effect, proc_med_effect)
            for symptom_code in symptoms:
                symptom_med_effect = self.c1.get_effect(symptom_code, medication_index, "Sym", "Med")
                max_symptom_med_effect = max(max_symptom_med_effect, symptom_med_effect)
            if (
                max_diag_med_effect < self.c1_low_limit[0]
                and max_proc_med_effect < self.c1_low_limit[1]
                and max_symptom_med_effect < self.c1_low_limit[2]
            ):

                reviewed_probabilities[0, medication_index] -= self.c1_minus_weight
            elif (
                max_diag_med_effect > self.c1_high_limit[0]
                or max_proc_med_effect > self.c1_high_limit[1]
                or max_symptom_med_effect > self.c1_high_limit[2]
            ):
                reviewed_probabilities[0, medication_index] += self.c1_plus_weight

        return reviewed_probabilities

class BasicModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            embedding_dim=256,
            device=torch.device("cpu:0"),
    ):
        super(BasicModel, self).__init__()

        self.device = device
        self.embedding_dim = embedding_dim

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], embedding_dim) for i in range(4)]
        )
        self.emb_fuse_weight = nn.Embedding(4, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.query = nn.Linear(4 * embedding_dim, vocab_size[3])

        # graphs, bipartite matrix
        self.tensor_ddi_adj = ddi_adj
        self.init_weights()

    def forward(self, patient):
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        current_admission = patient[-1]
        diagnosis_embedding = sum_embedding(
            self.dropout(
                self.embeddings[0](
                    torch.LongTensor(current_admission[5]).unsqueeze(dim=0).to(self.device)
                )
            )
        )  # (1,1,dim)
        procedure_embedding = sum_embedding(
            self.dropout(
                self.embeddings[1](
                    torch.LongTensor(current_admission[6]).unsqueeze(dim=0).to(self.device)
                )
            )
        )
        symptom_embedding = self.dropout(
            self.embeddings[2](torch.LongTensor(current_admission[7]).unsqueeze(dim=0).to(self.device))
        )
        symptom_embedding = sum_embedding(symptom_embedding)

        if current_admission == patient[0]:
            prior_medication_embedding = torch.zeros(1, 1, self.embedding_dim).to(self.device)
        else:
            previous_admission = patient[-2]
            prior_medication_embedding = sum_embedding(
                self.dropout(
                    self.embeddings[3](
                        torch.LongTensor(previous_admission[8]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )

        emb_fuse_weight = self.emb_fuse_weight(torch.tensor([0, 1, 2, 3]).to(self.device))
        patient_representations = torch.cat(
            [
                diagnosis_embedding * emb_fuse_weight[0],
                procedure_embedding * emb_fuse_weight[1],
                symptom_embedding * emb_fuse_weight[2],
                prior_medication_embedding * emb_fuse_weight[3],
            ],
            dim=-1).squeeze(0)
        result = self.query(patient_representations)  # (1, dim)

        negative_prediction_probabilities = F.sigmoid(result)
        negative_prediction_probabilities = negative_prediction_probabilities.t() * negative_prediction_probabilities  # (voc_size, voc_size)

        batch_negative_loss = 0.0005 * negative_prediction_probabilities.mul(self.tensor_ddi_adj).sum()

        return result, batch_negative_loss

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


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
        transformed_features = torch.mm(input, self.weight)
        convolved_features = torch.mm(adj, transformed_features)
        if self.bias is not None:
            return convolved_features + self.bias
        else:
            return convolved_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
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
        row_sums = np.array(mx.sum(1))
        inverse_row_sums = np.power(row_sums, -1).flatten()
        inverse_row_sums[np.isinf(inverse_row_sums)] = 0.
        inverse_row_sum_matrix = np.diagflat(inverse_row_sums)
        normalized_matrix = inverse_row_sum_matrix.dot(mx)
        return normalized_matrix



class RaVSNet(nn.Module):
    def __init__(
            self,
            causal_graph,
            n_diagnoses: int,
            n_procedures: int,
            n_symptoms: int,
            n_medications: int,
            ddi_adj,
            ehr_adj_med_diag,
            ehr_adj_med_proc,
            ehr_adj_med_med,
            ehr_adj_med_sym,
            medication_list,
            pretrained_embedding,
            embedding_dim: int = 256,
            device=torch.device("cuda:1"),
    ):
        super(RaVSNet, self).__init__()

        self.device = device
        self.emb_dim = embedding_dim
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_symptoms = n_symptoms
        self.n_medications = n_medications
        
        self.medication_list = medication_list

        self.causal_graph = causal_graph

        self.embeddings = nn.ModuleList(
            pretrained_embedding
        )
        self.homo_graph = nn.ModuleList([
            homo_relation_graph(embedding_dim, device),
            homo_relation_graph(embedding_dim, device),
            homo_relation_graph(embedding_dim, device),
            homo_relation_graph(embedding_dim, device)
        ])

        self.emb_fuse_weight = nn.Embedding(4, 1)
        self.cross_att = ScaledDotProductAttention(4 * embedding_dim, 4 * embedding_dim, embedding_dim, 4)
        self.drug_output = nn.Linear(embedding_dim, embedding_dim)
        self.drug_layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.diag_gcn = GCN(voc_size=n_diagnoses + n_medications, emb_dim=embedding_dim, adj=ehr_adj_med_diag,
                            device=device)
        self.proc_gcn = GCN(voc_size=n_procedures + n_medications, emb_dim=embedding_dim, adj=ehr_adj_med_proc,
                            device=device)
        self.sym_gcn = GCN(voc_size=n_symptoms + n_medications, emb_dim=embedding_dim, adj=ehr_adj_med_sym,
                           device=device)
        self.med_gcn = GCN(voc_size=n_medications, emb_dim=embedding_dim, adj=ehr_adj_med_med, device=device)

        self.mab1 = MAB(embedding_dim, embedding_dim, embedding_dim, 2, use_ln=True)
        self.mab2 = MAB(embedding_dim, embedding_dim, embedding_dim, 2, use_ln=True)
        self.mab3 = MAB(embedding_dim, embedding_dim, embedding_dim, 2, use_ln=True)
        self.mab4 = MAB(embedding_dim, embedding_dim, embedding_dim, 2, use_ln=True)
        self.mab5 = MAB(embedding_dim, embedding_dim, embedding_dim, 2, use_ln=True)
        self.mab6 = MAB(embedding_dim, embedding_dim, embedding_dim, 2, use_ln=True)
        self.sab1 = SAB(embedding_dim, embedding_dim, 2, use_ln=True)
        self.sab2 = SAB(embedding_dim, embedding_dim, 2, use_ln=True)
        self.sab3 = SAB(embedding_dim, embedding_dim, 2, use_ln=True)
        self.sab4 = SAB(embedding_dim, embedding_dim, 2, use_ln=True)
        self.sab5 = SAB(embedding_dim, embedding_dim, 2, use_ln=True)
        self.sab6 = SAB(embedding_dim, embedding_dim, 2, use_ln=True)


        self.pat_fuse = nn.Linear(9 * embedding_dim, embedding_dim)
        self.med_fuse = nn.Linear(6 * embedding_dim, embedding_dim)
        self.fuse_weight = nn.Embedding(2, 1)
        self.recomb = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * n_medications, n_medications)
        )
        self.recomd = nn.Sequential(
            nn.ReLU(),
            nn.Linear(9 * embedding_dim, n_medications)
        )
        self.docter_weight = nn.Embedding(2, n_medications)
        self.review = CausaltyReview(self.causal_graph, n_medications)
        self.tensor_ddi_adj = ddi_adj
        self.ehr_adj_med_diag = ehr_adj_med_diag
        self.ehr_adj_med_proc = ehr_adj_med_proc
        self.ehr_adj_med_sym = ehr_adj_med_sym
        self.ehr_adj_med_med = ehr_adj_med_med

        self.MLP_layer1 = nn.Linear(embedding_dim * 4, 1)
        self.MLP_layer2 = nn.Linear(2, 1)
        self.gumbel_tau = 0.3
        self.att_tau = 20
        self.linear_layer = nn.Linear(4*embedding_dim, 4 * embedding_dim)

    """ calculate target-aware attention """

    def calc_cross_visit_scores(self, embedding):
        """ embedding: (batch * visit_num * emb) """

        # Extract the current att value when calculating attention
        visit_keys = embedding[:, :]  # key: past visits and current visit
        current_visit_query = embedding[-1:, :]  # query: current visit
        attention_scores = torch.mm(self.linear_layer(current_visit_query), visit_keys.transpose(0, 1)) / math.sqrt(
            current_visit_query.size(-1))  # attention weight
        encoder_attention_scores = attention_scores
        attention_weights = F.softmax(attention_scores / self.att_tau, dim=-1)
        encoder_attention_weights = F.softmax(encoder_attention_scores / self.att_tau, dim=-1)
        return attention_weights, encoder_attention_weights

    def forward(self, features):
        patient = features["patient"]
        visit_embedding_table = features["visit_embedding_table"]
        
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(0)  # (1,1,dim)

        diagnosis_ehr_embeddings = self.diag_gcn.forward()
        procedure_ehr_embeddings = self.proc_gcn.forward()
        medication_ehr_embeddings = self.med_gcn.forward()
        symptom_ehr_embeddings = self.sym_gcn.forward()


        diagnosis_drug_embeddings = diagnosis_ehr_embeddings[:self.n_medications].unsqueeze(0)
        procedure_drug_embeddings = procedure_ehr_embeddings[:self.n_medications].unsqueeze(0)
        medication_drug_embeddings = medication_ehr_embeddings.unsqueeze(0)
        symptom_drug_embeddings = symptom_ehr_embeddings[:self.n_medications].unsqueeze(0)

        diagnosis_procedure_drug_context = self.mab1(diagnosis_drug_embeddings, procedure_drug_embeddings)
        diagnosis_procedure_drug_context = self.sab1(diagnosis_procedure_drug_context)
        diagnosis_medication_drug_context = self.mab2(diagnosis_drug_embeddings, medication_drug_embeddings)
        diagnosis_medication_drug_context = self.sab2(diagnosis_medication_drug_context)
        diagnosis_symptom_drug_context = self.mab3(diagnosis_drug_embeddings, symptom_drug_embeddings)
        diagnosis_symptom_drug_context = self.sab3(diagnosis_symptom_drug_context)

        procedure_medication_drug_context = self.mab4(procedure_drug_embeddings, medication_drug_embeddings)
        procedure_medication_drug_context = self.sab4(procedure_medication_drug_context)
        procedure_symptom_drug_context = self.mab5(procedure_drug_embeddings, symptom_drug_embeddings)
        procedure_symptom_drug_context = self.sab5(procedure_symptom_drug_context)

        medication_symptom_drug_context = self.mab6(medication_drug_embeddings, symptom_drug_embeddings)
        medication_symptom_drug_context = self.sab6(medication_symptom_drug_context)

        drug_context = torch.cat([
            diagnosis_procedure_drug_context,
            diagnosis_medication_drug_context,
            diagnosis_symptom_drug_context,
            procedure_medication_drug_context,
            procedure_symptom_drug_context,
            medication_symptom_drug_context,
        ], dim=-1)  # (1,med_num,dim*6)
        medication_context = self.med_fuse(drug_context)  # (med_num,dim)
        diagnosis_sequence, procedure_sequence, symptom_sequence, medication_sequence = [], [], [], []
        for admission_index, admission in enumerate(patient):
            diagnosis_embeddings = self.dropout(
                self.embeddings[0](torch.LongTensor(admission[0]).unsqueeze(dim=0).to(self.device))
            )
            diagnosis_graph = self.causal_graph.get_graph(admission[4], "Diag")
            diagnosis_context = self.homo_graph[0](diagnosis_graph, diagnosis_embeddings)

            procedure_embeddings = self.dropout(
                self.embeddings[1](torch.LongTensor(admission[1]).unsqueeze(dim=0).to(self.device))
            )
            procedure_graph = self.causal_graph.get_graph(admission[4], "Proc")
            procedure_context = self.homo_graph[1](procedure_graph, procedure_embeddings)

            symptom_embeddings = self.dropout(
                self.embeddings[2](torch.LongTensor(admission[2]).unsqueeze(dim=0).to(self.device))
            )
            symptom_graph = self.causal_graph.get_graph(admission[4], "Sym")
            symptom_context = self.homo_graph[2](symptom_graph, symptom_embeddings)

            if admission == patient[0]:
                medication_context_embedding = torch.zeros(1, 1, self.embedding_dim).to(self.device)
            else:
                previous_admission = patient[admission_index - 1]
                medication_embeddings = self.dropout(
                    self.embeddings[3](torch.LongTensor(previous_admission[3]).unsqueeze(dim=0).to(self.device))
                )
                medication_graph = self.causal_graph.get_graph(previous_admission[4], "Med")
                medication_context_embedding = self.homo_graph[3](medication_graph, medication_embeddings)

            diagnosis_sequence.append(torch.sum(diagnosis_context, keepdim=True, dim=1))
            procedure_sequence.append(torch.sum(procedure_context, keepdim=True, dim=1))
            symptom_sequence.append(torch.sum(symptom_context, keepdim=True, dim=1))
            medication_sequence.append(torch.sum(medication_context_embedding, keepdim=True, dim=1))

        diagnosis_sequence = torch.cat(diagnosis_sequence, dim=1)  # (1,seq,dim)
        procedure_sequence = torch.cat(procedure_sequence, dim=1)  # (1,seq,dim)
        symptom_sequence = torch.cat(symptom_sequence, dim=1)  # (1,seq,dim)
        medication_sequence = torch.cat(medication_sequence, dim=1)  # (1,seq,dim)
        if len(patient) >= 2:
            #当前健康嵌入
            patient_sequence_representation = torch.concatenate([
                diagnosis_sequence,
                procedure_sequence,
                symptom_sequence,
                medication_sequence,
            ], dim=-1).squeeze(dim=0)   #(seq,dim*4)
            visit_selection_logits = self.MLP_layer1(patient_sequence_representation) #(seq,1)
            current_visit_query = visit_selection_logits[-1:, :]
            current_visit_query = current_visit_query.repeat(visit_selection_logits.size()[0], 1)
            visit_selection_features = torch.cat([visit_selection_logits, current_visit_query], dim=-1)  # (seq,2)
            visit_selection_probability = torch.sigmoid(self.MLP_layer2(visit_selection_features))
            gumbel_input = torch.cat([visit_selection_probability, 1 - visit_selection_probability], dim=-1)
            sampled_visit_mask = F.gumbel_softmax(gumbel_input, tau=self.gumbel_tau, hard=True)[:, 0]
            visit_mask = torch.cat([sampled_visit_mask[:-1], torch.ones(1, device = self.device)])   #保证最后一次选中
            diagnosis_sequence = visit_mask.unsqueeze(0).unsqueeze(-1) * diagnosis_sequence
            procedure_sequence = visit_mask.unsqueeze(0).unsqueeze(-1) * procedure_sequence
            symptom_sequence = visit_mask.unsqueeze(0).unsqueeze(-1) * symptom_sequence
            medication_sequence = visit_mask.unsqueeze(0).unsqueeze(-1) * medication_sequence
        visit_embedding = torch.concatenate([
            diagnosis_sequence,
            procedure_sequence,
            symptom_sequence,
            medication_sequence,
        ], dim=-1).squeeze(dim=0)
        cross_visit_scores, scores_encoder = self.calc_cross_visit_scores(visit_embedding)
        visit_embedding = visit_embedding * cross_visit_scores.T
        patient_representations = torch.sum(visit_embedding, dim=0, keepdim=True)

        current_admission = patient[-1]
        comparison_visit_embeddings = torch.cat((
            visit_embedding_table[:current_admission[4]],
            visit_embedding_table[current_admission[4] + 1:]
        ), dim=0)
        similar_visit_scores = torch.cosine_similarity(patient_representations, comparison_visit_embeddings, dim=1)

        top_scores, top_indices = torch.topk(similar_visit_scores, k=10)
        original_visit_indices = []
        similar_patient_list = []
        for top_index in top_indices:
            similar_patient_list.append(comparison_visit_embeddings[top_index.item()].unsqueeze(dim=0))
            if top_index.item() >= current_admission[4]:
                original_visit_indices.append(top_index.item() + 1)
            else:
                original_visit_indices.append(top_index.item())
        similar_patient_embeddings = torch.cat(similar_patient_list, dim=0).unsqueeze(dim=0)  # (1,topk,4*dim)
        similar_patient_embeddings = self.cross_att(
            patient_representations.unsqueeze(dim=0),
            similar_patient_embeddings,
            similar_patient_embeddings,
        )  # (1,1,4*dim)
        patient_embedding = torch.cat([patient_representations, similar_patient_embeddings.squeeze(dim=0)], dim=-1)  # (1,12*dim)

        similar_medication = [self.medication_list[visit_index] for visit_index in original_visit_indices]
        similar_medication_list = []
        for medication_codes in similar_medication:
            similar_medication_embedding = self.dropout(
                self.embeddings[3](torch.LongTensor(medication_codes).unsqueeze(dim=0).to(self.device)))  # (1,1,dim)
            similar_medication_embedding = sum_embedding(similar_medication_embedding)
            similar_medication_list.append(similar_medication_embedding)

        similar_medication_embeddings = torch.cat(similar_medication_list, dim=1).squeeze(dim=0)  # (10,dim)
        similar_medication_embeddings = torch.mm(top_scores.unsqueeze(dim=0), similar_medication_embeddings)
        similar_medication_embeddings = self.drug_layernorm(
            similar_medication_embeddings + self.drug_output(similar_medication_embeddings)
        )
        patient_embedding = torch.cat([patient_embedding, similar_medication_embeddings], dim=-1)  # (9,dim)

        fused_patient_embedding = self.pat_fuse(patient_embedding)
        fuse_weight = self.fuse_weight(torch.tensor([0, 1]).to(self.device))
        docter_weight = self.docter_weight(torch.tensor([0, 1]).to(self.device))
        medication_similarity = torch.cosine_similarity(
            fused_patient_embedding,
            medication_context.squeeze(0),
            dim=1,
        ).unsqueeze(0)
        docter_direct = self.recomd(patient_embedding)
        docter_recomb = self.recomb(
            torch.cat([docter_direct * fuse_weight[0], medication_similarity * fuse_weight[1]], dim=1)
        )
        result = docter_direct * docter_weight[0] + docter_recomb * docter_weight[1]
        result = self.review(result, patient[-1][0], patient[-1][1], patient[-1][2])
        diagnosis_medication_adjacency = self.ehr_adj_med_diag[:self.vocab_size[3], :self.vocab_size[0]].t().to(self.device)
        diagnosis_medication_signal = torch.sum(diagnosis_medication_adjacency[current_admission[0]], keepdim=True, dim=0)
        procedure_medication_adjacency = self.ehr_adj_med_diag[:self.vocab_size[3], :self.vocab_size[1]].t().to(self.device)
        procedure_medication_signal = torch.sum(procedure_medication_adjacency[current_admission[1]], keepdim=True, dim=0)
        symptom_medication_adjacency = self.ehr_adj_med_diag[:self.vocab_size[3], :self.vocab_size[2]].t().to(self.device)
        symptom_medication_signal = torch.sum(symptom_medication_adjacency[current_admission[2]], keepdim=True, dim=0)
        result += F.sigmoid(
            diagnosis_medication_signal + procedure_medication_signal + symptom_medication_signal
        )

        negative_prediction_probabilities = F.sigmoid(result)
        negative_prediction_probabilities = negative_prediction_probabilities.t() * negative_prediction_probabilities  # (voc_size[2], voc_size[2])
        batch_negative_loss = 0.0005 * negative_prediction_probabilities.mul(self.tensor_ddi_adj).sum()

        return result, batch_negative_loss, patient_representations
