import torch
import torch.nn as nn

from ehrdrec.utils.constants import ReservedId


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).int().unsqueeze(0)
        x = x + self.embeddings(pos).expand_as(x)
        return self.dropout(x)


class PatientEncoder(nn.Module):
    def __init__(
        self,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        number_of_heads: int = 4,
        encoder_layers: int = 2,
        patient_separate: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(PatientEncoder, self).__init__()
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.number_of_heads = number_of_heads
        self.encoder_layers = encoder_layers
        self.patient_separate = patient_separate
        self.device = device

        self.patient_memory_contact = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 4),
            nn.Tanh(),
            nn.Linear(self.embedding_dim * 4, self.n_medications),
            nn.Tanh(),
            nn.Dropout(p=self.dropout),
        )

        if self.patient_separate:
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(self.n_diagnoses, self.embedding_dim // 2),
                    nn.Embedding(self.n_procedures, self.embedding_dim // 2),
                ]
            )
            self.diagnosis_projection = nn.Linear(self.n_diagnoses, self.embedding_dim // 2)
            self.procedure_projection = nn.Linear(self.n_procedures, self.embedding_dim // 2)
            self.transformer_diagnoses = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim // 2,
                    nhead=self.number_of_heads,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                num_layers=self.encoder_layers,
            )
            self.transformer_procedures = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim // 2,
                    nhead=self.number_of_heads,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                num_layers=self.encoder_layers,
            )
            self.patient_layer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )

            self.position_embedding_diagnoses = LearnablePositionalEncoding(self.embedding_dim // 2, dropout=self.dropout)
            self.position_embedding_procedures = LearnablePositionalEncoding(self.embedding_dim // 2, dropout=self.dropout)
            self.patient_encoder = self.patient_encoder_separate
        else:
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(self.n_diagnoses, self.embedding_dim),
                    nn.Embedding(self.n_procedures, self.embedding_dim),
                ]
            )
            self.diagnosis_projection = nn.Linear(self.n_diagnoses, self.embedding_dim)
            self.procedure_projection = nn.Linear(self.n_procedures, self.embedding_dim)
            self.transformer_visit = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim,
                    nhead=self.number_of_heads,
                    dropout=self.dropout,
                    batch_first=True,
                ),
                num_layers=self.encoder_layers,
            )
            self.position_embedding_diagnoses = LearnablePositionalEncoding(self.embedding_dim, dropout=self.dropout)
            self.position_embedding_procedures = LearnablePositionalEncoding(self.embedding_dim, dropout=self.dropout)
            self.patient_encoder = self.patient_encoder_unified

    def _code_set_representation(
        self,
        values: torch.Tensor,
        embedding: nn.Embedding,
        projection: nn.Linear,
        vocab_size: int,
    ) -> torch.Tensor:
        if values.is_floating_point() and values.size(-1) == vocab_size:
            return projection(values)

        values = values.long().clamp(min=0, max=vocab_size - 1)
        mask = values.ne(int(ReservedId.PAD))
        embedded = embedding(values)
        embedded = embedded * mask.unsqueeze(-1)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return embedded.sum(dim=-2) / counts

    def _sequence_padding_mask(self, values: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor | None:
        if lengths is None:
            return None
        positions = torch.arange(values.size(1), device=values.device).unsqueeze(0)
        return positions >= lengths.to(values.device).unsqueeze(1)

    def patient_encoder_separate(self, diagnoses, procedures, lengths=None):
        diagnoses_embeddings = self._code_set_representation(
            diagnoses,
            self.embeddings[0],
            self.diagnosis_projection,
            self.n_diagnoses,
        )
        procedures_embeddings = self._code_set_representation(
            procedures,
            self.embeddings[1],
            self.procedure_projection,
            self.n_procedures,
        )

        diagnoses_embeddings = self.position_embedding_diagnoses(diagnoses_embeddings)
        procedures_embeddings = self.position_embedding_procedures(procedures_embeddings)
        padding_mask = self._sequence_padding_mask(diagnoses_embeddings, lengths)

        diagnoses_encoded = self.transformer_diagnoses(
            diagnoses_embeddings,
            src_key_padding_mask=padding_mask,
        )
        procedures_encoded = self.transformer_procedures(
            procedures_embeddings,
            src_key_padding_mask=padding_mask,
        )
        patient_representation = torch.cat((diagnoses_encoded, procedures_encoded), dim=-1)
        return self.patient_layer(patient_representation)

    def patient_encoder_unified(self, diagnoses, procedures, lengths=None):
        diagnoses_embeddings = self._code_set_representation(
            diagnoses,
            self.embeddings[0],
            self.diagnosis_projection,
            self.n_diagnoses,
        )
        procedures_embeddings = self._code_set_representation(
            procedures,
            self.embeddings[1],
            self.procedure_projection,
            self.n_procedures,
        )

        diagnoses_embeddings = self.position_embedding_diagnoses(diagnoses_embeddings)
        procedures_embeddings = self.position_embedding_procedures(procedures_embeddings)
        combined_embeddings = diagnoses_embeddings + procedures_embeddings

        return self.transformer_visit(
            combined_embeddings,
            src_key_padding_mask=self._sequence_padding_mask(combined_embeddings, lengths),
        )


class RPNet(PatientEncoder):
    def __init__(
        self,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        embedding_dim: int,
        encoder_layers: int,
        number_of_heads: int,
        dropout: float,
        patient_separate: bool,
        ddi_adjacency_matrix: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        super(RPNet, self).__init__(
            n_diagnoses=n_diagnoses,
            n_procedures=n_procedures,
            n_medications=n_medications,
            embedding_dim=embedding_dim,
            dropout=dropout,
            number_of_heads=number_of_heads,
            encoder_layers=encoder_layers,
            patient_separate=patient_separate,
            device=device,
        )
        self.register_buffer("ddi_adjacency_matrix", ddi_adjacency_matrix.float())
        self.cls_final = nn.Linear(self.embedding_dim, self.n_medications)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embeddings[0].weight.data.uniform_(-initrange, initrange)
        self.embeddings[1].weight.data.uniform_(-initrange, initrange)

    def _last_valid_visit(self, visit_representations: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(visit_representations.size(0), device=visit_representations.device)
        visit_idx = lengths.to(visit_representations.device).clamp_min(1) - 1
        return visit_representations[batch_idx, visit_idx]

    def _medication_history_dense(self, medication_history: torch.Tensor) -> torch.Tensor:
        if medication_history.is_floating_point() and medication_history.size(-1) == self.n_medications:
            return medication_history.float()

        medication_history = medication_history.long()
        valid = medication_history.ne(int(ReservedId.PAD))
        medication_history = medication_history.clamp(min=0, max=self.n_medications - 1)
        dense = torch.zeros(
            *medication_history.shape[:2],
            self.n_medications,
            dtype=torch.float32,
            device=medication_history.device,
        )
        dense.scatter_add_(
            dim=-1,
            index=medication_history,
            src=valid.to(torch.float32),
        )
        return dense.clamp_max(1.0)

    def drug_retrieval(
        self,
        current_visit_representation: torch.Tensor,
        visit_representations: torch.Tensor,
        medication_history: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        medication_history = self._medication_history_dense(medication_history)

        batch_size, max_visits, _ = visit_representations.shape
        positions = torch.arange(max_visits, device=visit_representations.device).unsqueeze(0)
        current_visit_idx = lengths.to(visit_representations.device).clamp_min(1).unsqueeze(1) - 1
        history_mask = positions < current_visit_idx

        if not history_mask.any():
            return torch.zeros(
                batch_size,
                self.n_medications,
                dtype=visit_representations.dtype,
                device=visit_representations.device,
            )

        current_expanded = current_visit_representation.unsqueeze(1).expand(-1, max_visits, -1)
        pair_representations = torch.cat([current_expanded, visit_representations], dim=-1)
        gates = self.patient_memory_contact(pair_representations)
        gated_history = gates * medication_history.to(gates.dtype)
        gated_history = gated_history * history_mask.unsqueeze(-1)
        return gated_history.sum(dim=1)

    def _ddi_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        pair_probabilities = probabilities.t().matmul(probabilities)
        return pair_probabilities.mul(self.ddi_adjacency_matrix.to(logits.device)).sum() / logits.size(0)

    def forward(self, features):
        diagnoses = features["diagnoses"]
        procedures = features["procedures"]
        medication_history = features["medication_history"]
        lengths = features.get(
            "lengths",
            torch.full(
                (diagnoses.size(0),),
                diagnoses.size(1),
                dtype=torch.long,
                device=diagnoses.device,
            ),
        )

        visit_representations = self.patient_encoder(diagnoses, procedures, lengths)
        current_visit_representation = self._last_valid_visit(visit_representations, lengths)

        logits = self.cls_final(current_visit_representation)
        logits = logits + self.drug_retrieval(
            current_visit_representation=current_visit_representation,
            visit_representations=visit_representations,
            medication_history=medication_history,
            lengths=lengths,
        )

        return {
            "predictions": logits,
            "losses": {
                "ddi_loss": self._ddi_loss(logits),
            },
        }
