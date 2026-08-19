from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Graph convolution used by the original COGNet implementation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        support = torch.mm(
            x,
            self.weight,
        )

        output = torch.mm(
            adjacency,
            support,
        )

        if self.bias is not None:
            output = output + self.bias

        return output


class SelfAttend(nn.Module):
    """
    Visit-level self-attention used by the original COGNet code.
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.Tanh(),
        )

        self.gate_layer = nn.Linear(
            32,
            1,
        )

    def forward(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gates = self.gate_layer(
            self.h1(sequences)
        ).squeeze(-1)

        if masks is not None:
            gates = gates + masks

        attention = F.softmax(
            gates,
            dim=-1,
        ).unsqueeze(-1)

        return torch.sum(
            sequences * attention,
            dim=1,
        )


class GCN(nn.Module):
    """
    COGNet's shared EHR/DDI graph encoder.
    """

    def __init__(
        self,
        medications_vocab_size: int,
        embedding_dim: int,
        ehr_adj: torch.Tensor,
        ddi_adj: torch.Tensor,
    ) -> None:
        super().__init__()

        self.medications_vocab_size = medications_vocab_size
        self.embedding_dim = embedding_dim

        ehr_adj = self._normalise(
            torch.as_tensor(
                ehr_adj,
                dtype=torch.float32,
            )
            + torch.eye(
                medications_vocab_size,
                dtype=torch.float32,
            )
        )

        ddi_adj = self._normalise(
            torch.as_tensor(
                ddi_adj,
                dtype=torch.float32,
            )
            + torch.eye(
                medications_vocab_size,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "ehr_adj",
            ehr_adj,
        )

        self.register_buffer(
            "ddi_adj",
            ddi_adj,
        )

        self.register_buffer(
            "x",
            torch.eye(
                medications_vocab_size,
                dtype=torch.float32,
            ),
        )

        # The original implementation intentionally shares gcn1 between
        # the EHR and DDI graphs, then uses separate second layers.
        self.gcn1 = GraphConvolution(
            medications_vocab_size,
            embedding_dim,
        )

        self.dropout = nn.Dropout(
            p=0.3,
        )

        self.gcn2 = GraphConvolution(
            embedding_dim,
            embedding_dim,
        )

        self.gcn3 = GraphConvolution(
            embedding_dim,
            embedding_dim,
        )

    @staticmethod
    def _normalise(
        matrix: torch.Tensor,
    ) -> torch.Tensor:
        rowsum = matrix.sum(
            dim=1,
            keepdim=True,
        )

        inverse = torch.where(
            rowsum > 0,
            rowsum.reciprocal(),
            torch.zeros_like(rowsum),
        )

        return matrix * inverse

    def forward(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ehr_node_embedding = self.gcn1(
            self.x,
            self.ehr_adj,
        )

        ddi_node_embedding = self.gcn1(
            self.x,
            self.ddi_adj,
        )

        ehr_node_embedding = F.relu(
            ehr_node_embedding
        )

        ddi_node_embedding = F.relu(
            ddi_node_embedding
        )

        ehr_node_embedding = self.dropout(
            ehr_node_embedding
        )

        ddi_node_embedding = self.dropout(
            ddi_node_embedding
        )

        ehr_node_embedding = self.gcn2(
            ehr_node_embedding,
            self.ehr_adj,
        )

        ddi_node_embedding = self.gcn3(
            ddi_node_embedding,
            self.ddi_adj,
        )

        return (
            ehr_node_embedding,
            ddi_node_embedding,
        )


class MedTransformerDecoder(nn.Module):
    """
    Medication decoder used by COGNet.
    """

    def __init__(
        self,
        embedding_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(
            embedding_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.m2d_multihead_attn = nn.MultiheadAttention(
            embedding_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.m2p_multihead_attn = nn.MultiheadAttention(
            embedding_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            embedding_dim,
            dim_feedforward,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

        self.linear2 = nn.Linear(
            dim_feedforward,
            embedding_dim,
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim,
            eps=layer_norm_eps,
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim,
            eps=layer_norm_eps,
        )

        self.norm3 = nn.LayerNorm(
            embedding_dim,
            eps=layer_norm_eps,
        )

        self.dropout1 = nn.Dropout(
            dropout,
        )

        self.dropout2 = nn.Dropout(
            dropout,
        )

        self.dropout3 = nn.Dropout(
            dropout,
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        input_medication_embedding: torch.Tensor,
        input_medication_memory: torch.Tensor,
        input_disease_embedding: torch.Tensor,
        input_procedure_embedding: torch.Tensor,
        input_medication_self_mask: torch.Tensor,
        disease_mask: torch.Tensor,
        procedure_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_count = input_medication_embedding.size(0)
        target_length = input_medication_embedding.size(1)

        subsequent_mask = self._generate_square_subsequent_mask(
            target_length,
            input_count * self.nhead,
            input_disease_embedding.device,
        )

        self_attention_mask = (
            subsequent_mask
            + input_medication_self_mask
        )

        x = (
            input_medication_embedding
            + input_medication_memory
        )

        x = self.norm1(
            x
            + self._self_attention_block(
                x,
                self_attention_mask,
            )
        )

        x = self.norm2(
            x
            + self._medication_to_diagnosis_block(
                x,
                input_disease_embedding,
                disease_mask,
            )
            + self._medication_to_procedure_block(
                x,
                input_procedure_embedding,
                procedure_mask,
            )
        )

        x = self.norm3(
            x
            + self._feed_forward_block(
                x
            )
        )

        return x

    def _self_attention_block(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]

        return self.dropout1(
            x
        )

    def _medication_to_diagnosis_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.m2d_multihead_attn(
            x,
            memory,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]

        return self.dropout2(
            x
        )

    def _medication_to_procedure_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.m2p_multihead_attn(
            x,
            memory,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]

        return self.dropout2(
            x
        )

    def _feed_forward_block(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )

        return self.dropout3(
            x
        )

    @staticmethod
    def _generate_square_subsequent_mask(
        size: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = (
            torch.triu(
                torch.ones(
                    (size, size),
                    device=device,
                )
            )
            == 1
        ).transpose(0, 1)

        mask = (
            mask.float()
            .masked_fill(
                mask == 0,
                -1e9,
            )
            .masked_fill(
                mask == 1,
                0.0,
            )
        )

        return mask.unsqueeze(0).repeat(
            batch_size,
            1,
            1,
        )


class Beam:
    """
    Small local beam-search implementation matching the search semantics
    used in the original COGNet repository.

    Repeated medication tokens are not allowed within one generated
    prescription.
    """

    def __init__(
        self,
        size: int,
        bos_token: int,
        eos_token: int,
        device: torch.device,
    ) -> None:
        self.size = size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.device = device

        self.done = False
        self.beam_finished = [False] * size

        self.scores = torch.zeros(
            size,
            dtype=torch.float32,
            device=device,
        )

        self.prev_ks: list[torch.Tensor] = []

        self.next_ys = [
            torch.full(
                (size,),
                bos_token,
                dtype=torch.long,
                device=device,
            )
        ]

        self.probability_history: list[
            list[torch.Tensor]
        ] = []

    def get_current_state(
        self,
        sort: bool = True,
    ) -> torch.Tensor:
        if len(self.next_ys) == 1:
            return self.next_ys[0].unsqueeze(1)

        if sort:
            _, keys = self.sort_scores()
            keys = keys.tolist()
        else:
            keys = list(
                range(self.size)
            )

        hypotheses = [
            [
                self.bos_token,
                *self.get_hypothesis(k),
            ]
            for k in keys
        ]

        return torch.as_tensor(
            hypotheses,
            dtype=torch.long,
            device=self.device,
        )

    def advance(
        self,
        word_log_probabilities: torch.Tensor,
    ) -> bool:
        vocabulary_size = (
            word_log_probabilities.size(1)
        )

        if self.done:
            identity = torch.arange(
                self.size,
                device=self.device,
            )

            self.prev_ks.append(
                identity
            )

            self.next_ys.append(
                torch.full(
                    (self.size,),
                    self.eos_token,
                    dtype=torch.long,
                    device=self.device,
                )
            )

            self.probability_history.append(
                [
                    torch.zeros(
                        vocabulary_size,
                        device=self.device,
                    )
                    for _ in range(self.size)
                ]
            )

            return True

        active_indices = torch.tensor(
            [
                index
                for index, finished
                in enumerate(
                    self.beam_finished
                )
                if not finished
            ],
            dtype=torch.long,
            device=self.device,
        )

        finished_indices = torch.tensor(
            [
                index
                for index, finished
                in enumerate(
                    self.beam_finished
                )
                if finished
            ],
            dtype=torch.long,
            device=self.device,
        )

        active_word_log_probs = (
            word_log_probabilities[
                active_indices
            ]
        )

        current_output = self.get_current_state(
            sort=False
        )

        active_scores = self.scores[
            active_indices
        ]

        finished_scores = self.scores[
            finished_indices
        ]

        if self.prev_ks:
            beam_scores = (
                active_word_log_probs
                + active_scores.unsqueeze(1)
            )
        else:
            # At the first decoding step every beam is identical,
            # so only the first row is considered.
            beam_scores = (
                active_word_log_probs[0]
            )

        flattened = beam_scores.reshape(
            -1
        )

        active_candidate_count = (
            flattened.numel()
        )

        flattened = torch.cat(
            [
                flattened,
                finished_scores,
            ],
            dim=0,
        )

        sorted_scores, sorted_ids = (
            torch.sort(
                flattened,
                descending=True,
            )
        )

        selected_scores = []
        selected_words = []
        selected_beams = []
        new_finished_status = []
        selected_probability_vectors = []

        candidate_index = 0

        while len(selected_scores) < self.size:
            candidate_score = (
                sorted_scores[
                    candidate_index
                ]
            )

            candidate_id = (
                sorted_ids[
                    candidate_index
                ]
            )

            if candidate_id >= active_candidate_count:
                source_beam = finished_indices[
                    candidate_id
                    - active_candidate_count
                ]

                selected_scores.append(
                    candidate_score
                )

                selected_words.append(
                    torch.tensor(
                        self.eos_token,
                        device=self.device,
                    )
                )

                selected_beams.append(
                    source_beam
                )

                new_finished_status.append(
                    True
                )

                selected_probability_vectors.append(
                    torch.zeros(
                        vocabulary_size,
                        device=self.device,
                    )
                )

            else:
                source_active_index = (
                    candidate_id
                    // vocabulary_size
                )

                source_beam = active_indices[
                    source_active_index
                ]

                word = (
                    candidate_id
                    - source_active_index
                    * vocabulary_size
                )

                # The original COGNet beam search excludes repeated
                # medication tokens within one prescription.
                if word.item() not in (
                    current_output[
                        source_beam
                    ].tolist()
                ):
                    selected_scores.append(
                        candidate_score
                    )

                    selected_words.append(
                        word
                    )

                    selected_beams.append(
                        source_beam
                    )

                    new_finished_status.append(
                        word.item()
                        in {
                            self.eos_token,
                            self.bos_token,
                        }
                    )

                    selected_probability_vectors.append(
                        active_word_log_probs[
                            source_active_index
                        ].detach()
                    )

            candidate_index += 1

            if candidate_index >= len(
                sorted_scores
            ):
                raise RuntimeError(
                    "COGNet beam search could not "
                    "fill the requested beam."
                )

        self.scores = torch.stack(
            selected_scores
        )

        self.prev_ks.append(
            torch.stack(
                selected_beams
            ).long()
        )

        self.next_ys.append(
            torch.stack(
                selected_words
            ).long()
        )

        self.probability_history.append(
            selected_probability_vectors
        )

        self.beam_finished = (
            new_finished_status
        )

        self.done = all(
            self.beam_finished
        )

        return self.done

    def sort_scores(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sort(
            self.scores,
            descending=True,
        )

    def get_hypothesis(
        self,
        k: int | torch.Tensor,
    ) -> list[int]:
        if isinstance(k, torch.Tensor):
            k = int(k.item())

        hypothesis: list[int] = []

        for step in range(
            len(self.prev_ks) - 1,
            -1,
            -1,
        ):
            hypothesis.append(
                int(
                    self.next_ys[
                        step + 1
                    ][k].item()
                )
            )

            k = int(
                self.prev_ks[
                    step
                ][k].item()
            )

        return hypothesis[::-1]

    def get_probability_list(
        self,
        k: int | torch.Tensor,
    ) -> list[torch.Tensor]:
        if isinstance(k, torch.Tensor):
            k = int(k.item())

        probabilities: list[
            torch.Tensor
        ] = []

        for step in range(
            len(self.prev_ks) - 1,
            -1,
            -1,
        ):
            probabilities.append(
                self.probability_history[
                    step
                ][k]
            )

            k = int(
                self.prev_ks[
                    step
                ][k].item()
            )

        return probabilities[::-1]