from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ehrdrec.contracts.models import LossOutput, ModelOutput
from ehrdrec.requirements.model import (
    Feature,
    InputRequirement,
    InputStructure,
    Representation,
)
from ehrdrec.models.base import TorchEHRDrecModel
from .layers import RETAINGRU


class RETAIN(TorchEHRDrecModel):
    """
    EHRDRec implementation of RETAIN.

    Based on:
        Choi et al. (2016)
        "RETAIN: An Interpretable Predictive Model for Healthcare
        using Reverse Time Attention Mechanism"

    EHRDRec supplies diagnosis and procedure information as sequences
    of per-visit multi-hot vectors.

    For a patient with T visits:

        diagnoses:
            Tensor[T, diagnoses_vocab_size]

        procedures:
            Tensor[T, procedures_vocab_size]

    RETAIN concatenates the two feature spaces to construct the visit
    vector x_i described in the original paper:

        x_i = [diagnoses_i, procedures_i]

    The visit representation is then:

        v_i = W_emb x_i

    Training, batching, device placement, optimization, prediction,
    evaluation, and checkpointing are handled by EHRDRec.
    """

    _inputs = {
        InputRequirement(
            Feature.DIAGNOSES,
            Representation.MULTI_HOT,
            InputStructure.VISIT_SEQUENCE,
        ),
        InputRequirement(
            Feature.PROCEDURES,
            Representation.MULTI_HOT,
            InputStructure.VISIT_SEQUENCE,
        ),
    }

    _requirements = set()

    def __init__(
        self,
        context,
        embedding_dim: int = 128,
        alpha_hidden_dim: int = 128,
        beta_hidden_dim: int = 128,
        keep_prob_embedding: float = 0.5,
        keep_prob_context: float = 0.5,
        l2_output: float = 0.001,
        l2_embedding: float = 0.001,
        l2_alpha: float = 0.001,
        l2_beta: float = 0.001,
    ) -> None:
        super().__init__(context)

        if not 0.0 < keep_prob_embedding <= 1.0:
            raise ValueError(
                "keep_prob_embedding must be in (0, 1]."
            )

        if not 0.0 < keep_prob_context <= 1.0:
            raise ValueError(
                "keep_prob_context must be in (0, 1]."
            )

        # ============================================================
        # Vocabulary dimensions
        # ============================================================
        if context.vocab.diagnoses is None:
            raise ValueError(
                "RETAIN requires a diagnoses vocabulary."
            )

        if context.vocab.procedures is None:
            raise ValueError(
                "RETAIN requires a procedures vocabulary."
            )

        if context.vocab.medications is None:
            raise ValueError(
                "RETAIN requires a medications vocabulary."
            )

        diagnoses_vocab_size = context.vocab.diagnoses
        procedures_vocab_size = context.vocab.procedures
        medications_vocab_size = context.vocab.medications

        self.input_vocab_size = (
            diagnoses_vocab_size
            + procedures_vocab_size
        )

        self.output_dim = medications_vocab_size
        self.embedding_dim = embedding_dim

        # ============================================================
        # Regularisation configuration
        # ============================================================

        self.l2_output = l2_output
        self.l2_embedding = l2_embedding
        self.l2_alpha = l2_alpha
        self.l2_beta = l2_beta

        # ============================================================
        # Visit embedding
        #
        # Original RETAIN:
        #
        #     v_i = W_emb x_i
        #
        # nn.Linear(..., bias=False) directly implements this.
        # ============================================================

        self.embedding = nn.Linear(
            self.input_vocab_size,
            embedding_dim,
            bias=False,
        )

        self.embedding_dropout = nn.Dropout(
            p=1.0 - keep_prob_embedding,
        )

        self.context_dropout = nn.Dropout(
            p=1.0 - keep_prob_context,
        )

        # ============================================================
        # Reverse-time attention networks
        # ============================================================

        self.alpha_gru = RETAINGRU(
            input_dim=embedding_dim,
            hidden_dim=alpha_hidden_dim,
        )

        self.beta_gru = RETAINGRU(
            input_dim=embedding_dim,
            hidden_dim=beta_hidden_dim,
        )

        # Visit-level scalar attention.
        self.alpha = nn.Linear(
            alpha_hidden_dim,
            1,
        )

        # Feature-level vector attention.
        self.beta = nn.Linear(
            beta_hidden_dim,
            embedding_dim,
        )

        # ============================================================
        # Medication prediction
        # ============================================================

        self.output = nn.Linear(
            embedding_dim,
            medications_vocab_size,
        )

        # ============================================================
        # Initialisation
        # ============================================================

        self._init_weights()

    # ================================================================
    # Initialisation
    # ================================================================

    def _init_weights(self) -> None:
        """
        Initialise RETAIN parameters following the original implementation.
        """

        nn.init.uniform_(
            self.embedding.weight,
            -0.1,
            0.1,
        )

        nn.init.uniform_(
            self.alpha.weight,
            -0.1,
            0.1,
        )
        nn.init.zeros_(
            self.alpha.bias,
        )

        nn.init.uniform_(
            self.beta.weight,
            -0.1,
            0.1,
        )
        nn.init.zeros_(
            self.beta.bias,
        )

        nn.init.uniform_(
            self.output.weight,
            -0.1,
            0.1,
        )
        nn.init.zeros_(
            self.output.bias,
        )

    # ================================================================
    # Patient forward
    # ================================================================

    def _forward_patient(
        self,
        diagnoses: torch.Tensor,
        procedures: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform RETAIN for one patient.

        Parameters
        ----------
        diagnoses:
            Multi-hot diagnosis vectors with shape:

                (num_visits, diagnoses_vocab_size)

        procedures:
            Multi-hot procedure vectors with shape:

                (num_visits, procedures_vocab_size)

        Returns
        -------
        torch.Tensor
            Raw medication logits with shape:

                (1, medications_vocab_size)
        """

        if diagnoses.ndim != 2:
            raise ValueError(
                "RETAIN diagnoses input must have shape "
                "(num_visits, diagnoses_vocab_size)."
            )

        if procedures.ndim != 2:
            raise ValueError(
                "RETAIN procedures input must have shape "
                "(num_visits, procedures_vocab_size)."
            )

        if diagnoses.shape[0] != procedures.shape[0]:
            raise ValueError(
                "Diagnosis and procedure sequences must contain "
                "the same number of visits."
            )

        if diagnoses.shape[0] == 0:
            raise ValueError(
                "RETAIN requires at least one visit."
            )

        # ------------------------------------------------------------
        # Construct original RETAIN visit vector x_i
        # ------------------------------------------------------------

        visits = torch.cat(
            [
                diagnoses.float(),
                procedures.float(),
            ],
            dim=-1,
        )

        # ------------------------------------------------------------
        # Step 1: visit embeddings
        #
        #     v_i = W_emb x_i
        # ------------------------------------------------------------

        embeddings = self.embedding(
            visits,
        )

        embeddings = self.embedding_dropout(
            embeddings,
        )

        # ------------------------------------------------------------
        # Steps 2 and 3:
        # Reverse-time RNNs generating alpha and beta
        # ------------------------------------------------------------

        reverse_embeddings = torch.flip(
            embeddings,
            dims=[0],
        ).unsqueeze(1)

        reverse_alpha_hidden = self.alpha_gru(
            reverse_embeddings,
        )

        reverse_beta_hidden = self.beta_gru(
            reverse_embeddings,
        )

        # Restore chronological visit order.
        alpha_hidden = torch.flip(
            reverse_alpha_hidden,
            dims=[0],
        ).squeeze(1)

        beta_hidden = torch.flip(
            reverse_beta_hidden,
            dims=[0],
        ).squeeze(1)

        # ------------------------------------------------------------
        # Alpha: visit-level attention
        # ------------------------------------------------------------

        alpha_logits = self.alpha(
            alpha_hidden,
        ).squeeze(-1)

        alpha = torch.softmax(
            alpha_logits,
            dim=0,
        )

        # ------------------------------------------------------------
        # Beta: feature-level attention
        # ------------------------------------------------------------

        beta = torch.tanh(
            self.beta(
                beta_hidden,
            )
        )

        # ------------------------------------------------------------
        # Step 4: patient context
        #
        # c = Σ alpha_i * beta_i ⊙ v_i
        # ------------------------------------------------------------

        patient_context = (
            alpha.unsqueeze(-1)
            * beta
            * embeddings
        ).sum(dim=0)

        patient_context = self.context_dropout(
            patient_context,
        )

        # ------------------------------------------------------------
        # Step 5: medication prediction
        # ------------------------------------------------------------

        logits = self.output(
            patient_context,
        )

        return logits.unsqueeze(0)

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        batch: Any,
    ) -> ModelOutput:
        """
        Perform a RETAIN forward pass.

        Expected batch inputs
        ---------------------
        batch.diagnoses:
            List of tensors, one per patient:

                [
                    Tensor[T_1, diagnoses_vocab_size],
                    Tensor[T_2, diagnoses_vocab_size],
                    ...
                ]

        batch.procedures:
            List of tensors, one per patient:

                [
                    Tensor[T_1, procedures_vocab_size],
                    Tensor[T_2, procedures_vocab_size],
                    ...
                ]

        Returns
        -------
        torch.Tensor
            Raw medication logits with shape:

                (batch_size, medications_vocab_size)
        """

        if len(batch.diagnoses) != len(batch.procedures):
            raise ValueError(
                "Diagnosis and procedure batches must contain "
                "the same number of patients."
            )

        patient_logits = [
            self._forward_patient(
                diagnoses,
                procedures,
            )
            for diagnoses, procedures in zip(
                batch.diagnoses,
                batch.procedures,
            )
        ]

        return ModelOutput(
            scores=torch.cat(
                patient_logits,
                dim=0,
            )
        )

    # ================================================================
    # Loss
    # ================================================================

    def loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossOutput:
        base_loss = super().loss(
            outputs=outputs,
            targets=targets,
        )

        regularisation_loss = (
            self.l2_output * self.output.weight.square().sum()
            + self.l2_embedding * self.embedding.weight.square().sum()
            + self.l2_alpha * self.alpha.weight.square().sum()
            + self.l2_beta * self.beta.weight.square().sum()
        )

        return LossOutput(
            total=base_loss.total + regularisation_loss,
            components={
                **base_loss.components,
                "regularisation": regularisation_loss,
            },
        )