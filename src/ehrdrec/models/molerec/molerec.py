from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ehrdrec.data.requirements import ModelRequirement
from ehrdrec.models.base import TorchEHRDrecModel

from .layers import AdjAttenAgger, GNNGraph, SAB


class MoleRec(TorchEHRDrecModel):
    """
    EHRDRec implementation of MoleRec.

    Based on the original implementation of:

        MoleRec: Combinatorial Drug Recommendation with
        Substructure-Aware Molecular Representation Learning
        Yang et al., WWW 2023.

    Patient input
    -------------
    A patient history is an ordered list of visits:

        [
            {
                "diagnoses": list[int],
                "procedures": list[int],
            },
            ...
        ]

    ``medications`` may also be present in the visit dictionaries because
    the common EHRDRec patient representation may contain it, but MoleRec
    does not use historical medication codes when constructing the patient
    representation.

    The prediction target is the current visit's medication multi-hot vector:

        batch["Y"] -> Tensor[medications_vocab_size]

    Molecular resources
    -------------------
    MoleRec additionally requires:

        - molecular graph data for the medication molecules
        - medication-to-molecule average projection matrix
        - medication-to-substructure incidence matrix (drug_fragment_mask)
        - DDI adjacency matrix
        - optionally substructure graph data

    If ``use_embedding=True``, MoleRec learns a free embedding table for
    substructures and does not require substructure graph data.

    Output
    ------
    Raw medication logits with shape:

        (1, medications_vocab_size)

    Sigmoid conversion and recommendation thresholding are intentionally
    left to the common EHRDRec evaluation layer.
    """

    requirements = {
        ModelRequirement.DIAGNOSES,
        ModelRequirement.PROCEDURES,
        ModelRequirement.MOLECULAR_GRAPHS,
        ModelRequirement.MEDICATION_MOLECULE_PROJECTION,
        ModelRequirement.MEDICATION_SUBSTRUCTURE_MATRIX,
        ModelRequirement.SUBSTRUCTURE_GRAPHS,
        ModelRequirement.DDI_GRAPH,
    }

    def __init__(
        self,
        diagnoses_vocab_size: int,
        procedures_vocab_size: int,
        medications_vocab_size: int,
        molecule_data: dict[str, Any],
        drug_fragment_mask: torch.Tensor,
        ddi_adj: torch.Tensor,
        average_projection: torch.Tensor,
        global_para: dict[str, Any],
        substruct_data: dict[str, Any] | None = None,
        substruct_para: dict[str, Any] | None = None,
        embedding_dim: int = 64,
        global_dim: int | None = None,
        substruct_dim: int | None = None,
        use_embedding: bool = False,
        dropout: float = 0.7,
        epochs: int = 50,
        learning_rate: float = 5e-4,
        target_ddi: float = 0.06,
        ddi_annealing_coef: float = 2.5,
        ddi_loss_weight: float = 0.0005,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        # ============================================================
        # Runtime
        # ============================================================

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        # ============================================================
        # Configuration
        # ============================================================

        self.diagnoses_vocab_size = diagnoses_vocab_size
        self.procedures_vocab_size = procedures_vocab_size
        self.medications_vocab_size = medications_vocab_size

        self.embedding_dim = embedding_dim
        self.global_dim = (
            embedding_dim
            if global_dim is None
            else global_dim
        )
        self.substruct_dim = (
            embedding_dim
            if substruct_dim is None
            else substruct_dim
        )

        self.use_embedding = use_embedding
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.target_ddi = target_ddi
        self.ddi_annealing_coef = ddi_annealing_coef
        self.ddi_loss_weight = ddi_loss_weight

        # ============================================================
        # Fixed molecular / DDI resources
        # ============================================================

        drug_fragment_mask = torch.as_tensor(
            drug_fragment_mask,
            dtype=torch.float32,
        )

        ddi_adj = torch.as_tensor(
            ddi_adj,
            dtype=torch.float32,
        )

        average_projection = torch.as_tensor(
            average_projection,
            dtype=torch.float32,
        )

        if drug_fragment_mask.ndim != 2:
            raise ValueError(
                "drug_fragment_mask must have shape "
                "(medications, substructures)."
            )

        if drug_fragment_mask.shape[0] != medications_vocab_size:
            raise ValueError(
                "drug_fragment_mask first dimension must equal "
                "medications_vocab_size."
            )

        if ddi_adj.shape != (
            medications_vocab_size,
            medications_vocab_size,
        ):
            raise ValueError(
                "ddi_adj must have shape "
                "(medications_vocab_size, medications_vocab_size)."
            )

        if average_projection.shape[0] != medications_vocab_size:
            raise ValueError(
                "average_projection first dimension must equal "
                "medications_vocab_size."
            )

        self.substructure_count = int(
            drug_fragment_mask.shape[1]
        )

        self.register_buffer(
            "drug_fragment_mask",
            drug_fragment_mask,
        )

        self.register_buffer(
            "ddi_adj",
            ddi_adj,
        )

        self.register_buffer(
            "average_projection",
            average_projection,
        )

        # PyG-style graph batch objects are not nn.Module buffers, so retain
        # them as model resources and move them onto the configured device.
        self.molecule_data = self._move_resource_dict(
            molecule_data
        )

        self.substruct_data = (
            None
            if substruct_data is None
            else self._move_resource_dict(
                substruct_data
            )
        )

        # ============================================================
        # Molecular encoders
        # ============================================================

        self.global_encoder = GNNGraph(
            **global_para
        )

        if self.use_embedding:
            self.substruct_embedding = nn.Parameter(
                torch.zeros(
                    self.substructure_count,
                    embedding_dim,
                )
            )
            self.substruct_encoder = None
        else:
            if substruct_para is None:
                raise ValueError(
                    "substruct_para is required when "
                    "use_embedding=False."
                )

            if self.substruct_data is None:
                raise ValueError(
                    "substruct_data is required when "
                    "use_embedding=False."
                )

            self.substruct_embedding = None
            self.substruct_encoder = GNNGraph(
                **substruct_para
            )

        # ============================================================
        # Patient representation
        # ============================================================

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    diagnoses_vocab_size,
                    embedding_dim,
                ),
                nn.Embedding(
                    procedures_vocab_size,
                    embedding_dim,
                ),
            ]
        )

        self.sequence_encoders = nn.ModuleList(
            [
                nn.GRU(
                    embedding_dim,
                    embedding_dim,
                    batch_first=True,
                ),
                nn.GRU(
                    embedding_dim,
                    embedding_dim,
                    batch_first=True,
                ),
            ]
        )

        if 0.0 < dropout < 1.0:
            self.rnn_dropout = nn.Dropout(
                p=dropout
            )
        else:
            self.rnn_dropout = nn.Identity()

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                embedding_dim * 4,
                embedding_dim,
            ),
        )

        # ============================================================
        # Substructure-aware medication representation
        # ============================================================

        self.substructure_attention = SAB(
            self.substruct_dim,
            self.substruct_dim,
            2,
            use_ln=True,
        )

        self.substructure_relevance = nn.Linear(
            embedding_dim,
            self.substructure_count,
        )

        self.aggregator = AdjAttenAgger(
            self.global_dim,
            self.substruct_dim,
            max(
                self.global_dim,
                self.substruct_dim,
            ),
        )

        self.score_extractor = nn.Sequential(
            nn.Linear(
                self.substruct_dim,
                self.substruct_dim // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                self.substruct_dim // 2,
                1,
            ),
        )

        self._initialise_weights()

        self.to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

    # ================================================================
    # Initialisation / resources
    # ================================================================

    def _initialise_weights(
        self,
    ) -> None:
        init_range = 0.1

        for embedding in self.embeddings:
            embedding.weight.data.uniform_(
                -init_range,
                init_range,
            )

        if self.substruct_embedding is not None:
            nn.init.xavier_uniform_(
                self.substruct_embedding
            )

    def _move_resource_dict(
        self,
        resource: dict[str, Any],
    ) -> dict[str, Any]:
        moved: dict[str, Any] = {}

        for key, value in resource.items():
            if hasattr(value, "to"):
                moved[key] = value.to(
                    self.device
                )
            else:
                moved[key] = value

        return moved

    # ================================================================
    # Patient input handling
    # ================================================================

    @staticmethod
    def _normalise_history(
        patient_history: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(
            patient_history,
            (list, tuple),
        ):
            raise TypeError(
                "MoleRec input must be an ordered "
                "list of visit dictionaries."
            )

        history = list(
            patient_history
        )

        if not history:
            raise ValueError(
                "MoleRec requires at least one visit."
            )

        for visit in history:
            if not isinstance(
                visit,
                dict,
            ):
                raise TypeError(
                    "Each MoleRec visit must be a dictionary."
                )

            if "diagnoses" not in visit:
                raise KeyError(
                    "Each MoleRec visit requires "
                    "'diagnoses'."
                )

            if "procedures" not in visit:
                raise KeyError(
                    "Each MoleRec visit requires "
                    "'procedures'."
                )

            if not visit["diagnoses"]:
                raise ValueError(
                    "MoleRec requires at least one "
                    "diagnosis code per visit."
                )

            if not visit["procedures"]:
                raise ValueError(
                    "MoleRec requires at least one "
                    "procedure code per visit."
                )

        return history

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        x: list[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Predict medication logits for the final visit represented by ``x``.

        ``x`` contains the longitudinal diagnosis and procedure history up to
        and including the target visit.
        """

        patient_history = self._normalise_history(
            x
        )

        diagnosis_sequence = []
        procedure_sequence = []

        for visit in patient_history:
            diagnosis_indices = torch.as_tensor(
                visit["diagnoses"],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            procedure_indices = torch.as_tensor(
                visit["procedures"],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            diagnosis_repr = self.rnn_dropout(
                self.embeddings[0](
                    diagnosis_indices
                )
            )

            procedure_repr = self.rnn_dropout(
                self.embeddings[1](
                    procedure_indices
                )
            )

            diagnosis_sequence.append(
                torch.sum(
                    diagnosis_repr,
                    keepdim=True,
                    dim=1,
                )
            )

            procedure_sequence.append(
                torch.sum(
                    procedure_repr,
                    keepdim=True,
                    dim=1,
                )
            )

        diagnosis_sequence_tensor = torch.cat(
            diagnosis_sequence,
            dim=1,
        )

        procedure_sequence_tensor = torch.cat(
            procedure_sequence,
            dim=1,
        )

        (
            diagnosis_outputs,
            diagnosis_hidden,
        ) = self.sequence_encoders[0](
            diagnosis_sequence_tensor
        )

        (
            procedure_outputs,
            procedure_hidden,
        ) = self.sequence_encoders[1](
            procedure_sequence_tensor
        )

        sequence_repr = torch.cat(
            [
                diagnosis_hidden,
                procedure_hidden,
            ],
            dim=-1,
        )

        last_visit_repr = torch.cat(
            [
                diagnosis_outputs[:, -1],
                procedure_outputs[:, -1],
            ],
            dim=-1,
        )

        patient_repr = torch.cat(
            [
                sequence_repr.flatten(),
                last_visit_repr.flatten(),
            ]
        )

        query = self.query(
            patient_repr
        )

        substructure_weights = torch.sigmoid(
            self.substructure_relevance(
                query
            )
        )

        # ------------------------------------------------------------
        # Medication molecule embeddings
        # ------------------------------------------------------------

        global_embeddings = self.global_encoder(
            **self.molecule_data
        )

        global_embeddings = torch.mm(
            self.average_projection,
            global_embeddings,
        )

        # ------------------------------------------------------------
        # Substructure embeddings
        # ------------------------------------------------------------

        if self.use_embedding:
            assert self.substruct_embedding is not None
            substructure_embeddings = (
                self.substruct_embedding
            )
        else:
            assert self.substruct_encoder is not None
            assert self.substruct_data is not None

            substructure_embeddings = (
                self.substruct_encoder(
                    **self.substruct_data
                )
            )

        substructure_embeddings = (
            self.substructure_attention(
                substructure_embeddings.unsqueeze(
                    0
                )
            ).squeeze(0)
        )

        # ------------------------------------------------------------
        # Patient-conditioned medication representations
        # ------------------------------------------------------------

        medication_embeddings = self.aggregator(
            global_embeddings,
            substructure_embeddings,
            substructure_weights,
            mask=torch.logical_not(
                self.drug_fragment_mask > 0
            ),
        )

        logits = self.score_extractor(
            medication_embeddings
        ).t()

        return logits

    def _forward_batch(
        self,
        patient_histories: list[
            list[dict[str, Any]]
        ],
    ) -> torch.Tensor:
        """
        MoleRec's source implementation evaluates one longitudinal patient
        prefix at a time. Preserve that behaviour while allowing EHRDRec
        DataLoaders to contain more than one example.
        """

        return torch.cat(
            [
                self.forward(history)
                for history in patient_histories
            ],
            dim=0,
        )

    # ================================================================
    # Prediction
    # ================================================================

    def predict(
        self,
        x: list[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Return raw medication logits for one patient history.

        The common evaluator should apply sigmoid / thresholding.
        """

        self.eval()

        with torch.no_grad():
            return self.forward(
                x
            )

    # ================================================================
    # Original MoleRec objective
    # ================================================================

    def _ddi_penalty(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable DDI penalty from the original MoleRec model.
        """

        probabilities = torch.sigmoid(
            logits
        )

        pair_probabilities = (
            probabilities.unsqueeze(-1)
            * probabilities.unsqueeze(-2)
        )

        return (
            self.ddi_loss_weight
            * pair_probabilities.mul(
                self.ddi_adj
            ).sum()
        )

    def _predicted_ddi_rate(
        self,
        logits: torch.Tensor,
    ) -> float:
        """
        DDI rate of recommendations obtained with the source threshold 0.5.

        Used only to choose the annealing weight; gradients do not flow
        through this quantity.
        """

        prediction = (
            torch.sigmoid(logits)
            >= 0.5
        )

        rates: list[float] = []

        ddi_upper = torch.triu(
            self.ddi_adj,
            diagonal=1,
        )

        for row in prediction:
            selected = torch.nonzero(
                row,
                as_tuple=False,
            ).flatten()

            count = int(
                selected.numel()
            )

            if count < 2:
                rates.append(
                    0.0
                )
                continue

            possible_pairs = (
                count * (count - 1) / 2
            )

            submatrix = ddi_upper[
                selected
            ][:, selected]

            interacting_pairs = float(
                (submatrix > 0)
                .sum()
                .item()
            )

            rates.append(
                interacting_pairs
                / possible_pairs
            )

        return (
            sum(rates) / len(rates)
            if rates
            else 0.0
        )

    @staticmethod
    def _multilabel_margin_target(
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a multi-hot target to the target representation expected by
        torch.nn.functional.multilabel_margin_loss.
        """

        if target.ndim == 1:
            target = target.unsqueeze(
                0
            )

        batch_size, output_dim = (
            target.shape
        )

        margin_target = torch.full(
            (
                batch_size,
                output_dim,
            ),
            -1,
            dtype=torch.long,
            device=target.device,
        )

        for row_index in range(
            batch_size
        ):
            positive_indices = torch.nonzero(
                target[row_index] > 0,
                as_tuple=False,
            ).flatten()

            margin_target[
                row_index,
                :positive_indices.numel(),
            ] = positive_indices

        return margin_target

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Original MoleRec training objective.

        Accuracy objective:
            0.95 * BCE + 0.05 * multilabel-margin loss

        If the thresholded recommendation's DDI rate is above ``target_ddi``,
        MoleRec anneals between the accuracy objective and its differentiable
        DDI penalty.
        """

        del kwargs

        target = torch.as_tensor(
            target,
            dtype=torch.float32,
            device=self.device,
        )

        if target.ndim == 1:
            target = target.unsqueeze(
                0
            )

        if pred.shape != target.shape:
            raise ValueError(
                "MoleRec target shape must match "
                "the medication logits: "
                f"pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}."
            )

        loss_bce = (
            F.binary_cross_entropy_with_logits(
                pred,
                target,
            )
        )

        sigmoid_pred = torch.sigmoid(
            pred
        )

        margin_target = (
            self._multilabel_margin_target(
                target
            )
        )

        loss_multi = F.multilabel_margin_loss(
            sigmoid_pred,
            margin_target,
        )

        accuracy_loss = (
            0.95 * loss_bce
            + 0.05 * loss_multi
        )

        current_ddi_rate = (
            self._predicted_ddi_rate(
                pred.detach()
            )
        )

        if (
            current_ddi_rate
            <= self.target_ddi
        ):
            return accuracy_loss

        ddi_loss = self._ddi_penalty(
            pred
        )

        beta = (
            self.ddi_annealing_coef
            * (
                1.0
                - current_ddi_rate
                / self.target_ddi
            )
        )

        beta = min(
            math.exp(beta),
            1.0,
        )

        return (
            beta * accuracy_loss
            + (1.0 - beta)
            * ddi_loss
        )

    # ================================================================
    # Training
    # ================================================================

    def fit(
        self,
        train_data: DataLoader,
        validation_data: DataLoader,
        resources: dict[str, Any] | None = None,
    ) -> None:
        """
        Train MoleRec using EHRDRec batches.

        Expected batch structure:

            batch["x"]:
                list of patient histories, where each history is:
                    [
                        {
                            "diagnoses": list[int],
                            "procedures": list[int],
                        },
                        ...
                    ]

            batch["Y"]:
                multi-hot medication targets with shape:
                    (batch, medications_vocab_size)

        Molecular and DDI resources are supplied to the constructor, so the
        generic ``resources`` argument is unused.
        """

        del resources

        for _epoch in range(
            self.epochs
        ):
            # --------------------------------------------------------
            # Training
            # --------------------------------------------------------

            self.train()

            for batch in train_data:
                patient_histories = batch[
                    "x"
                ]

                targets = torch.as_tensor(
                    batch["Y"],
                    dtype=torch.float32,
                    device=self.device,
                )

                if targets.ndim == 1:
                    targets = targets.unsqueeze(
                        0
                    )

                if len(
                    patient_histories
                ) != targets.shape[0]:
                    raise ValueError(
                        "MoleRec batch contains a different "
                        "number of patient histories and targets."
                    )

                # The source implementation updates once per target visit.
                # Preserve that behaviour even when the EHRDRec DataLoader
                # groups several examples into one batch.
                for history, target in zip(
                    patient_histories,
                    targets,
                ):
                    logits = self.forward(
                        history
                    )

                    loss = self.loss(
                        logits,
                        target.unsqueeze(0),
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # --------------------------------------------------------
            # Validation
            # --------------------------------------------------------

            self.eval()

            validation_predictions = []
            validation_targets = []

            with torch.no_grad():
                for batch in validation_data:
                    patient_histories = batch[
                        "x"
                    ]

                    targets = torch.as_tensor(
                        batch["Y"],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    if targets.ndim == 1:
                        targets = targets.unsqueeze(
                            0
                        )

                    logits = self._forward_batch(
                        patient_histories
                    )

                    validation_predictions.append(
                        logits.detach().cpu()
                    )

                    validation_targets.append(
                        targets.detach().cpu()
                    )

            # Model selection / early stopping is intentionally not performed
            # here. The common EHRDRec experiment layer should own that policy
            # so all models are selected under the same validation protocol.

    # ================================================================
    # Saving
    # ================================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the trained MoleRec state.
        """

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )
