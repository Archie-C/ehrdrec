from __future__ import annotations

from pathlib import Path
from typing import Any

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ehrdrec.models.base import TorchEHRDrecModel
from .layers import GCN, MedTransformerDecoder, SelfAttend, Beam


# =====================================================================
# COGNet-local layers
# =====================================================================





# =====================================================================
# COGNet
# =====================================================================


class COGNet(TorchEHRDrecModel):
    """
    EHRDRec implementation of COGNet.

    Based on:
        Rui Wu, Zhaopeng Qiu, Jiacheng Jiang, Guilin Qi, and Xian Wu.
        "Conditional Generation Net for Medication Recommendation."
        WWW 2022.

    COGNet predicts a medication set autoregressively. It combines:

        - Transformer encoders for diagnoses and procedures
        - encoded medication history
        - cross-visit attention
        - EHR and DDI graph memory
        - a copy/generate medication decoder

    EHRDRec-facing patient input
    ----------------------------
    A patient history is an ordered list of visits:

        [
            {
                "diagnoses": list[int],
                "procedures": list[int],
                "medications": list[int],
            },
            ...
        ]

    During training, the medication list for each visit is also the
    autoregressive target sequence. Therefore its ordering matters.
    To reproduce the published implementation, medication sets should
    be converted to the same rare-first target ordering before training.

    The model constructs padding, masks, sequence lengths and shifted
    historical-medication tensors internally. These are model-specific
    implementation details rather than EHRDRec dataset requirements.

    Notes
    -----
    ``forward`` is the teacher-forced training forward pass and returns
    log-probabilities with shape:

        (batch, visits, target_length + 1, medications_vocab_size + 2)

    ``generate`` performs autoregressive beam-search inference.

    ``predict`` returns per-medication score vectors suitable for a common
    evaluator, while the actual generated medication sets are available
    from ``generate``.
    """

    def __init__(
        self,
        diagnoses_vocab_size: int,
        procedures_vocab_size: int,
        medications_vocab_size: int,
        ehr_adj: torch.Tensor,
        ddi_adj: torch.Tensor,
        embedding_dim: int = 64,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        max_len: int = 45,
        beam_size: int = 4,
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
        # Model configuration
        # ============================================================

        self.diagnoses_vocab_size = (
            diagnoses_vocab_size
        )

        self.procedures_vocab_size = (
            procedures_vocab_size
        )

        self.medications_vocab_size = (
            medications_vocab_size
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.epochs = epochs
        self.learning_rate = (
            learning_rate
        )

        self.max_len = max_len
        self.beam_size = beam_size

        self.nhead = 2

        # Medication-only generation tokens.
        self.SOS_TOKEN = (
            medications_vocab_size
        )

        self.END_TOKEN = (
            medications_vocab_size
            + 1
        )

        self.MED_PAD_TOKEN = (
            medications_vocab_size
            + 2
        )

        # COGNet's original input representation reserves the
        # same three extra positions for all code vocabularies.
        self.DIAG_PAD_TOKEN = (
            diagnoses_vocab_size
            + 2
        )

        self.PROC_PAD_TOKEN = (
            procedures_vocab_size
            + 2
        )

        # ============================================================
        # Code embeddings
        # ============================================================

        self.diagnoses_embedding = (
            nn.Sequential(
                nn.Embedding(
                    diagnoses_vocab_size
                    + 3,
                    embedding_dim,
                    padding_idx=(
                        self.DIAG_PAD_TOKEN
                    ),
                ),
                nn.Dropout(0.3),
            )
        )

        self.procedures_embedding = (
            nn.Sequential(
                nn.Embedding(
                    procedures_vocab_size
                    + 3,
                    embedding_dim,
                    padding_idx=(
                        self.PROC_PAD_TOKEN
                    ),
                ),
                nn.Dropout(0.3),
            )
        )

        self.medications_embedding = (
            nn.Sequential(
                nn.Embedding(
                    medications_vocab_size
                    + 3,
                    embedding_dim,
                    padding_idx=(
                        self.MED_PAD_TOKEN
                    ),
                ),
                nn.Dropout(0.3),
            )
        )

        # ============================================================
        # Encoders
        # ============================================================

        self.medication_encoder = (
            nn.TransformerEncoderLayer(
                embedding_dim,
                self.nhead,
                batch_first=True,
                dropout=0.2,
            )
        )

        self.diagnoses_encoder = (
            nn.TransformerEncoderLayer(
                embedding_dim,
                self.nhead,
                batch_first=True,
                dropout=0.2,
            )
        )

        self.procedures_encoder = (
            nn.TransformerEncoderLayer(
                embedding_dim,
                self.nhead,
                batch_first=True,
                dropout=0.2,
            )
        )

        self.diagnoses_self_attend = (
            SelfAttend(
                embedding_dim
            )
        )

        self.procedures_self_attend = (
            SelfAttend(
                embedding_dim
            )
        )

        # ============================================================
        # Graph memory
        # ============================================================

        self.gcn = GCN(
            medications_vocab_size,
            embedding_dim,
            ehr_adj,
            ddi_adj,
        )

        self.inter = nn.Parameter(
            torch.empty(1)
        )

        # Keep the original DDI adjacency available as a fixed resource
        # for evaluation / inspection, even though COGNet's published
        # training loop does not add a separate DDI penalty.
        self.register_buffer(
            "ddi_adj",
            torch.as_tensor(
                ddi_adj,
                dtype=torch.float32,
            ),
        )

        # ============================================================
        # Medication decoder
        # ============================================================

        self.decoder = (
            MedTransformerDecoder(
                embedding_dim=(
                    embedding_dim
                ),
                nhead=self.nhead,
                dim_feedforward=(
                    embedding_dim * 2
                ),
                dropout=0.2,
                layer_norm_eps=1e-5,
            )
        )

        # Generate branch.
        self.Wo = nn.Linear(
            embedding_dim,
            medications_vocab_size
            + 2,
        )

        # Copy branch.
        self.Wc = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Copy/generate switch.
        self.W_z = nn.Linear(
            embedding_dim,
            1,
        )

        # ============================================================
        # Initialisation
        # ============================================================

        nn.init.uniform_(
            self.inter,
            -0.1,
            0.1,
        )

        self.to(
            self.device
        )

        # ============================================================
        # Training components
        # ============================================================

        self.optimizer = (
            torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
            )
        )

    # ================================================================
    # EHRDRec input preparation
    # ================================================================

    @staticmethod
    def _normalise_histories(
        x: Any,
    ) -> list[list[dict[str, list[int]]]]:
        """
        Accept either one patient history or a batch of patient histories.
        """

        if isinstance(x, dict) and "x" in x:
            x = x["x"]

        if not isinstance(x, (list, tuple)):
            raise TypeError(
                "COGNet input must be a patient history "
                "or a batch of patient histories."
            )

        if len(x) == 0:
            raise ValueError(
                "COGNet received an empty input batch."
            )

        # One patient:
        # [
        #   {"diagnoses": ..., "procedures": ..., "medications": ...},
        #   ...
        # ]
        if isinstance(x[0], dict):
            histories = [
                list(x)
            ]

        # Batch:
        # [
        #   [visit, visit, ...],
        #   [visit, visit, ...],
        # ]
        else:
            histories = [
                list(history)
                for history in x
            ]

        for history in histories:
            if len(history) == 0:
                raise ValueError(
                    "COGNet patient histories "
                    "must contain at least one visit."
                )

            for visit in history:
                for field in (
                    "diagnoses",
                    "procedures",
                    "medications",
                ):
                    if field not in visit:
                        raise KeyError(
                            "COGNet visits require "
                            f"the field {field!r}."
                        )

                if not visit["diagnoses"]:
                    raise ValueError(
                        "COGNet requires at least one "
                        "diagnosis code per visit."
                    )

                if not visit["procedures"]:
                    raise ValueError(
                        "COGNet requires at least one "
                        "procedure code per visit."
                    )

        return histories

    def _prepare_histories(
        self,
        x: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Convert semantic EHRDRec visit histories into COGNet's padded batch.
        """

        histories = (
            self._normalise_histories(
                x
            )
        )

        batch_size = len(
            histories
        )

        visit_lengths = torch.tensor(
            [
                len(history)
                for history in histories
            ],
            dtype=torch.long,
            device=self.device,
        )

        max_visits = int(
            visit_lengths.max().item()
        )

        max_diagnoses = max(
            len(visit["diagnoses"])
            for history in histories
            for visit in history
        )

        max_procedures = max(
            len(visit["procedures"])
            for history in histories
            for visit in history
        )

        # A visit may theoretically contain no medications. Keep one
        # allocated position because the Transformer encoder cannot
        # operate on a zero-length sequence.
        max_medications = max(
            1,
            max(
                len(
                    visit[
                        "medications"
                    ]
                )
                for history
                in histories
                for visit
                in history
            ),
        )

        diseases = torch.full(
            (
                batch_size,
                max_visits,
                max_diagnoses,
            ),
            self.DIAG_PAD_TOKEN,
            dtype=torch.long,
            device=self.device,
        )

        procedures = torch.full(
            (
                batch_size,
                max_visits,
                max_procedures,
            ),
            self.PROC_PAD_TOKEN,
            dtype=torch.long,
            device=self.device,
        )

        # Important: the source COGNet implementation pads medication
        # IDs with 0 and masks those positions. MED_PAD_TOKEN cannot be
        # used here because the copy scatter operates over an output
        # vocabulary of medications_vocab_size + 2.
        medications = torch.zeros(
            (
                batch_size,
                max_visits,
                max_medications,
            ),
            dtype=torch.long,
            device=self.device,
        )

        disease_mask = torch.full(
            (
                batch_size,
                max_visits,
                max_diagnoses,
            ),
            -1e9,
            dtype=torch.float32,
            device=self.device,
        )

        procedure_mask = torch.full(
            (
                batch_size,
                max_visits,
                max_procedures,
            ),
            -1e9,
            dtype=torch.float32,
            device=self.device,
        )

        medication_mask = torch.full(
            (
                batch_size,
                max_visits,
                max_medications,
            ),
            -1e9,
            dtype=torch.float32,
            device=self.device,
        )

        diagnosis_lengths = torch.zeros(
            (
                batch_size,
                max_visits,
            ),
            dtype=torch.long,
            device=self.device,
        )

        procedure_lengths = torch.zeros(
            (
                batch_size,
                max_visits,
            ),
            dtype=torch.long,
            device=self.device,
        )

        medication_lengths = torch.zeros(
            (
                batch_size,
                max_visits,
            ),
            dtype=torch.long,
            device=self.device,
        )

        for batch_index, history in enumerate(
            histories
        ):
            for visit_index, visit in enumerate(
                history
            ):
                diagnosis_codes = (
                    torch.as_tensor(
                        visit[
                            "diagnoses"
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                procedure_codes = (
                    torch.as_tensor(
                        visit[
                            "procedures"
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                medication_codes = (
                    torch.as_tensor(
                        visit[
                            "medications"
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                diagnosis_count = int(
                    diagnosis_codes.numel()
                )

                procedure_count = int(
                    procedure_codes.numel()
                )

                medication_count = int(
                    medication_codes.numel()
                )

                diseases[
                    batch_index,
                    visit_index,
                    :diagnosis_count,
                ] = diagnosis_codes

                procedures[
                    batch_index,
                    visit_index,
                    :procedure_count,
                ] = procedure_codes

                disease_mask[
                    batch_index,
                    visit_index,
                    :diagnosis_count,
                ] = 0.0

                procedure_mask[
                    batch_index,
                    visit_index,
                    :procedure_count,
                ] = 0.0

                if medication_count > 0:
                    medications[
                        batch_index,
                        visit_index,
                        :medication_count,
                    ] = medication_codes

                    medication_mask[
                        batch_index,
                        visit_index,
                        :medication_count,
                    ] = 0.0

                diagnosis_lengths[
                    batch_index,
                    visit_index,
                ] = diagnosis_count

                procedure_lengths[
                    batch_index,
                    visit_index,
                ] = procedure_count

                medication_lengths[
                    batch_index,
                    visit_index,
                ] = medication_count

        return {
            "diseases": diseases,
            "procedures": procedures,
            "medications": medications,
            "visit_lengths": (
                visit_lengths
            ),
            "diagnosis_lengths": (
                diagnosis_lengths
            ),
            "procedure_lengths": (
                procedure_lengths
            ),
            "medication_lengths": (
                medication_lengths
            ),
            "disease_mask": (
                disease_mask
            ),
            "procedure_mask": (
                procedure_mask
            ),
            "medication_mask": (
                medication_mask
            ),
        }

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
        Train COGNet.

        Expected EHRDRec batches
        ------------------------
        The DataLoader should yield a dictionary containing:

            batch["x"] -> list of patient histories

        Each patient history follows the semantic representation described
        in the class docstring.

        ``batch["Y"]`` is not required by this implementation because the
        per-visit medication sequence is both historical patient data and
        the teacher-forced target sequence.

        ``resources`` is unused because EHR and DDI graphs are supplied
        when the model is constructed.
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
                x = (
                    batch["x"]
                    if isinstance(
                        batch,
                        dict,
                    )
                    else batch
                )

                prepared = (
                    self._prepare_histories(
                        x
                    )
                )

                log_probabilities = (
                    self._forward_prepared(
                        prepared
                    )
                )

                loss = self.loss(
                    log_probabilities,
                    prepared,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # --------------------------------------------------------
            # Validation
            # --------------------------------------------------------

            self.eval()

            with torch.no_grad():
                validation_losses = []

                for batch in validation_data:
                    x = (
                        batch["x"]
                        if isinstance(
                            batch,
                            dict,
                        )
                        else batch
                    )

                    prepared = (
                        self._prepare_histories(
                            x
                        )
                    )

                    log_probabilities = (
                        self._forward_prepared(
                            prepared
                        )
                    )

                    validation_losses.append(
                        self.loss(
                            log_probabilities,
                            prepared,
                        ).detach().cpu()
                    )

            # TODO:
            # EHRDRec's common validation / model-selection mechanism
            # should eventually own early stopping and checkpoint choice.
            # The source repository selects checkpoints using downstream
            # recommendation metrics rather than validation NLL alone.

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        x: Any,
    ) -> torch.Tensor:
        """
        Teacher-forced COGNet forward pass.

        This method is intended for training / loss computation. For
        inference use ``generate`` or ``predict`` so the target medication
        sequence is not fed into the decoder.
        """

        prepared = (
            self._prepare_histories(
                x
            )
        )

        return self._forward_prepared(
            prepared
        )

    def _forward_prepared(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        diseases = batch[
            "diseases"
        ]

        procedures = batch[
            "procedures"
        ]

        medications = batch[
            "medications"
        ]

        disease_mask = batch[
            "disease_mask"
        ]

        procedure_mask = batch[
            "procedure_mask"
        ]

        medication_mask = batch[
            "medication_mask"
        ]

        (
            input_disease_embedding,
            input_procedure_embedding,
            encoded_medication,
            cross_visit_scores,
            last_seq_medication,
            last_medication_mask,
            drug_memory,
        ) = self._encode_prepared(
            diseases=diseases,
            procedures=procedures,
            medications=medications,
            disease_mask=disease_mask,
            procedure_mask=(
                procedure_mask
            ),
            medication_mask=(
                medication_mask
            ),
        )

        batch_size = (
            medications.size(0)
        )

        max_visits = (
            medications.size(1)
        )

        input_medication = torch.full(
            (
                batch_size,
                max_visits,
                1,
            ),
            self.SOS_TOKEN,
            dtype=torch.long,
            device=self.device,
        )

        input_medication = torch.cat(
            [
                input_medication,
                medications,
            ],
            dim=2,
        )

        sos_mask = torch.zeros(
            (
                batch_size,
                max_visits,
                1,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        decoder_medication_mask = (
            torch.cat(
                [
                    sos_mask,
                    medication_mask,
                ],
                dim=-1,
            )
        )

        return self.decode(
            input_medications=(
                input_medication
            ),
            input_disease_embedding=(
                input_disease_embedding
            ),
            input_procedure_embedding=(
                input_procedure_embedding
            ),
            last_medication_embedding=(
                encoded_medication
            ),
            last_medications=(
                last_seq_medication
            ),
            cross_visit_scores=(
                cross_visit_scores
            ),
            disease_mask=(
                disease_mask
            ),
            procedure_mask=(
                procedure_mask
            ),
            medication_mask=(
                decoder_medication_mask
            ),
            last_medication_mask=(
                last_medication_mask
            ),
            drug_memory=drug_memory,
        )

    # ================================================================
    # Encoder
    # ================================================================

    def _encode_prepared(
        self,
        diseases: torch.Tensor,
        procedures: torch.Tensor,
        medications: torch.Tensor,
        disease_mask: torch.Tensor,
        procedure_mask: torch.Tensor,
        medication_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, max_visits, max_medications = (
            medications.shape
        )

        max_diagnoses = diseases.size(
            2
        )

        max_procedures = procedures.size(
            2
        )

        # ------------------------------------------------------------
        # Diagnosis and procedure encoders
        # ------------------------------------------------------------

        input_disease_embedding = (
            self.diagnoses_embedding(
                diseases
            ).view(
                batch_size
                * max_visits,
                max_diagnoses,
                self.embedding_dim,
            )
        )

        input_procedure_embedding = (
            self.procedures_embedding(
                procedures
            ).view(
                batch_size
                * max_visits,
                max_procedures,
                self.embedding_dim,
            )
        )

        disease_encoder_mask = (
            disease_mask.view(
                batch_size
                * max_visits,
                max_diagnoses,
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(
                1,
                self.nhead,
                max_diagnoses,
                1,
            )
            .view(
                batch_size
                * max_visits
                * self.nhead,
                max_diagnoses,
                max_diagnoses,
            )
        )

        procedure_encoder_mask = (
            procedure_mask.view(
                batch_size
                * max_visits,
                max_procedures,
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(
                1,
                self.nhead,
                max_procedures,
                1,
            )
            .view(
                batch_size
                * max_visits
                * self.nhead,
                max_procedures,
                max_procedures,
            )
        )

        input_disease_embedding = (
            self.diagnoses_encoder(
                input_disease_embedding,
                src_mask=(
                    disease_encoder_mask
                ),
            ).view(
                batch_size,
                max_visits,
                max_diagnoses,
                self.embedding_dim,
            )
        )

        input_procedure_embedding = (
            self.procedures_encoder(
                input_procedure_embedding,
                src_mask=(
                    procedure_encoder_mask
                ),
            ).view(
                batch_size,
                max_visits,
                max_procedures,
                self.embedding_dim,
            )
        )

        # ------------------------------------------------------------
        # Visit representations
        # ------------------------------------------------------------

        visit_diagnosis_embedding = (
            self.diagnoses_self_attend(
                input_disease_embedding.view(
                    batch_size
                    * max_visits,
                    max_diagnoses,
                    self.embedding_dim,
                ),
                disease_mask.view(
                    batch_size
                    * max_visits,
                    max_diagnoses,
                ),
            ).view(
                batch_size,
                max_visits,
                self.embedding_dim,
            )
        )

        visit_procedure_embedding = (
            self.procedures_self_attend(
                input_procedure_embedding.view(
                    batch_size
                    * max_visits,
                    max_procedures,
                    self.embedding_dim,
                ),
                procedure_mask.view(
                    batch_size
                    * max_visits,
                    max_procedures,
                ),
            ).view(
                batch_size,
                max_visits,
                self.embedding_dim,
            )
        )

        cross_visit_scores = (
            self._calc_cross_visit_scores(
                visit_diagnosis_embedding,
                visit_procedure_embedding,
            )
        )

        # ------------------------------------------------------------
        # Previous-visit medication encoding
        # ------------------------------------------------------------

        first_visit_medications = (
            torch.zeros(
                (
                    batch_size,
                    1,
                    max_medications,
                ),
                dtype=torch.long,
                device=self.device,
            )
        )

        last_seq_medication = (
            torch.cat(
                [
                    first_visit_medications,
                    medications[
                        :,
                        :-1,
                        :,
                    ],
                ],
                dim=1,
            )
        )

        first_visit_mask = torch.full(
            (
                batch_size,
                1,
                max_medications,
            ),
            -1e9,
            dtype=torch.float32,
            device=self.device,
        )

        last_medication_mask = (
            torch.cat(
                [
                    first_visit_mask,
                    medication_mask[
                        :,
                        :-1,
                        :,
                    ],
                ],
                dim=1,
            )
        )

        last_medication_embedding = (
            self.medications_embedding(
                last_seq_medication
            )
        )

        last_medication_encoder_mask = (
            last_medication_mask.view(
                batch_size
                * max_visits,
                max_medications,
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(
                1,
                self.nhead,
                max_medications,
                1,
            )
            .view(
                batch_size
                * max_visits
                * self.nhead,
                max_medications,
                max_medications,
            )
        )

        encoded_medication = (
            self.medication_encoder(
                last_medication_embedding.view(
                    batch_size
                    * max_visits,
                    max_medications,
                    self.embedding_dim,
                ),
                src_mask=(
                    last_medication_encoder_mask
                ),
            ).view(
                batch_size,
                max_visits,
                max_medications,
                self.embedding_dim,
            )
        )

        # ------------------------------------------------------------
        # Graph memory
        # ------------------------------------------------------------

        (
            ehr_embedding,
            ddi_embedding,
        ) = self.gcn()

        drug_memory = (
            ehr_embedding
            - ddi_embedding
            * self.inter
        )

        special_token_memory = (
            torch.zeros(
                (
                    3,
                    self.embedding_dim,
                ),
                dtype=drug_memory.dtype,
                device=self.device,
            )
        )

        drug_memory = torch.cat(
            [
                drug_memory,
                special_token_memory,
            ],
            dim=0,
        )

        return (
            input_disease_embedding,
            input_procedure_embedding,
            encoded_medication,
            cross_visit_scores,
            last_seq_medication,
            last_medication_mask,
            drug_memory,
        )

    # ================================================================
    # Decoder
    # ================================================================

    def decode(
        self,
        input_medications: torch.Tensor,
        input_disease_embedding: torch.Tensor,
        input_procedure_embedding: torch.Tensor,
        last_medication_embedding: torch.Tensor,
        last_medications: torch.Tensor,
        cross_visit_scores: torch.Tensor,
        disease_mask: torch.Tensor,
        procedure_mask: torch.Tensor,
        medication_mask: torch.Tensor,
        last_medication_mask: torch.Tensor,
        drug_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode one or more medication positions.

        Returns log-probabilities over:
            medication vocabulary + SOS + EOS
        """

        batch_size = (
            input_medications.size(0)
        )

        max_visits = (
            input_medications.size(1)
        )

        input_medication_count = (
            input_medications.size(2)
        )

        max_diagnoses = (
            input_disease_embedding.size(
                2
            )
        )

        max_procedures = (
            input_procedure_embedding.size(
                2
            )
        )

        input_medication_embeddings = (
            self.medications_embedding(
                input_medications
            ).view(
                batch_size
                * max_visits,
                input_medication_count,
                self.embedding_dim,
            )
        )

        input_medication_memory = (
            drug_memory[
                input_medications
            ].view(
                batch_size
                * max_visits,
                input_medication_count,
                self.embedding_dim,
            )
        )

        medication_self_mask = (
            medication_mask.view(
                batch_size
                * max_visits,
                input_medication_count,
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(
                1,
                self.nhead,
                input_medication_count,
                1,
            )
            .view(
                batch_size
                * max_visits
                * self.nhead,
                input_medication_count,
                input_medication_count,
            )
        )

        medication_to_disease_mask = (
            disease_mask.view(
                batch_size
                * max_visits,
                max_diagnoses,
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(
                1,
                self.nhead,
                input_medication_count,
                1,
            )
            .view(
                batch_size
                * max_visits
                * self.nhead,
                input_medication_count,
                max_diagnoses,
            )
        )

        medication_to_procedure_mask = (
            procedure_mask.view(
                batch_size
                * max_visits,
                max_procedures,
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(
                1,
                self.nhead,
                input_medication_count,
                1,
            )
            .view(
                batch_size
                * max_visits
                * self.nhead,
                input_medication_count,
                max_procedures,
            )
        )

        decoder_hidden = self.decoder(
            input_medication_embedding=(
                input_medication_embeddings
            ),
            input_medication_memory=(
                input_medication_memory
            ),
            input_disease_embedding=(
                input_disease_embedding.view(
                    batch_size
                    * max_visits,
                    max_diagnoses,
                    self.embedding_dim,
                )
            ),
            input_procedure_embedding=(
                input_procedure_embedding.view(
                    batch_size
                    * max_visits,
                    max_procedures,
                    self.embedding_dim,
                )
            ),
            input_medication_self_mask=(
                medication_self_mask
            ),
            disease_mask=(
                medication_to_disease_mask
            ),
            procedure_mask=(
                medication_to_procedure_mask
            ),
        )

        # ------------------------------------------------------------
        # Generate branch
        # ------------------------------------------------------------

        generate_scores = self.Wo(
            decoder_hidden
        )

        generate_scores = (
            generate_scores.view(
                batch_size,
                max_visits,
                input_medication_count,
                self.medications_vocab_size
                + 2,
            )
        )

        generate_probability = (
            F.softmax(
                generate_scores,
                dim=-1,
            )
        )

        # ------------------------------------------------------------
        # Copy branch
        # ------------------------------------------------------------

        copy_probability = (
            self._copy_medication(
                decoder_hidden.view(
                    batch_size,
                    max_visits,
                    input_medication_count,
                    self.embedding_dim,
                ),
                last_medication_embedding,
                last_medication_mask,
                cross_visit_scores,
            )
        )

        copy_probability_to_vocabulary = (
            torch.zeros_like(
                generate_probability
            ).view(
                batch_size,
                max_visits
                * input_medication_count,
                -1,
            )
        )

        copy_source = (
            last_medications.view(
                batch_size,
                1,
                -1,
            )
            .repeat(
                1,
                max_visits
                * input_medication_count,
                1,
            )
        )

        copy_probability_to_vocabulary.scatter_add_(
            2,
            copy_source,
            copy_probability,
        )

        copy_probability_to_vocabulary = (
            copy_probability_to_vocabulary.view(
                batch_size,
                max_visits,
                input_medication_count,
                -1,
            )
        )

        # ------------------------------------------------------------
        # Copy/generate switch
        # ------------------------------------------------------------

        switch_probability = torch.sigmoid(
            self.W_z(
                decoder_hidden
            )
        ).view(
            batch_size,
            max_visits,
            input_medication_count,
            1,
        )

        probability = (
            generate_probability
            * switch_probability
            + copy_probability_to_vocabulary
            * (
                1.0
                - switch_probability
            )
        )

        # There is no previous visit for the first admission.
        probability[
            :,
            0,
            :,
            :,
        ] = generate_probability[
            :,
            0,
            :,
            :,
        ]

        # Avoid log(0) while preserving the source probability model.
        return torch.log(
            probability.clamp_min(
                1e-12
            )
        )

    # ================================================================
    # Attention helpers
    # ================================================================

    def _calc_cross_visit_scores(
        self,
        visit_diagnosis_embedding: torch.Tensor,
        visit_procedure_embedding: torch.Tensor,
    ) -> torch.Tensor:
        max_visits = (
            visit_diagnosis_embedding.size(
                1
            )
        )

        batch_size = (
            visit_diagnosis_embedding.size(
                0
            )
        )

        mask = (
            torch.triu(
                torch.ones(
                    (
                        max_visits,
                        max_visits,
                    ),
                    device=self.device,
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

        mask = mask.unsqueeze(
            0
        ).repeat(
            batch_size,
            1,
            1,
        )

        padding = torch.zeros(
            (
                batch_size,
                1,
                self.embedding_dim,
            ),
            dtype=(
                visit_diagnosis_embedding.dtype
            ),
            device=self.device,
        )

        diagnosis_keys = torch.cat(
            [
                padding,
                visit_diagnosis_embedding[
                    :,
                    :-1,
                    :,
                ],
            ],
            dim=1,
        )

        procedure_keys = torch.cat(
            [
                padding,
                visit_procedure_embedding[
                    :,
                    :-1,
                    :,
                ],
            ],
            dim=1,
        )

        diagnosis_scores = (
            torch.matmul(
                visit_diagnosis_embedding,
                diagnosis_keys.transpose(
                    -2,
                    -1,
                ),
            )
            / math.sqrt(
                visit_diagnosis_embedding.size(
                    -1
                )
            )
        )

        procedure_scores = (
            torch.matmul(
                visit_procedure_embedding,
                procedure_keys.transpose(
                    -2,
                    -1,
                ),
            )
            / math.sqrt(
                visit_procedure_embedding.size(
                    -1
                )
            )
        )

        return F.softmax(
            diagnosis_scores
            + procedure_scores
            + mask,
            dim=-1,
        )

    def _copy_medication(
        self,
        decoder_hidden: torch.Tensor,
        last_medication_embedding: torch.Tensor,
        last_medication_mask: torch.Tensor,
        cross_visit_scores: torch.Tensor,
    ) -> torch.Tensor:
        max_visits = (
            decoder_hidden.size(1)
        )

        input_medication_count = (
            decoder_hidden.size(2)
        )

        max_previous_medications = (
            last_medication_embedding.size(
                2
            )
        )

        copy_query = (
            self.Wc(
                decoder_hidden
            ).view(
                -1,
                max_visits
                * input_medication_count,
                self.embedding_dim,
            )
        )

        attention_scores = (
            torch.matmul(
                copy_query,
                last_medication_embedding.view(
                    -1,
                    max_visits
                    * max_previous_medications,
                    self.embedding_dim,
                ).transpose(
                    -2,
                    -1,
                ),
            )
            / math.sqrt(
                self.embedding_dim
            )
        )

        medication_mask = (
            last_medication_mask.view(
                -1,
                1,
                max_visits
                * max_previous_medications,
            ).repeat(
                1,
                max_visits
                * input_medication_count,
                1,
            )
        )

        attention_scores = F.softmax(
            attention_scores
            + medication_mask,
            dim=-1,
        )

        visit_scores = (
            cross_visit_scores.repeat(
                1,
                1,
                input_medication_count,
            ).view(
                -1,
                max_visits
                * input_medication_count,
                max_visits,
            )
        )

        visit_scores = (
            visit_scores.unsqueeze(
                -1
            )
            .repeat(
                1,
                1,
                1,
                max_previous_medications,
            )
            .view(
                -1,
                max_visits
                * input_medication_count,
                max_visits
                * max_previous_medications,
            )
        )

        scores = (
            attention_scores
            * visit_scores
        ).clamp_min(
            1e-9
        )

        return scores / scores.sum(
            dim=-1,
            keepdim=True,
        )

    # ================================================================
    # Loss
    # ================================================================

    def loss(
        self,
        pred: torch.Tensor,
        target: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        COGNet token-level negative log-likelihood.

        For each valid visit the target is:

            med_1, med_2, ..., med_n, EOS

        Predictions at padded visits and padded medication positions are
        excluded from the objective.
        """

        del kwargs

        if not isinstance(
            target,
            dict,
        ):
            target = (
                self._prepare_histories(
                    target
                )
            )

        medications = target[
            "medications"
        ]

        visit_lengths = target[
            "visit_lengths"
        ]

        medication_lengths = target[
            "medication_lengths"
        ]

        prediction_rows = []
        target_rows = []

        batch_size = (
            medications.size(0)
        )

        for batch_index in range(
            batch_size
        ):
            visit_count = int(
                visit_lengths[
                    batch_index
                ].item()
            )

            for visit_index in range(
                visit_count
            ):
                medication_count = int(
                    medication_lengths[
                        batch_index,
                        visit_index,
                    ].item()
                )

                visit_targets = medications[
                    batch_index,
                    visit_index,
                    :medication_count,
                ]

                eos = torch.tensor(
                    [
                        self.END_TOKEN
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                visit_targets = torch.cat(
                    [
                        visit_targets,
                        eos,
                    ],
                    dim=0,
                )

                visit_predictions = pred[
                    batch_index,
                    visit_index,
                    :(
                        medication_count
                        + 1
                    ),
                    :,
                ]

                prediction_rows.append(
                    visit_predictions
                )

                target_rows.append(
                    visit_targets
                )

        if not prediction_rows:
            raise ValueError(
                "COGNet received no valid "
                "visits for loss computation."
            )

        predictions = torch.cat(
            prediction_rows,
            dim=0,
        )

        targets = torch.cat(
            target_rows,
            dim=0,
        )

        return F.nll_loss(
            predictions,
            targets,
        )

    # ================================================================
    # Inference
    # ================================================================

    def generate(
        self,
        x: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate medication sets using beam search.

        Returns one result per patient:

            {
                "medications": list[list[int]],
                "scores": Tensor[num_visits, medications_vocab_size],
            }

        ``scores`` contains the per-medication score representation used
        for downstream ranking/PRAUC-style evaluation. The generated
        medication sets themselves are the primary discrete predictions.
        """

        self.eval()

        histories = (
            self._normalise_histories(
                x
            )
        )

        outputs = []

        with torch.no_grad():
            for history in histories:
                outputs.append(
                    self._generate_single(
                        history
                    )
                )

        return outputs

    def _generate_single(
        self,
        history: list[
            dict[str, list[int]]
        ],
    ) -> dict[str, Any]:
        prepared = (
            self._prepare_histories(
                history
            )
        )

        diseases = prepared[
            "diseases"
        ]

        procedures = prepared[
            "procedures"
        ]

        medications = prepared[
            "medications"
        ]

        disease_mask = prepared[
            "disease_mask"
        ]

        procedure_mask = prepared[
            "procedure_mask"
        ]

        medication_mask = prepared[
            "medication_mask"
        ]

        (
            input_disease_embedding,
            input_procedure_embedding,
            encoded_medication,
            cross_visit_scores,
            last_seq_medication,
            last_medication_mask,
            drug_memory,
        ) = self._encode_prepared(
            diseases=diseases,
            procedures=procedures,
            medications=medications,
            disease_mask=disease_mask,
            procedure_mask=(
                procedure_mask
            ),
            medication_mask=(
                medication_mask
            ),
        )

        visit_count = len(
            history
        )

        beams = [
            Beam(
                size=self.beam_size,
                bos_token=(
                    self.SOS_TOKEN
                ),
                eos_token=(
                    self.END_TOKEN
                ),
                device=self.device,
            )
            for _ in range(
                visit_count
            )
        ]

        # The original COGNet test routine performs beam search with
        # patient batch size 1 and repeats each encoded visit over the
        # beam dimension.
        input_disease_embedding = (
            input_disease_embedding.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        input_procedure_embedding = (
            input_procedure_embedding.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        encoded_medication = (
            encoded_medication.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        last_seq_medication = (
            last_seq_medication.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        cross_visit_scores = (
            cross_visit_scores.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        disease_mask = (
            disease_mask.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        procedure_mask = (
            procedure_mask.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        last_medication_mask = (
            last_medication_mask.repeat_interleave(
                self.beam_size,
                dim=0,
            )
        )

        drug_memory = (
            drug_memory
        )

        for _step in range(
            self.max_len
        ):
            decoder_input = torch.cat(
                [
                    beam.get_current_state().unsqueeze(
                        1
                    )
                    for beam in beams
                ],
                dim=1,
            )

            decoder_length = (
                decoder_input.size(
                    2
                )
            )

            decoder_medication_mask = (
                torch.zeros(
                    (
                        self.beam_size,
                        visit_count,
                        decoder_length,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

            log_probabilities = (
                self.decode(
                    input_medications=(
                        decoder_input
                    ),
                    input_disease_embedding=(
                        input_disease_embedding
                    ),
                    input_procedure_embedding=(
                        input_procedure_embedding
                    ),
                    last_medication_embedding=(
                        encoded_medication
                    ),
                    last_medications=(
                        last_seq_medication
                    ),
                    cross_visit_scores=(
                        cross_visit_scores
                    ),
                    disease_mask=(
                        disease_mask
                    ),
                    procedure_mask=(
                        procedure_mask
                    ),
                    medication_mask=(
                        decoder_medication_mask
                    ),
                    last_medication_mask=(
                        last_medication_mask
                    ),
                    drug_memory=(
                        drug_memory
                    ),
                )
            )

            next_token_log_probabilities = (
                log_probabilities[
                    :,
                    :,
                    -1,
                    :,
                ]
            )

            all_done = True

            for visit_index, beam in enumerate(
                beams
            ):
                beam.advance(
                    next_token_log_probabilities[
                        :,
                        visit_index,
                        :,
                    ]
                )

                if not beam.done:
                    all_done = False

            if all_done:
                break

        generated_visits: list[
            list[int]
        ] = []

        visit_scores = torch.zeros(
            (
                visit_count,
                self.medications_vocab_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for visit_index, beam in enumerate(
            beams
        ):
            _, indices = (
                beam.sort_scores()
            )

            best_index = indices[
                0
            ]

            hypothesis = (
                beam.get_hypothesis(
                    best_index
                )
            )

            probability_vectors = (
                beam.get_probability_list(
                    best_index
                )
            )

            medications_out = []
            medication_probability_vectors = []

            for token, log_probability_vector in zip(
                hypothesis,
                probability_vectors,
            ):
                if token in {
                    self.SOS_TOKEN,
                    self.END_TOKEN,
                }:
                    break

                if token < (
                    self.medications_vocab_size
                ):
                    medications_out.append(
                        token
                    )

                    medication_probability_vectors.append(
                        torch.exp(
                            log_probability_vector[
                                :self.medications_vocab_size
                            ]
                        )
                    )

            generated_visits.append(
                medications_out
            )

            if medication_probability_vectors:
                probability_matrix = (
                    torch.stack(
                        medication_probability_vectors,
                        dim=0,
                    )
                )

                # Match the source test-time construction in spirit:
                # unselected medications receive their maximum probability
                # across decoding steps; generated medications receive their
                # probability at the step where they were selected.
                score_vector = (
                    probability_matrix.max(
                        dim=0
                    ).values
                )

                for step_index, medication in enumerate(
                    medications_out
                ):
                    score_vector[
                        medication
                    ] = (
                        probability_matrix[
                            step_index,
                            medication,
                        ]
                    )

                visit_scores[
                    visit_index
                ] = score_vector

        return {
            "medications": (
                generated_visits
            ),
            "scores": visit_scores,
        }

    def predict(
        self,
        x: Any,
    ) -> torch.Tensor:
        """
        Return per-medication prediction scores.

        For one patient:
            Tensor[num_visits, medications_vocab_size]

        For multiple patients:
            Tensor[batch, max_visits, medications_vocab_size]

        The actual autoregressively generated medication sets are available
        through ``generate``.
        """

        results = self.generate(
            x
        )

        if len(results) == 1:
            return results[
                0
            ]["scores"]

        max_visits = max(
            result["scores"].size(0)
            for result in results
        )

        output = torch.zeros(
            (
                len(results),
                max_visits,
                self.medications_vocab_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for index, result in enumerate(
            results
        ):
            visit_count = (
                result["scores"].size(0)
            )

            output[
                index,
                :visit_count,
                :,
            ] = result[
                "scores"
            ]

        return output

    # ================================================================
    # Saving
    # ================================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the trained COGNet model state.
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


