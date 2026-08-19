from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ehrdrec.models.base import TorchEHRDrecModel
from .layers import RETAINGRU



class RETAIN(TorchEHRDrecModel):
    """
    EHRDRec implementation of RETAIN.

    Ported from the original Theano implementation by Edward Choi et al.

    The model represents every visit as the sum of the embeddings of the
    medical codes present in that visit. Two reverse-time GRUs generate:

        - alpha: scalar visit-level attention weights
        - beta: vector-valued feature-level attention weights

    These are combined with the visit embeddings to form the patient
    context used for prediction.

    Expected EHRDRec input
    ----------------------
    A patient history is a list of visit dictionaries of the form:

        {
            "codes": list[int],
            "time": float,       # required only when use_time=True
        }

    For the original binary RETAIN setting, output_dim should be 1.
    For a multi-label task such as medication recommendation, output_dim
    may instead be the medication vocabulary size; that is an adaptation
    of the original classifier head rather than a property of the source
    implementation itself.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_dim: int = 1,
        embedding_dim: int = 128,
        alpha_hidden_dim: int = 128,
        beta_hidden_dim: int = 128,
        embedding_weights: torch.Tensor | None = None,
        embedding_finetune: bool = True,
        use_time: bool = False,
        use_log_time: bool = True,
        keep_prob_embedding: float = 0.5,
        keep_prob_context: float = 0.5,
        l2_output: float = 0.001,
        l2_embedding: float = 0.001,
        l2_alpha: float = 0.001,
        l2_beta: float = 0.001,
        epochs: int = 10,
        solver: str = "adadelta",
        learning_rate: float | None = None,
        log_eps: float = 1e-8,
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
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        # ============================================================
        # Model configuration
        # ============================================================

        self.input_vocab_size = input_vocab_size
        self.output_dim = output_dim
        self.alpha_hidden_dim = alpha_hidden_dim
        self.beta_hidden_dim = beta_hidden_dim

        self.embedding_finetune = embedding_finetune
        self.use_time = use_time
        self.use_log_time = use_log_time

        self.keep_prob_embedding = keep_prob_embedding
        self.keep_prob_context = keep_prob_context

        self.l2_output = l2_output
        self.l2_embedding = l2_embedding
        self.l2_alpha = l2_alpha
        self.l2_beta = l2_beta

        self.epochs = epochs
        self.solver = solver.lower()

        # Retained for configuration/provenance compatibility with the
        # original implementation. BCE-with-logits is numerically stable
        # and therefore does not require an explicit epsilon.
        self.log_eps = log_eps

        if not 0.0 < keep_prob_embedding <= 1.0:
            raise ValueError("keep_prob_embedding must be in (0, 1].")

        if not 0.0 < keep_prob_context <= 1.0:
            raise ValueError("keep_prob_context must be in (0, 1].")

        # ============================================================
        # Visit embedding
        # ============================================================

        if embedding_weights is not None:
            embedding_weights = torch.as_tensor(
                embedding_weights,
                dtype=torch.float32,
            )

            if embedding_weights.ndim != 2:
                raise ValueError(
                    "embedding_weights must have shape "
                    "(input_vocab_size, embedding_dim)."
                )

            if embedding_weights.shape[0] != input_vocab_size:
                raise ValueError(
                    "embedding_weights first dimension must equal "
                    "input_vocab_size."
                )

            embedding_dim = int(embedding_weights.shape[1])

            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=not embedding_finetune,
            )
        else:
            self.embedding = nn.Embedding(
                input_vocab_size,
                embedding_dim,
            )
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.embedding.weight.requires_grad_(embedding_finetune)

        self.embedding_dim = embedding_dim

        self.embedding_dropout = nn.Dropout(
            p=1.0 - keep_prob_embedding
        )
        self.context_dropout = nn.Dropout(
            p=1.0 - keep_prob_context
        )

        # ============================================================
        # Reverse-time attention networks
        # ============================================================

        gru_input_dim = embedding_dim + (1 if use_time else 0)

        self.alpha_gru = RETAINGRU(
            input_dim=gru_input_dim,
            hidden_dim=alpha_hidden_dim,
        )

        self.beta_gru = RETAINGRU(
            input_dim=gru_input_dim,
            hidden_dim=beta_hidden_dim,
        )

        self.alpha = nn.Linear(
            alpha_hidden_dim,
            1,
        )

        self.beta = nn.Linear(
            beta_hidden_dim,
            embedding_dim,
        )

        # ============================================================
        # Output layer
        # ============================================================

        self.output = nn.Linear(
            embedding_dim,
            output_dim,
        )

        # ============================================================
        # Initialisation
        # ============================================================

        self._init_weights()
        self.to(self.device)

        # ============================================================
        # Optimizer
        # ============================================================

        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

        if self.solver == "adadelta":
            # Matches the original Adadelta update:
            # rho = 0.95, eps = 1e-6, no additional learning-rate scale.
            lr = 1.0 if learning_rate is None else learning_rate
            self.optimizer = torch.optim.Adadelta(
                trainable_parameters,
                lr=lr,
                rho=0.95,
                eps=1e-6,
            )

        elif self.solver == "adam":
            # The original custom Adam implementation corresponds to
            # standard Adam with betas=(0.9, 0.999), eps=1e-8.
            lr = 2e-4 if learning_rate is None else learning_rate
            self.optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        else:
            raise ValueError(
                "solver must be either 'adadelta' or 'adam'."
            )

    # ================================================================
    # Initialisation
    # ================================================================

    def _init_weights(self) -> None:
        """
        Match the original RETAIN parameter initialisation.
        """

        nn.init.uniform_(self.alpha.weight, -0.1, 0.1)
        nn.init.zeros_(self.alpha.bias)

        nn.init.uniform_(self.beta.weight, -0.1, 0.1)
        nn.init.zeros_(self.beta.bias)

        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        nn.init.zeros_(self.output.bias)

    # ================================================================
    # Input representation
    # ================================================================

    def _visit_embeddings(
        self,
        patient_history: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct visit embeddings and optional time values.

        Returns
        -------
        visit_embeddings:
            Tensor of shape (sequence_length, embedding_dim).

        gru_inputs:
            Tensor of shape
            (sequence_length, embedding_dim [+ 1 if time is used]).
        """

        if len(patient_history) == 0:
            raise ValueError("patient_history cannot be empty.")

        visit_embeddings = []
        time_values = []

        for visit in patient_history:
            if "codes" not in visit:
                raise KeyError(
                    "Each RETAIN visit must contain a 'codes' field."
                )

            codes = torch.as_tensor(
                visit["codes"],
                dtype=torch.long,
                device=self.device,
            )

            if codes.numel() == 0:
                visit_embedding = torch.zeros(
                    self.embedding_dim,
                    dtype=self.embedding.weight.dtype,
                    device=self.device,
                )
            else:
                # Equivalent to multiplying the original multi-hot visit
                # vector by W_emb: code embeddings are summed, not averaged.
                visit_embedding = self.embedding(codes).sum(dim=0)

            visit_embeddings.append(visit_embedding)

            if self.use_time:
                if "time" not in visit:
                    raise KeyError(
                        "RETAIN was configured with use_time=True, but a "
                        "visit does not contain a 'time' value."
                    )
                time_values.append(float(visit["time"]))

        visit_embeddings_tensor = torch.stack(
            visit_embeddings,
            dim=0,
        )

        # The source implementation applies embedding dropout before both
        # the GRUs and the final alpha * beta * embedding aggregation.
        visit_embeddings_tensor = self.embedding_dropout(
            visit_embeddings_tensor
        )

        gru_inputs = visit_embeddings_tensor

        if self.use_time:
            times = torch.as_tensor(
                time_values,
                dtype=visit_embeddings_tensor.dtype,
                device=self.device,
            )

            if self.use_log_time:
                times = torch.log1p(times)

            gru_inputs = torch.cat(
                [
                    visit_embeddings_tensor,
                    times.unsqueeze(-1),
                ],
                dim=-1,
            )

        return visit_embeddings_tensor, gru_inputs

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        patient_history: list[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Perform a RETAIN forward pass for one patient history.

        Parameters
        ----------
        patient_history:
            Ordered visit history. Each visit must contain ``codes`` and,
            when ``use_time=True``, ``time``.

        Returns
        -------
        torch.Tensor
            Raw logits with shape (1, output_dim).

        Notes
        -----
        The original Theano code computes a context vector for every prefix
        of a padded sequence and then selects the final valid timestep for
        each patient. EHRDRec supplies one complete task example at a time,
        so computing the context for the complete history is equivalent to
        the final-prefix prediction used by the original implementation.
        """

        embeddings, gru_inputs = self._visit_embeddings(
            patient_history
        )

        # The original implementation runs both GRUs in reverse temporal
        # order and then reverses their hidden-state sequences back to the
        # original visit order.
        reverse_inputs = torch.flip(
            gru_inputs,
            dims=[0],
        ).unsqueeze(1)

        reverse_alpha_hidden = self.alpha_gru(
            reverse_inputs
        )
        reverse_beta_hidden = self.beta_gru(
            reverse_inputs
        )

        alpha_hidden = torch.flip(
            reverse_alpha_hidden,
            dims=[0],
        ).squeeze(1) * 0.5

        beta_hidden = torch.flip(
            reverse_beta_hidden,
            dims=[0],
        ).squeeze(1) * 0.5

        # ------------------------------------------------------------
        # Alpha: visit-level scalar attention
        # ------------------------------------------------------------

        alpha_logits = self.alpha(
            alpha_hidden
        ).squeeze(-1)

        alpha = torch.softmax(
            alpha_logits,
            dim=0,
        )

        # ------------------------------------------------------------
        # Beta: visit-level feature attention
        # ------------------------------------------------------------

        beta = torch.tanh(
            self.beta(beta_hidden)
        )

        # ------------------------------------------------------------
        # Patient context
        # ------------------------------------------------------------

        context = (
            alpha.unsqueeze(-1)
            * beta
            * embeddings
        ).sum(dim=0)

        context = self.context_dropout(
            context
        )

        logits = self.output(
            context
        )

        return logits.unsqueeze(0)

    # ================================================================
    # Batch helper
    # ================================================================

    def _forward_batch(
        self,
        patient_histories: list[list[dict[str, Any]]],
    ) -> torch.Tensor:
        """
        Evaluate a batch of variable-length patient histories.

        The original model pads sequences and evaluates them together.
        EHRDRec can preserve the same model semantics more simply by
        evaluating each variable-length history independently and stacking
        the resulting logits.
        """

        return torch.cat(
            [self.forward(history) for history in patient_histories],
            dim=0,
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
        Train RETAIN using EHRDRec-provided data.

        Expected batches contain:

            batch["x"] -> list of patient histories
            batch["Y"] -> target tensor

        ``resources`` is currently unused because RETAIN does not require
        additional graph or ontology resources.
        """

        for _epoch in range(self.epochs):
            # --------------------------------------------------------
            # Training
            # --------------------------------------------------------

            self.train()

            for batch in train_data:
                patient_histories = batch["x"]
                target = torch.as_tensor(
                    batch["Y"],
                    dtype=torch.float32,
                    device=self.device,
                )

                logits = self._forward_batch(
                    patient_histories
                )

                target = self._normalise_target_shape(
                    target,
                    logits,
                )

                loss = self.loss(
                    logits,
                    target,
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
                    patient_histories = batch["x"]
                    target = torch.as_tensor(
                        batch["Y"],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    logits = self._forward_batch(
                        patient_histories
                    )

                    target = self._normalise_target_shape(
                        target,
                        logits,
                    )

                    validation_predictions.append(
                        logits.detach().cpu()
                    )
                    validation_targets.append(
                        target.detach().cpu()
                    )

            # TODO:
            # The original implementation selects the best epoch using
            # validation ROC-AUC. EHRDRec's standard validation/early-
            # stopping interface should own that policy once it is defined.

    # ================================================================
    # Prediction
    # ================================================================

    def predict(
        self,
        x: list[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Generate raw prediction logits for one patient history.

        Sigmoid conversion, thresholding, and final metric computation are
        intentionally left to EHRDRec.
        """

        self.eval()

        with torch.no_grad():
            logits = self.forward(x)

        return logits

    # ================================================================
    # Loss
    # ================================================================

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the RETAIN objective.

        This matches the source objective:

            binary cross-entropy
            + L2(output weights)
            + L2(alpha weights)
            + L2(beta weights)
            + optional L2(embedding weights)
        """

        prediction_loss = F.binary_cross_entropy_with_logits(
            pred,
            target.float(),
            reduction="mean",
        )

        regularisation = (
            self.l2_output * self.output.weight.pow(2).sum()
            + self.l2_alpha * self.alpha.weight.pow(2).sum()
            + self.l2_beta * self.beta.weight.pow(2).sum()
        )

        if self.embedding_finetune:
            regularisation = (
                regularisation
                + self.l2_embedding
                * self.embedding.weight.pow(2).sum()
            )

        return prediction_loss + regularisation

    @staticmethod
    def _normalise_target_shape(
        target: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalise common target layouts to match the logits.
        """

        if target.ndim == 0:
            target = target.reshape(1, 1)

        elif target.ndim == 1 and logits.ndim == 2:
            if logits.shape[1] == 1:
                target = target.unsqueeze(-1)
            elif logits.shape[0] == 1:
                target = target.unsqueeze(0)

        if target.shape != logits.shape:
            raise ValueError(
                "Target shape does not match RETAIN output shape: "
                f"target={tuple(target.shape)}, "
                f"logits={tuple(logits.shape)}."
            )

        return target

    # ================================================================
    # Saving
    # ================================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the trained RETAIN model state.
        """

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )
