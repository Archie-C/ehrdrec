
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.data.requirements import ModelRequirement
from ehrdrec.models.base import TorchEHRDrecModel


class MICRON(TorchEHRDrecModel):
    """
    EHRDRec implementation of MICRON.

    Based on the original implementation:
    https://github.com/ycq091044/MICRON/blob/main/src/models.py

    MICRON uses:
    ...

    Notes
    -----
    This implementation intentionally keeps the model-specific architecture
    close to the original MICRON implementation while exposing training and
    prediction through the EHRDRec model interface.
    """

    requirements = {
        ModelRequirement.DIAGNOSES,
        ModelRequirement.PROCEDURES,
        ModelRequirement.DDI_GRAPH,
    }
    
    def __init__(
        self,
        diagnoses_vocab_size: int,
        procedures_vocab_size: int,
        medications_vocab_size: int,
        ddi_adj: torch.Tensor,
        embedding_dim: int = 256,
        dropout: float = 0.5,
        epochs: int = 40,
        learning_rate: float = 2e-4,
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
        
        self.diagnoses_vocab_size = diagnoses_vocab_size
        self.procedures_vocab_size = procedures_vocab_size
        self.medications_vocab_size = medications_vocab_size
        
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.learning_rate = learning_rate

        # ============================================================
        # Patient representation
        # ============================================================
        
        self.diagnoses_embedding = nn.Embedding(
            diagnoses_vocab_size,
            embedding_dim,
        )

        self.procedures_embedding = nn.Embedding(
            procedures_vocab_size,
            embedding_dim,
        )

        self.dropout = nn.Dropout(p=dropout)
        
        self.health_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

        
        # ============================================================
        # Medication prediction layer
        # ============================================================

        self.output = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, medications_vocab_size),
        )

        # ============================================================
        # Fixed model resources
        # ============================================================
        
        self.register_buffer(
            "ddi_adj",
            torch.as_tensor(
                ddi_adj,
                dtype=torch.float32,
            ),
        )
        
        # ============================================================
        # Initialisation
        # ============================================================

        self._init_weights()
        self.to(self.device)

        # ============================================================
        # Training components
        # ============================================================

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        
        
    # ================================================================
    # Initialisation
    # ================================================================

    def _init_weights(self) -> None:
        """
        Initialise the diagnosis and procedure embeddings.
        """

        init_range = 0.1

        nn.init.uniform_(
            self.diagnoses_embedding.weight,
            -init_range,
            init_range,
        )

        nn.init.uniform_(
            self.procedures_embedding.weight,
            -init_range,
            init_range,
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
        Train the MICRON model.

        Parameters
        ----------
        train_data:
            Training data provided by EHRDRec.

        validation_data:
            Validation data provided by EHRDRec.

        resources:
            Additional resources supplied by EHRDRec.

            MICRON currently receives its required resources during
            construction, so this argument is unused. It remains here for
            compatibility with the current EHRDRec model interface.
        """

        for epoch in range(self.epochs):

            # --------------------------------------------------------
            # Training
            # --------------------------------------------------------

            self.train()

            for batch in train_data:

                x = batch["x"]
                target = batch["Y"].to(self.device)

                logits, ddi_loss = self.forward(x)

                loss = self.loss(
                    logits,
                    target,
                    ddi_loss=ddi_loss,
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

                    x = batch["x"]
                    target = batch["Y"].to(self.device)

                    logits = self.forward(x)

                    validation_predictions.append(
                        logits.detach().cpu()
                    )

                    validation_targets.append(
                        target.detach().cpu()
                    )

            # TODO:
            # Validation metric computation and early stopping should
            # eventually be handled through a standard EHRDRec mechanism.
            #
            # For example:
            #
            # metrics = self.metrics.compute(
            #     validation_predictions,
            #     validation_targets,
            # )
            #
            # if self.early_stopping(metrics):
            #     break

    # ================================================================
    # Forward
    # ================================================================
    
    def forward(
        self,
        patient_history: list[dict[str, list[int]]],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a MICRON forward pass.

        Parameters
        ----------
        patient_history:
            Longitudinal sequence of patient admissions.

            Each admission is expected to contain:

                {
                    "diagnoses": list[int],
                    "procedures": list[int]
                }

        Returns
        -------
        During training:
            (logits, ddi_loss)

        During evaluation:
            logits
        """

        def sum_embedding(
            embedding: torch.Tensor,
        ) -> torch.Tensor:
            """
            Sum of code embeddings within a single visit.

            Input:
                (1, n_codes, embedding_dim)

            Output:
                (1, 1, embedding_dim)
            """
            return embedding.sum(dim=1).unsqueeze(dim=0)
        
        current_diagnoses_embedding = sum_embedding(
            self.dropout(
                self.diagnoses_embedding(
                    torch.as_tensor(
                        patient_history[-1]["diagnoses"],
                        dtype=torch.long,
                        device=self.device,
                    ).unsqueeze(0)
                )
            )
        )
        
        current_procedures_embedding = sum_embedding(
            self.dropout(
                self.procedures_embedding(
                    torch.as_tensor(
                        patient_history[-1]["procedures"],
                        dtype=torch.long,
                        device=self.device,
                    ).unsqueeze(0) 
                )
            )
        )
        
        if len(patient_history) == 1:
            diagnoses_embedding_last = current_diagnoses_embedding * torch.tensor(0.0)
            procedures_embedding_last = current_procedures_embedding * torch.tensor(0.0)
        else:
            diagnoses_embedding_last = sum_embedding(
                self.dropout(
                    self.diagnoses_embedding(
                        torch.as_tensor(
                            patient_history[-2]["diagnoses"],
                            dtype=torch.long,
                            device=self.device,
                        ).unsqueeze(0)
                    )
                )
            )

            procedures_embedding_last = sum_embedding(
                self.dropout(
                    self.procedures_embedding(
                        torch.as_tensor(
                            patient_history[-2]["procedures"],
                            dtype=torch.long,
                            device=self.device,
                        ).unsqueeze(0)
                    )
                )
            )
        
        # ------------------------------------------------------------
        # Health representation
        # ------------------------------------------------------------
        
        health_representation = torch.cat(
            [
                current_diagnoses_embedding, 
                current_procedures_embedding
            ],
            dim=-1
        ).squeeze(dim=0)
        
        health_representation_last = torch.cat(
            [
                diagnoses_embedding_last,
                procedures_embedding_last
            ],
            dim=-1
        ).squeeze(dim=0)
        
        health_representation = self.health_net(
            health_representation
        )[-1:, :]
        
        health_representation_last = self.health_net(
            health_representation_last
        )[-1:, :]
        
        health_residual = health_representation - health_representation_last
        
        # ------------------------------------------------------------
        # Final medication prediction
        # ------------------------------------------------------------

        logits = self.output(
            health_representation
        )

        # ------------------------------------------------------------
        # Losses
        # ------------------------------------------------------------

        if self.training:
            
            medication_representation_last = self.output(
                health_representation_last
            )
            
            medication_residual = self.output(
                health_residual
            )
            
            prediction_probability = torch.sigmoid(
                logits
            )

            pairwise_probability = (
                prediction_probability.t()
                * prediction_probability
            )

            ddi_loss = pairwise_probability.mul(self.ddi_adj).mean()

            reconstruction_loss = 1 / self.ddi_adj.shape[0] * torch.sum(
                torch.pow(
                    (prediction_probability - torch.sigmoid(medication_representation_last + medication_residual)),
                    2
                )
            )
            
            return logits, reconstruction_loss, ddi_loss

        return logits
    
    # ================================================================
    # Prediction
    # ================================================================

    def predict(
        self,
        x: list[dict[str, list[int]]],
    ) -> torch.Tensor:
        """
        Generate raw medication prediction scores for one patient example.

        Thresholding and final metric is performed by EHRDRec rather than by the model itself.
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
        ddi_loss: torch.Tensor | float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the MICRON training objective.

        TODO: Implement the exact loss function used in the original MICRON implementation.
        """

        prediction_loss = nn.functional.binary_cross_entropy_with_logits(
            pred,
            target.float(),
        )

        return prediction_loss + ddi_loss

    # ================================================================
    # Saving
    # ================================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the trained MICRON model state.
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