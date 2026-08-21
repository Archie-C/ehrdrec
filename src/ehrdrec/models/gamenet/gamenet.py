from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ehrdrec.data.requirements import ModelRequirement
from ehrdrec.models.base import TorchEHRDrecModel
from .layers import GCN


class GAMENet(TorchEHRDrecModel):
    """
    EHRDRec implementation of GAMENet.

    Based on the original implementation:
    https://github.com/sjy1203/GAMENet/blob/master/code/models.py

    GAMENet uses:
    ...

    Notes
    -----
    This implementation intentionally keeps the model-specific architecture
    close to the original GAMENet implementation while exposing training and
    prediction through the EHRDRec model interface.
    """

    requirements = {
        ModelRequirement.DIAGNOSES,
        ModelRequirement.PROCEDURES,
        ModelRequirement.MEDICATION_HISTORY,
        ModelRequirement.EHR_MEDICATION_GRAPH,
        ModelRequirement.DDI_GRAPH,
    }
    
    def __init__(
        self,
        diagnoses_vocab_size: int,
        procedures_vocab_size: int,
        medications_vocab_size: int,
        ddi_adj: torch.Tensor,
        ehr_adj: torch.Tensor,
        ddi_in_memory: bool,
        ddi_loss_enabled: bool,
        embedding_dim: int = 64,
        dropout: float = 0.4,
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

        self.diagnoses_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim * 2,
            batch_first=True,
        )

        self.procedures_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim * 2,
            batch_first=True,
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                embedding_dim * 4,
                embedding_dim,
            ),
        )
        
        # ============================================================
        # GCNs
        # ============================================================
        
        self.ehr_gcn = GCN(
            voc_size=medications_vocab_size, 
            emb_dim=embedding_dim, 
            adj=ehr_adj, 
            device=self.device
        )
        
        self.ddi_gcn = GCN(
            voc_size=medications_vocab_size,
            emb_dim=embedding_dim,
            adj=ddi_adj,
            device=self.device
        )
        
        self.inter = nn.Parameter(torch.FloatTensor(1))
        
        # ============================================================
        # Medication prediction layers
        # ============================================================

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, medications_vocab_size),
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
        
        nn.init.uniform_(
            self.inter,
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
        Train the GAMENet model.

        Parameters
        ----------
        train_data:
            Training data provided by EHRDRec.

        validation_data:
            Validation data provided by EHRDRec.

        resources:
            Additional resources supplied by EHRDRec.

            GAMENet currently receives its required resources during
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
        Perform a GAMENet forward pass.

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
    
        diagnoses = []
        procedures = []

        def mean_embedding(
            embedding: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute the mean of code embeddings within a single visit.

            Input:
                (1, n_codes, embedding_dim)

            Output:
                (1, 1, embedding_dim)
            """
            return embedding.mean(dim=1).unsqueeze(dim=0)
        
        for admission in patient_history:
            diagnosis_codes = torch.as_tensor(
                admission["diagnoses"],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            procedure_codes = torch.as_tensor(
                admission["procedures"],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            diagnosis_embedding = self.diagnoses_embedding(
                diagnosis_codes
            )

            procedure_embedding = self.procedures_embedding(
                procedure_codes
            )

            diagnosis_embedding = self.dropout(
                diagnosis_embedding
            )

            procedure_embedding = self.dropout(
                procedure_embedding
            )

            diagnoses.append(
                mean_embedding(diagnosis_embedding)
            )

            procedures.append(
                mean_embedding(procedure_embedding)
            )
        
        # ------------------------------------------------------------
        # Longitudinal patient representation
        # ------------------------------------------------------------
        
        diagnoses = torch.cat(
            diagnoses,
            dim=1,
        )

        procedures = torch.cat(
            procedures,
            dim=1,
        )

        diagnoses, _ = self.diagnoses_encoder(
            diagnoses
        )

        procedures, _ = self.procedures_encoder(
            procedures
        )

        patient_representation = torch.cat(
            [
                diagnoses,
                procedures,
            ],
            dim=-1,
        ).squeeze(0)
        
        queries = self.query(
            patient_representation
        )

        # Use the final visit representation as the patient query.
        query = queries[-1:]
        
        # ------------------------------------------------------------
        # Graph Memory Banks
        # ------------------------------------------------------------
        
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.inter * self.ddi_gcn()
        else:
            drug_memory = self.ehr_gcn()
        
        if len(patient_history) > 1:
            history_keys = queries[:-1]
            history_values = torch.zeros(
                len(patient_history) - 1,
                self.medications_vocab_size,
                device=self.device,
            )
            for idx, adm in enumerate(patient_history[:-1]):
                history_values[idx, adm["medications"]] = 1.0
            
        key_weights = F.softmax(
            torch.mm(
                query,
                drug_memory.t()
            ),
            dim=-1
        )
        fact1 = torch.mm(
            key_weights,
            drug_memory
        )
        
        if len(patient_history) > 1:
            visit_weight = F.softmax(
                torch.mm(
                    query,
                    history_keys.t()
                ),
                dim=-1
            )
            weighted_history = visit_weight.mm(history_values)
            fact2 = torch.mm(weighted_history, drug_memory)
        else:
            fact2 = fact1
        
        
        # ------------------------------------------------------------
        # Final medication prediction
        # ------------------------------------------------------------

        logits = self.output(
            torch.cat(
                [
                    query,
                    fact1,
                    fact2,
                ],
                dim=-1
            )
        )
        
        # ------------------------------------------------------------
        # DDI loss
        # ------------------------------------------------------------

        if self.training:

            prediction_probability = torch.sigmoid(
                logits
            )

            pairwise_probability = (
                prediction_probability.t()
                * prediction_probability
            )

            ddi_loss = pairwise_probability.mul(self.ddi_adj).mean()

            return logits, ddi_loss

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
        Compute the GAMENet training objective.

        TODO: Implement the exact loss function used in the original GAMENet implementation.
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
        Save the trained GAMENet model state.
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