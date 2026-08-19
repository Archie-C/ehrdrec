from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.models.base import TorchEHRDrecModel
from .layers import MaskLinear, MolecularGraphNeuralNetwork


class SafeDrug(TorchEHRDrecModel):
    """
    EHRDRec implementation of SafeDrug.

    Based on the original implementation:
    https://github.com/ycq091044/SafeDrug/blob/main/src/models.py

    SafeDrug uses:
        - longitudinal diagnosis codes
        - longitudinal procedure codes
        - molecular graph information
        - drug-fragment information
        - a drug-drug interaction adjacency matrix

    Notes
    -----
    This implementation intentionally keeps the model-specific architecture
    close to the original SafeDrug implementation while exposing training and
    prediction through the EHRDRec model interface.
    """
    
    def __init__(
        self,
        diagnoses_vocab_size: int,
        procedures_vocab_size: int,
        medications_vocab_size: int,
        n_fingerprints: int,
        drug_fragment_mask: torch.Tensor,
        mpnn_set: Any,
        average_projection: torch.Tensor,
        ddi_adj: torch.Tensor,
        embedding_dim: int = 256,
        molecular_graph_embedding_layers: int = 2,
        dropout: float = 0.5,
        epochs: int = 50,
        learning_rate: float = 1e-3,
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
            hidden_size=embedding_dim,
            batch_first=True,
        )

        self.procedures_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                embedding_dim * 2,
                embedding_dim,
            ),
        )
        
        # ============================================================
        # Bipartite drug-fragment component
        # ============================================================
        
        num_fragments = drug_fragment_mask.shape[1]

        self.bipartite_transform = nn.Linear(
            embedding_dim,
            num_fragments,
        )

        self.bipartite_output = MaskLinear(
            num_fragments,
            medications_vocab_size,
            False,
        )

        # ============================================================
        # Molecular graph component
        # ============================================================
        
        self.mpnn_molecule_set = list(zip(*mpnn_set))
        
        self.mpnn = MolecularGraphNeuralNetwork(
            n_fingerprints,
            embedding_dim,
            layer_hidden=molecular_graph_embedding_layers,
            device=self.device,
        ).forward(self.mpnn_molecule_set)
        
        self.mpnn_emb = torch.mm(
            self.average_projection.to(self.device),
            self.mpnn_emb.to(self.device)
        ).to(self.device)
        
        # ============================================================
        # Medication prediction layers
        # ============================================================

        self.mpnn_output = nn.Linear(
            medications_vocab_size,
            medications_vocab_size,
        )

        self.mpnn_layernorm = nn.LayerNorm(
            medications_vocab_size
        )

        # ============================================================
        # Fixed model resources
        # ============================================================
        
        self.register_buffer(
            "average_projection",
            torch.as_tensor(
                average_projection,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "ddi_adj",
            torch.as_tensor(
                ddi_adj,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "drug_fragment_mask",
            torch.as_tensor(
                drug_fragment_mask,
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
        Train the SafeDrug model.

        Parameters
        ----------
        train_data:
            Training data provided by EHRDRec.

        validation_data:
            Validation data provided by EHRDRec.

        resources:
            Additional resources supplied by EHRDRec.

            SafeDrug currently receives its required resources during
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
        Perform a SafeDrug forward pass.

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

        def sum_embedding(
            embedding: torch.Tensor,
        ) -> torch.Tensor:
            """
            Sum code embeddings within a single visit.

            Input:
                (1, n_codes, embedding_dim)

            Output:
                (1, 1, embedding_dim)
            """
            return embedding.sum(dim=1).unsqueeze(dim=0)
        
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
                sum_embedding(diagnosis_embedding)
            )

            procedures.append(
                sum_embedding(procedure_embedding)
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

        # Use the final visit representation as the patient query.
        query = self.query(
            patient_representation
        )[-1:, :]
        
        # ------------------------------------------------------------
        # Molecular graph branch
        # ------------------------------------------------------------
        
        mpnn_match = torch.sigmoid(
            torch.mm(
                query, 
                self.MPNN_emb.t()
                )
        )
        
        mpnn_attention = self.MPNN_layernorm(
            mpnn_match + self.MPNN_output(mpnn_match)
        )
        
        # ------------------------------------------------------------
        # Bipartite drug-fragment branch
        # -----------------------------------------------------------
        
        bipartite_query = torch.sigmoid(
            self.bipartite_transform(query)
        )

        bipartite_embedding = self.bipartite_output(
            bipartite_query,
            self.drug_fragment_mask.t(),
        )
        
        # ------------------------------------------------------------
        # Final medication prediction
        # ------------------------------------------------------------

        logits = torch.mul(
            bipartite_embedding,
            mpnn_attention,
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

            ddi_loss = (
                0.0005
                * pairwise_probability
                .mul(self.ddi_adj)
                .sum()
            )

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
        Compute the SafeDrug training objective.

        TODO: Implement the exact loss function used in the original SafeDrug implementation.
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
        Save the trained SafeDrug model state.
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