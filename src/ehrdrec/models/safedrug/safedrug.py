from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ehrdrec.requirements.model import InputRequirement, Feature, Representation, InputStructure, ModelRequirement
from ehrdrec.models.base import TorchEHRDrecModel
from .layers import MaskLinear, MolecularGraphNeuralNetwork


class SafeDrug(TorchEHRDrecModel):
    """
    EHRDRec implementation of SafeDrug.
    """
    
    _inputs = {
        InputRequirement(
            Feature.DIAGNOSES,
            Representation.CODE_LIST,
            InputStructure.VISIT_SEQUENCE,
        ),
        InputRequirement(
            Feature.PROCEDURES,
            Representation.CODE_LIST,
            InputStructure.VISIT_SEQUENCE,
        ),
    }

    _requirements = {
        ModelRequirement.MOLECULAR_GRAPHS,
        ModelRequirement.MEDICATION_MOLECULE_PROJECTION,
        ModelRequirement.MEDICATION_SUBSTRUCTURE_MATRIX,
        ModelRequirement.DDI_GRAPH,
    }

    def __init__(
        self,
        context,
        embedding_dim: int = 256,
        molecular_graph_embedding_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(context)

        # ------------------------------------------------------------
        # Resolved model information
        # ------------------------------------------------------------

        diagnoses_vocab_size = context.vocab.diagnoses
        procedures_vocab_size = context.vocab.procedures
        medications_vocab_size = context.vocab.medications

        molecular_graphs = context.resources.molecular_graphs
        average_projection = (
            context.resources.medication_molecule_projection
        )
        drug_fragment_mask = (
            context.resources.medication_substructure_matrix
        )
        ddi_adj = context.resources.ddi_graph

        n_fingerprints = molecular_graphs.n_fingerprints
        mpnn_set = molecular_graphs.graphs

        # ------------------------------------------------------------
        # Patient representation
        # ------------------------------------------------------------

        self.diagnoses_embedding = nn.Embedding(
            diagnoses_vocab_size,
            embedding_dim,
        )

        self.procedures_embedding = nn.Embedding(
            procedures_vocab_size,
            embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

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

        # ------------------------------------------------------------
        # Bipartite drug-fragment component
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Molecular graph component
        # ------------------------------------------------------------

        self.mpnn_molecule_set = list(zip(*mpnn_set))

        self.mpnn = MolecularGraphNeuralNetwork(
            n_fingerprints=n_fingerprints,
            dim=embedding_dim,
            layer_hidden=molecular_graph_embedding_layers,
        )

        self.mpnn_output = nn.Linear(
            medications_vocab_size,
            medications_vocab_size,
        )

        self.mpnn_layernorm = nn.LayerNorm(
            medications_vocab_size,
        )

        # ------------------------------------------------------------
        # Fixed resources
        # ------------------------------------------------------------

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

        self._init_weights()

    def _init_weights(self) -> None:
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

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """
        Perform the SafeDrug forward pass.

        EHRDRec is responsible for constructing `batch` according to
        SafeDrug's declared requirements.
        """

        diagnoses = []
        procedures = []

        for visit_diagnoses, visit_procedures in zip(
            batch.diagnoses,
            batch.procedures,
        ):
            diagnosis_embedding = self.diagnoses_embedding(
                visit_diagnoses
            )

            procedure_embedding = self.procedures_embedding(
                visit_procedures
            )

            diagnosis_embedding = self.dropout(
                diagnosis_embedding
            )

            procedure_embedding = self.dropout(
                procedure_embedding
            )

            diagnoses.append(
                diagnosis_embedding.sum(dim=0)
            )

            procedures.append(
                procedure_embedding.sum(dim=0)
            )

        diagnoses = torch.stack(diagnoses).unsqueeze(0)
        procedures = torch.stack(procedures).unsqueeze(0)

        diagnoses, _ = self.diagnoses_encoder(diagnoses)
        procedures, _ = self.procedures_encoder(procedures)

        patient_representation = torch.cat(
            [diagnoses, procedures],
            dim=-1,
        )

        query = self.query(
            patient_representation[:, -1, :]
        )

        # ------------------------------------------------------------
        # Molecular graph branch
        # ------------------------------------------------------------

        molecule_embeddings = self.mpnn(
            self.mpnn_molecule_set
        )

        medication_embeddings = torch.mm(
            self.average_projection,
            molecule_embeddings,
        )

        mpnn_match = torch.sigmoid(
            torch.mm(
                query,
                medication_embeddings.t(),
            )
        )

        mpnn_attention = self.mpnn_layernorm(
            mpnn_match
            + self.mpnn_output(mpnn_match)
        )

        # ------------------------------------------------------------
        # Bipartite branch
        # ------------------------------------------------------------

        bipartite_query = torch.sigmoid(
            self.bipartite_transform(query)
        )

        bipartite_embedding = self.bipartite_output(
            bipartite_query,
            self.drug_fragment_mask.t(),
        )

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------

        logits = bipartite_embedding * mpnn_attention

        prediction_probability = torch.sigmoid(logits)

        pairwise_probability = (
            prediction_probability.t()
            * prediction_probability
        )

        ddi_loss = (
            0.0005
            * pairwise_probability.mul(self.ddi_adj).sum()
        )

        return {
            "logits": logits,
            "ddi_loss": ddi_loss,
        }

    def loss(self, **kwargs) -> torch.Tensor:
        outputs = kwargs["outputs"]
        targets = kwargs["targets"]

        prediction_loss = self.context.task.loss(
            outputs=outputs["logits"],
            targets=targets,
        )

        return prediction_loss + outputs["ddi_loss"]