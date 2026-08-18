import torch

from ehrdrec.models.torch.original.dgcl import DGCL, DGCLTrainingAdapter
from ehrdrec.training import Trainer
from ehrdrec.training.losses import OriginalGAMENetLoss


def _features():
    batch_size = 2
    n_visits = 3
    n_diagnoses = 9
    n_procedures = 8
    n_medications = 6

    features = {
        "diagnoses": torch.zeros(batch_size, n_visits, n_diagnoses),
        "procedures": torch.zeros(batch_size, n_visits, n_procedures),
        "medication_history": torch.zeros(batch_size, n_visits, n_medications),
        "lengths": torch.tensor([3, 2]),
    }
    features["diagnoses"][0, 0, [1, 2]] = 1
    features["diagnoses"][0, 2, [3]] = 1
    features["procedures"][0, 1, [2]] = 1
    features["medication_history"][0, 0, [1]] = 1
    features["medication_history"][0, 2, [4]] = 1
    features["diagnoses"][1, 0, [4]] = 1
    features["procedures"][1, 1, [3]] = 1
    features["medication_history"][1, 1, [5]] = 1

    targets = torch.zeros(batch_size, n_medications)
    targets[0, [2, 4]] = 1
    targets[1, [1]] = 1
    return features, targets


def _model():
    n_medications = 6
    ddi_adj = torch.zeros(n_medications, n_medications)
    ddi_adj[0, 1] = ddi_adj[1, 0] = 1
    dgcl = DGCL(
        n_diagnoses=9,
        n_procedures=8,
        n_medications=n_medications,
        ehr_adjacency_matrix=torch.zeros(n_medications, n_medications),
        ddi_adjacency_matrix=ddi_adj,
        ddi_mask_H=ddi_adj,
        embedding_dim=8,
        number_of_heads=2,
    )
    return DGCLTrainingAdapter(dgcl)


def test_dgcl_training_adapter_returns_standard_trainer_output():
    model = _model()
    features, _ = _features()

    output = model(features)

    assert output["predictions"].shape == (2, 6)
    assert output["losses"]["ddi_loss"].shape == ()


def test_dgcl_training_adapter_runs_with_standard_trainer():
    model = _model()
    features, targets = _features()
    trainer = Trainer(
        model=model,
        train_loader=[(features, targets)],
        val_loader=[(features, targets)],
        loss_fn=OriginalGAMENetLoss(ddi_weight=0.05),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device="cpu",
        epochs=1,
    )

    results = trainer.fit()

    assert results.final_train_loss is not None
