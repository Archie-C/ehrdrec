import torch

from examples.model_training.sspnet import build_sspnet_model
from ehrdrec.training.losses import OriginalGAMENetLoss


def test_sspnet_training_adapter_runs_on_synthetic_batch():
    batch_size = 2
    seq_len = 3
    n_diagnoses = 20
    n_procedures = 16
    n_medications = 12

    model = build_sspnet_model(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ehr_adj=torch.zeros(n_medications, n_medications),
        ddi_adj=torch.zeros(n_medications, n_medications),
        device=torch.device("cpu"),
    )
    features = {
        "diagnoses": torch.zeros(batch_size, seq_len, n_diagnoses),
        "procedures": torch.zeros(batch_size, seq_len, n_procedures),
        "medication_history": torch.zeros(batch_size, seq_len, n_medications),
        "lengths": torch.tensor([1, 3]),
    }
    features["diagnoses"][0, 0, [1, 2]] = 1
    features["procedures"][0, 0, [1]] = 1
    features["diagnoses"][1, :, 3] = 1
    features["procedures"][1, :, 4] = 1
    features["medication_history"][1, 0, [2, 3]] = 1
    features["medication_history"][1, 1, [4]] = 1

    targets = torch.rand(batch_size, n_medications)
    out = model(features)
    loss = OriginalGAMENetLoss()(
        out["predictions"],
        targets,
        model_output=out,
        features=features,
        losses=out.get("losses"),
    )
    loss.backward()

    assert out["predictions"].shape == (batch_size, n_medications)
    assert "ddi_loss" in out["losses"]
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())
