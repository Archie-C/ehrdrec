import torch

from examples.model_training.rpnet import build_rpnet, pretrain_rpnet


def test_rpnet_pretraining_helper_updates_on_synthetic_batch():
    batch_size = 4
    seq_len = 3
    n_diagnoses = 20
    n_procedures = 15
    n_medications = 12

    model = build_rpnet(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ddi_adj=torch.zeros(n_medications, n_medications),
        device=torch.device("cpu"),
    )
    features = {
        "diagnoses": torch.rand(batch_size, seq_len, n_diagnoses),
        "procedures": torch.rand(batch_size, seq_len, n_procedures),
        "medication_history": torch.rand(batch_size, seq_len, n_medications),
        "lengths": torch.tensor([1, 2, 3, 3]),
    }
    loader = [(features, torch.rand(batch_size, n_medications))]

    history = pretrain_rpnet(
        model,
        loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        mask_rate=0.15,
        contrastive_weight=0.1,
        temperature=0.2,
    )

    assert len(history) == 1
    assert history[0]["loss"] > 0
    assert history[0]["reconstruction_loss"] > 0
