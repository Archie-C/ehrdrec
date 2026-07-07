import torch

from ehrdrec.models.torch.original.rpnet import RPNet
from ehrdrec.training.losses import OriginalGAMENetLoss


N_DIAG = 20
N_PROC = 15
N_MEDS = 12
BATCH = 4
SEQ_LEN = 3


def _sym_adj(n: int) -> torch.Tensor:
    adj = torch.rand(n, n)
    adj = (adj + adj.T) / 2
    adj.fill_diagonal_(0)
    return adj


def _seq_batch(batch: int = BATCH, seq: int = SEQ_LEN) -> dict:
    return {
        "diagnoses": torch.rand(batch, seq, N_DIAG),
        "procedures": torch.rand(batch, seq, N_PROC),
        "medication_history": torch.rand(batch, seq, N_MEDS),
        "lengths": torch.randint(1, seq + 1, (batch,)),
    }


def _model(patient_separate: bool = True) -> RPNet:
    return RPNet(
        n_diagnoses=N_DIAG,
        n_procedures=N_PROC,
        n_medications=N_MEDS,
        ddi_adjacency_matrix=_sym_adj(N_MEDS),
        embedding_dim=32,
        encoder_layers=1,
        number_of_heads=4,
        dropout=0.0,
        patient_separate=patient_separate,
    )


def test_predictions_shape_for_dense_history_batch():
    out = _model()(_seq_batch())
    assert out["predictions"].shape == (BATCH, N_MEDS)


def test_returns_ddi_loss():
    out = _model()(_seq_batch())
    assert out["losses"]["ddi_loss"].shape == ()


def test_single_visit_patients_have_no_history_failure():
    x = _seq_batch()
    x["lengths"] = torch.ones(BATCH, dtype=torch.long)
    out = _model()(x)
    assert out["predictions"].shape == (BATCH, N_MEDS)


def test_padded_code_id_inputs():
    features = {
        "diagnoses": torch.randint(2, N_DIAG, (BATCH, SEQ_LEN, 4)),
        "procedures": torch.randint(2, N_PROC, (BATCH, SEQ_LEN, 3)),
        "medication_history": torch.randint(2, N_MEDS, (BATCH, SEQ_LEN, 5)),
        "lengths": torch.tensor([1, 2, 3, 2]),
    }
    features["diagnoses"][0, 1:] = 1
    features["procedures"][0, 1:] = 1
    features["medication_history"][0, 1:] = 1

    out = _model()(features)

    assert out["predictions"].shape == (BATCH, N_MEDS)
    assert torch.isfinite(out["predictions"]).all()


def test_unified_encoder_path():
    out = _model(patient_separate=False)(_seq_batch())
    assert out["predictions"].shape == (BATCH, N_MEDS)


def test_ddi_loss_integrates_with_existing_loss_function():
    model = _model()
    features = _seq_batch()
    targets = torch.rand(BATCH, N_MEDS)

    out = model(features)
    loss = OriginalGAMENetLoss()(
        out["predictions"],
        targets,
        model_output=out,
        features=features,
        losses=out["losses"],
    )
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())
