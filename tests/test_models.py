"""
Tests for ehrdrec.models — forward-pass shape, dtype, and behavioural invariants.

No training data or MIMIC files required. All inputs are synthetic tensors.
Adjacency matrices are random symmetric float tensors; the tests don't care
about meaningful drug relationships, only that the shapes flow correctly.
"""
import pytest
import torch
import torch.nn as nn

from ehrdrec.models.torch.mlp import MLP
from ehrdrec.models.torch.GAMENet import GameNetFast
from ehrdrec.models.torch.foursdrug import FourSDrug
from ehrdrec.models.torch.FastRx import FastRx
from ehrdrec.models.torch.micron import Micron
from ehrdrec.models.utils.gcn import GCN, GraphConvolution, normalise_adj


# ===========================================================================
# Shared constants and helpers
# ===========================================================================

N_DIAG  = 20
N_PROC  = 15
N_MEDS  = 12
BATCH   = 4
SEQ_LEN = 3  # visits per patient


def _sym_adj(n: int) -> torch.Tensor:
    """Random symmetric adjacency matrix."""
    a = torch.rand(n, n)
    a = (a + a.T) / 2
    a.fill_diagonal_(0)
    return a


def _seq_batch(batch: int = BATCH, seq: int = SEQ_LEN) -> dict:
    """Synthetic sequential input dict used by GAMENet, FastRx, Micron."""
    lengths = torch.randint(1, seq + 1, (batch,))
    return {
        "diagnoses":          torch.rand(batch, seq, N_DIAG),
        "procedures":         torch.rand(batch, seq, N_PROC),
        "medication_history": torch.rand(batch, seq, N_MEDS),
        "lengths":            lengths,
    }


# ===========================================================================
# GCN utilities
# ===========================================================================

class TestGCNUtils:
    def test_normalise_adj_shape(self):
        adj = _sym_adj(N_MEDS)
        out = normalise_adj(adj)
        assert out.shape == (N_MEDS, N_MEDS)

    def test_normalise_adj_no_nan(self):
        adj = _sym_adj(N_MEDS)
        out = normalise_adj(adj)
        assert not out.isnan().any()

    def test_normalise_adj_zero_row_handled(self):
        # An all-zero row (isolated node) should not produce nan/inf
        adj = torch.zeros(5, 5)
        out = normalise_adj(adj)
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_graph_convolution_shape(self):
        gc = GraphConvolution(8, 16)
        adj = normalise_adj(_sym_adj(N_MEDS))
        x   = torch.rand(N_MEDS, 8)
        out = gc(x, adj)
        assert out.shape == (N_MEDS, 16)

    def test_gcn_output_shape(self):
        gcn = GCN(n_nodes=N_MEDS, embed_dim=32)
        adj = _sym_adj(N_MEDS)
        out = gcn(adj)
        assert out.shape == (N_MEDS, 32)

    def test_gcn_no_nan(self):
        gcn = GCN(n_nodes=N_MEDS, embed_dim=32)
        out = gcn(_sym_adj(N_MEDS))
        assert not out.isnan().any()


# ===========================================================================
# MLP
# ===========================================================================

class TestMLP:
    def test_forward_shape(self):
        model = MLP(input_size=50, hidden_sizes=[64, 32], output_size=N_MEDS)
        x = torch.rand(BATCH, 50)
        out = model(x)
        assert out.shape == (BATCH, N_MEDS)

    def test_single_hidden_layer(self):
        model = MLP(input_size=10, hidden_sizes=[16], output_size=5)
        out = model(torch.rand(2, 10))
        assert out.shape == (2, 5)

    def test_no_hidden_layers(self):
        model = MLP(input_size=10, hidden_sizes=[], output_size=5)
        out = model(torch.rand(2, 10))
        assert out.shape == (2, 5)

    def test_output_dtype_float32(self):
        model = MLP(input_size=10, hidden_sizes=[8], output_size=4)
        out = model(torch.rand(2, 10))
        assert out.dtype == torch.float32

    def test_dropout_zero_in_eval(self):
        # With dropout=1.0 everything should be zeroed in train mode but
        # pass through unchanged in eval mode.
        model = MLP(input_size=4, hidden_sizes=[8], output_size=4, dropout=1.0)
        x = torch.rand(2, 4)
        model.eval()
        with torch.no_grad():
            out = model(x)
        # eval mode disables dropout so output should be non-trivially non-zero
        assert out.abs().sum().item() > 0

    def test_parameters_exist(self):
        model = MLP(input_size=10, hidden_sizes=[8], output_size=4)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_gradients_flow(self):
        model = MLP(input_size=10, hidden_sizes=[8], output_size=4)
        x = torch.rand(2, 10)
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ===========================================================================
# GameNetFast
# ===========================================================================

class TestGameNetFast:
    @pytest.fixture
    def model(self):
        return GameNetFast(
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            n_medications=N_MEDS,
            medication_adjacency_matrix=_sym_adj(N_MEDS),
            ddi_adjacency_matrix=_sym_adj(N_MEDS),
            diagnoses_embedding_dim=32,
            procedures_embedding_dim=32,
            hidden_dim=32,
            query_dim=32,
        )

    def test_output_key(self, model):
        out = model(_seq_batch())
        assert "predictions" in out

    def test_predictions_shape(self, model):
        out = model(_seq_batch())
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_output_dtype(self, model):
        out = model(_seq_batch())
        assert out["predictions"].dtype == torch.float32

    def test_no_nan(self, model):
        out = model(_seq_batch())
        assert not out["predictions"].isnan().any()

    def test_single_visit_patients(self, model):
        x = _seq_batch()
        x["lengths"] = torch.ones(BATCH, dtype=torch.long)
        out = model(x)
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_batch_size_one(self, model):
        out = model(_seq_batch(batch=1))
        assert out["predictions"].shape == (1, N_MEDS)

    def test_gradients_flow(self, model):
        out = model(_seq_batch())
        out["predictions"].sum().backward()
        assert model.beta.grad is not None


# ===========================================================================
# FourSDrug
# ===========================================================================

class TestFourSDrug:
    @pytest.fixture
    def model(self):
        return FourSDrug(num_symptoms=N_DIAG + N_PROC, num_drugs=N_MEDS, emb_dim=32)

    def _flat_input(self, batch: int = BATCH) -> torch.Tensor:
        return torch.rand(batch, N_DIAG + N_PROC)

    def test_output_key(self, model):
        out = model(self._flat_input())
        assert "predictions" in out

    def test_predictions_shape(self, model):
        out = model(self._flat_input())
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_output_dtype(self, model):
        out = model(self._flat_input())
        assert out["predictions"].dtype == torch.float32

    def test_no_nan(self, model):
        out = model(self._flat_input())
        assert not out["predictions"].isnan().any()

    def test_batch_size_one(self, model):
        out = model(self._flat_input(batch=1))
        assert out["predictions"].shape == (1, N_MEDS)

    def test_all_zero_input(self, model):
        # All-zero symptom vector should not cause nan/inf
        x = torch.zeros(BATCH, N_DIAG + N_PROC)
        out = model(x)
        assert not out["predictions"].isnan().any()

    def test_gradients_flow(self, model):
        out = model(self._flat_input())
        out["predictions"].sum().backward()
        assert model.symptom_importance.grad is not None


# ===========================================================================
# FastRx
# ===========================================================================

class TestFastRx:
    @pytest.fixture
    def model(self):
        return FastRx(
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            n_medications=N_MEDS,
            medication_adjacency_matrix=_sym_adj(N_MEDS),
            ddi_adjacency_matrix=_sym_adj(N_MEDS),
            embedding_dim=32,
            embedding_dim_fastformer=16,
            dropout=0.0,
        )

    def test_output_key(self, model):
        out = model(_seq_batch())
        assert "predictions" in out

    def test_predictions_shape(self, model):
        out = model(_seq_batch())
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_output_dtype(self, model):
        out = model(_seq_batch())
        assert out["predictions"].dtype == torch.float32

    def test_no_nan(self, model):
        model.eval()
        with torch.no_grad():
            out = model(_seq_batch())
        assert not out["predictions"].isnan().any()

    def test_all_first_visits(self, model):
        # All patients on their first visit (lengths=1) — exercises the no-history branch
        x = _seq_batch()
        x["lengths"] = torch.ones(BATCH, dtype=torch.long)
        out = model(x)
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_mixed_lengths(self, model):
        x = _seq_batch()
        x["lengths"] = torch.tensor([1, 2, 3, 1])
        out = model(x)
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_batch_size_one(self, model):
        out = model(_seq_batch(batch=1))
        assert out["predictions"].shape == (1, N_MEDS)

    def test_gradients_flow(self, model):
        out = model(_seq_batch())
        out["predictions"].sum().backward()
        assert model.inter.grad is not None


# ===========================================================================
# Micron
# ===========================================================================

class TestMicron:
    @pytest.fixture
    def model_train(self):
        m = Micron(
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            n_medications=N_MEDS,
            ddi_adjacency_matrix=_sym_adj(N_MEDS),
            embedding_dim=32,
            dropout=0.0,
            return_losses=True,
        )
        m.train()
        return m

    @pytest.fixture
    def model_eval(self):
        m = Micron(
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            n_medications=N_MEDS,
            ddi_adjacency_matrix=_sym_adj(N_MEDS),
            embedding_dim=32,
            dropout=0.0,
            return_losses=True,
        )
        m.eval()
        return m

    def test_predictions_shape_train(self, model_train):
        out = model_train(_seq_batch())
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_predictions_shape_eval(self, model_eval):
        with torch.no_grad():
            out = model_eval(_seq_batch())
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_losses_returned_in_train(self, model_train):
        out = model_train(_seq_batch())
        assert out["losses"] is not None
        assert "reconstruction_loss" in out["losses"]
        assert "ddi_loss" in out["losses"]

    def test_losses_none_in_eval(self, model_eval):
        with torch.no_grad():
            out = model_eval(_seq_batch())
        assert out["losses"] is None

    def test_loss_values_are_scalars(self, model_train):
        out = model_train(_seq_batch())
        assert out["losses"]["reconstruction_loss"].shape == ()
        assert out["losses"]["ddi_loss"].shape == ()

    def test_loss_values_non_negative(self, model_train):
        out = model_train(_seq_batch())
        assert out["losses"]["reconstruction_loss"].item() >= 0.0
        assert out["losses"]["ddi_loss"].item() >= 0.0

    def test_return_losses_false_skips_losses(self):
        model = Micron(
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            n_medications=N_MEDS,
            ddi_adjacency_matrix=_sym_adj(N_MEDS),
            embedding_dim=32,
            dropout=0.0,
            return_losses=False,
        )
        model.train()
        out = model(_seq_batch())
        assert out["losses"] is None

    def test_no_nan_predictions(self, model_eval):
        with torch.no_grad():
            out = model_eval(_seq_batch())
        assert not out["predictions"].isnan().any()

    def test_single_visit_patients(self, model_train):
        x = _seq_batch()
        x["lengths"] = torch.ones(BATCH, dtype=torch.long)
        out = model_train(x)
        assert out["predictions"].shape == (BATCH, N_MEDS)

    def test_gradients_flow(self, model_train):
        out = model_train(_seq_batch())
        total = out["predictions"].sum() + out["losses"]["reconstruction_loss"]
        total.backward()
        found_grad = any(
            p.grad is not None
            for p in model_train.parameters()
        )
        assert found_grad

    def test_batch_size_one(self, model_eval):
        with torch.no_grad():
            out = model_eval(_seq_batch(batch=1))
        assert out["predictions"].shape == (1, N_MEDS)
