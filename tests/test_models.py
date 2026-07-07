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