import torch
import torch.nn as nn


def normalise_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation: Ã = D̃^{-½}(A + I)D̃^{-½} (eq. 4 in GAMENet paper)."""
    a = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[deg == 0] = 0.0
    return d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        return adj @ x @ self.weight + self.bias


class GCN(nn.Module):
    def __init__(self, n_nodes, embed_dim, dropout=0.3):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.empty(n_nodes, embed_dim))
        self.gc1 = GraphConvolution(embed_dim, embed_dim)
        self.gc2 = GraphConvolution(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self, adj):
        a = normalise_adj(adj)
        # Z = Ã·tanh(Ã·W_e·W1) (eq. 5 in GAMENet paper)
        h = torch.tanh(self.gc1(self.node_embeddings, a))
        h = self.dropout(h)
        z = self.gc2(h, a)
        return z