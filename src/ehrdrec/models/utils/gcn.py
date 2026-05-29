import torch.nn as nn
import torch

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
        h = torch.relu(self.gc1(self.node_embeddings, adj))
        h = self.dropout(h)
        z = self.gc2(h, adj)
        return z