import torch
import torch.nn as nn
import math

from ehrdrec.utils.constants import ReservedId

class LabelAttention(nn.Module):
    def __init__(
        self,
        input_size: int, 
        projection_size: int,
        num_classes: int,
    ) -> None:
        super(LabelAttention, self).__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, projection_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)  # [batch_size,num_classes, seq_len]
        weighted_output = att_weights @ x  # [batch_size,num_classes, input_size]
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )
        
    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and reshape query, key, and value
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Broadcast mask to (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Fill masked positions with -inf

        # Compute attention probabilities
        attn_probs = nn.functional.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attended values
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Project attended values
        output = self.out_proj(attn_output)

        return output
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=384, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src, mask=None):
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask=mask)
        return src


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        attn_output = self.multihead_attn(query, key, value, mask=mask)
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output


class TransformerCrossAttn(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerCrossAttn, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.cross_attn_layers = nn.ModuleList([CrossAttention(d_model, nhead, dropout) for _ in range(num_layers)])
        self.feed_forward_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        ) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x1, x2, mask=None):
        x1_pos = self.pos_encoder(x1)
        x2_pos = self.pos_encoder(x2)

        for i in range(len(self.cross_attn_layers)):
            # x1 attend to x2
            x1_pos = self.cross_attn_layers[i](query=x2_pos, key=x1_pos, value=x1_pos, mask=mask)
            x1_pos = x1_pos + self.feed_forward_layers[i](x1_pos)
            x1_pos = self.norm_layers[i](x1_pos)

            # x2 attend to x1
            x2_pos = self.cross_attn_layers[i](query=x1_pos, key=x2_pos, value=x2_pos, mask=mask)
            x2_pos = x2_pos + self.feed_forward_layers[i](x2_pos)
            x2_pos = self.norm_layers[i](x2_pos)

        return x1_pos, x2_pos

# class LamRec(nn.Module):
#     def __init__(
#         self, 
#         embedding_dim: int = 128,
#     ) -> None:
#         super(LamRec, self).__init__()
#         self.embedding_dim = embedding_dim
        
#         self.conditions_embedding = nn.Embedding(
#             self.n_diagnoses,
#             embedding_dim,
#             padding_idx=ReservedId.PAD
#         )
#         self.procedures_embedding = nn.Embedding(
#             self.n_procedures,
#             embedding_dim,
#             padding_idx=ReservedId.PAD
#         )
        
#         self.seq_encoder = 
    
    
#     def forward(
#         self,
#         x
#     ) -> dict:
        