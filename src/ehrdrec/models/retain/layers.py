from __future__ import annotations

import torch
from torch import nn

class RETAINGRU(nn.Module):
    """
    GRU implementation matching the equations used in the original
    Theano RETAIN implementation.

    The original model uses one input matrix W, one recurrent matrix U,
    and one bias vector b for the reset, update, and candidate gates.
    Implementing the cell explicitly avoids small parameterisation
    differences between the original code and torch.nn.GRU.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.empty(input_dim, 3 * hidden_dim))
        self.U = nn.Parameter(torch.empty(hidden_dim, 3 * hidden_dim))
        self.b = nn.Parameter(torch.zeros(3 * hidden_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.W, -0.1, 0.1)
        nn.init.uniform_(self.U, -0.1, 0.1)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape (sequence_length, batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Hidden states with shape
            (sequence_length, batch_size, hidden_dim).
        """
        if x.ndim != 3:
            raise ValueError(
                "RETAIN GRU input must have shape "
                "(sequence_length, batch_size, input_dim)."
            )

        batch_size = x.shape[1]
        h = torch.zeros(
            batch_size,
            self.hidden_dim,
            dtype=x.dtype,
            device=x.device,
        )

        outputs = []

        for x_t in x:
            wx = x_t @ self.W + self.b
            uh = h @ self.U

            wx_r, wx_z, wx_h = wx.split(self.hidden_dim, dim=-1)
            uh_r, uh_z, uh_h = uh.split(self.hidden_dim, dim=-1)

            r = torch.sigmoid(wx_r + uh_r)
            z = torch.sigmoid(wx_z + uh_z)
            h_tilde = torch.tanh(wx_h + r * uh_h)

            h = z * h + (1.0 - z) * h_tilde
            outputs.append(h)

        return torch.stack(outputs, dim=0)