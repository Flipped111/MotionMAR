import torch
import torch.nn as nn


class Refinenet(nn.Module):
    """Bidirectional GRU refinement network for offline temporal smoothing."""
    def __init__(self, n_layers=2, hidden_dim=512, input_dim=132, output_dim=132, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Halve hidden size since bidirectional doubles the output dim.
        self.refine_net = nn.GRU(
            input_dim,
            hidden_dim // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.linear = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        # Small init so initial delta is near zero (residual-friendly).
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, hidden=None):
        bs = x.shape[0]
        if hidden is None:
            hidden = self.init_hidden(bs, x.device, x.dtype)

        feat, hidden = self.refine_net(x, hidden)
        delta = self.linear(feat)
        out = x + delta
        return out, hidden

    def init_hidden(self, batch_size, device=None, dtype=None):
        return torch.zeros(
            2 * self.n_layers,
            batch_size,
            self.hidden_dim // 2,
            device=device,
            dtype=dtype
        )
