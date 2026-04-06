import torch
import torch.nn as nn
from .linear import Linear

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None) -> None:
        super().__init__()

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def _silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.W2(self._silu(self.W1(x)) * self.W3(x))