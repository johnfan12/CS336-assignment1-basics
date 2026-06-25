import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

class Linear(torch.nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        w = torch.empty(d_out, d_in, device=device, dtype=dtype)
        self.W = nn.Parameter(w)
        sigma = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.W,std=sigma)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.W)

    