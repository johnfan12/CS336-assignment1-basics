from einops import einsum
import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.weights = nn.Parameter(torch.empty(self.out_dim, self.in_dim, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (self.in_dim + self.out_dim))
        nn.init.trunc_normal_(self.weights, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "batch_size seq_len in_dim, out_dim in_dim -> batch_size seq_len out_dim")
        