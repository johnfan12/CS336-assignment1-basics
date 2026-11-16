import torch
from torch import nn
import math
from einops import einsum, rearrange


class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0

        position = torch.arange(max_seq_len, device=device)
        dim_indices = torch.arange(0, d_k, 2, device=device)

        freqs = theta ** -(dim_indices / d_k)

        angles = position[:, None] * freqs[None, :]

        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)

        self.register_buffer("cos_table", cos_table)
        self.register_buffer("sin_table", sin_table)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        x1, x2 = rearrange(x, "... (half_d xy) -> ... half_d xy", xy=2).unbind(dim=-1)

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
        x_rot = rearrange(x_rot, "... half_d xy -> ... (half_d xy)")
        return x_rot