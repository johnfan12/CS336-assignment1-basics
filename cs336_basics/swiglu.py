import torch
from torch import nn
import math
from einops import einsum
from cs336_basics.Linear import Linear

def silu(x):
    return x / (1 + torch.exp(-x))

class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, in_features):
        return self.w2(silu(self.w1(in_features)) * (self.w3(in_features)))