import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

class Linear(torch.nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        w = torch.empty(d_out, d_in, device=device, dtype=dtype)
        self.weight = nn.Parameter(w)
        sigma = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.weight,std=sigma)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.weight)

def silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return in_features / (1 + torch.exp(-in_features))

def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    
    branch_a = silu(in_features @ w1_weight.T) # ... d_ff
    branch_b = in_features @ w3_weight.T # ... d_ff
    combine = branch_a * branch_b # ... d_ff
    return combine @ w2_weight.T

class Swiglu(nn.Module):
    def __init__(self, d_ff: int, d_model: int):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model 
        self.w1 = Linear(d_ff, d_model)
        self.w2 = Linear(d_model, d_ff)
        self.w3 = Linear(d_ff, d_model)

    def forward(self, in_features: Float[Tensor, "... d_model"]) -> Tensor:
        branch_a = silu(self.w1(in_features)) # ... d_ff
        branch_b = self.w3(in_features) # ... d_ff
        combine = branch_a * branch_b # ... d_ff
        return self.w2(combine)
    
def rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor):
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    return (in_features / rms) * weights

