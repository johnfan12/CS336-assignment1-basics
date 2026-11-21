import torch
from torch import nn
from cs336_basics.rmsnorm import Rmsnorm
from cs336_basics.mha import Multiheadattention
from cs336_basics.swiglu import Swiglu


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, rope_theta: float) -> None:
        super().__init__()
        self.rms1 = Rmsnorm(d_model)
        self.mha = Multiheadattention(d_model, num_heads, max_seq_len, theta=rope_theta, use_rope=True)
        self.rms2 = Rmsnorm(d_model)
        self.ffn = Swiglu(d_model, d_ff)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.rms1(x))
        y = x + self.ffn(self.rms2(x))

        return y