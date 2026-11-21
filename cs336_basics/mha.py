from cs336_basics.rope import Rope
from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.Linear import Linear
import torch
from torch import nn
from einops import rearrange, einsum

class Multiheadattention(nn.Module):
    def __init__(self, 
                d_model: int, 
                num_heads: int, 
                max_seq_len: int | None=None,
                theta: float | None=None,
                use_rope: bool=False, 
                device: torch.device | None=None, 
                dtype: torch.dtype | None=None
            ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        assert d_model % num_heads == 0

        self.qkv_proj_weight = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.o_proj_weight = Linear(d_model, d_model, device=device, dtype=dtype)

        if self.use_rope:
            self.rope = Rope(theta, self.head_dim, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, in_features, token_positions=None):
        qkv = self.qkv_proj_weight(in_features)

        q, k, v = rearrange(qkv, "... s (three d) -> three ... s d", three=3).unbind(0)

        q = rearrange(q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", head_dim=self.head_dim, num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", head_dim=self.head_dim, num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", head_dim=self.head_dim, num_heads=self.num_heads)

        seq_len = in_features.shape[-2]

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=in_features.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)


        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).bool()

        output_head = scaled_dot_product_attention(q, k, v, mask)

        output_head = rearrange(output_head, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")

        output = self.o_proj_weight(output_head)

        return output
