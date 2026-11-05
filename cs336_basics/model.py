import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from cs336_basics.utils import softmax
import math

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

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor) -> Tensor:
    dp = ((Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])).masked_fill(~mask, -torch.inf)  #[q k]
    return softmax(dp, dim=-1) @ V # [q, k] @ [k v] -> [q v]

def multihead_self_attention(d_model: int, num_heads: int, 
                             q_proj_weight: Tensor,
                             k_proj_weight: Tensor,
                             v_proj_weight: Tensor,
                             o_proj_weight: Tensor,
                             in_features: Tensor,
                             )->Tensor:
    assert d_model % num_heads == 0

    q = F.linear(in_features, q_proj_weight) # [... sequence_length d_model] @ [d_model d_model] -> [... sequence_length d_model]
    k = F.linear(in_features, k_proj_weight) # [... sequence_length d_model] @ [d_model d_model] -> [... sequence_length d_model]
    v = F.linear(in_features, v_proj_weight) # [... sequence_length d_model] @ [d_model d_model] -> [... sequence_length d_model]

    *batch_dims, sequence_length, input_dim = in_features.shape
    assert input_dim == d_model
    head_dim = d_model // num_heads

    q = q.reshape(*batch_dims, sequence_length, num_heads, head_dim).transpose(-3, -2) # [... sequence_length d_model] -> [... sequence_length num_heads head_dim] -> [... num_heads sequence_length head_dim] 
    k = k.reshape(*batch_dims, sequence_length, num_heads, head_dim).transpose(-3, -2) # [... sequence_length d_model] -> [... sequence_length num_heads head_dim] -> [... num_heads sequence_length head_dim] 
    v = v.reshape(*batch_dims, sequence_length, num_heads, head_dim).transpose(-3, -2) # [... sequence_length d_model] -> [... sequence_length num_heads head_dim] -> [... num_heads sequence_length head_dim] 
    
    #scores = q @ k.transpose(-2, -1) # [... num_heads sequence_length head_dim] @ [... num_heads head_dim sequence_length] -> [... num_heads sequence_length sequence_length]
    #scores = scores / math.sqrt(head_dim)
    #atten = softmax(scores, dim=-1) # [... num_heads sequence_length sequence_length]
     
    causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
    atten = scaled_dot_product_attention(q, k, v, causal_mask) # [... num_heads sequence_length sequence_length] @ [... num_heads sequence_length head_dim] -> [... num_heads sequence_length head_dim]

    atten = atten.transpose(-3, -2) # [... num_heads sequence_length head_dim] -> [... sequence_length num_heads head_dim]
    atten = atten.reshape(*batch_dims, sequence_length, d_model) # [... sequence_length num_heads head_dim] -> [... sequence_length d_model]

    return F.linear(atten, o_proj_weight) # [... sequence_length d_model] @ [d_model d_model] -> [... sequence_length d_model]

def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    dim_indices = torch.arange(0, d_k, 2)
    inv_freq = 1.0 / (theta ** (dim_indices.float() / d_k))

    positions = torch.arange(0, max_seq_len).float()
    angles = torch.outer(positions, inv_freq)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos_pos = cos[token_positions]
    sin_pos = sin[token_positions]

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    x_rot_even = x_even * cos_pos - x_odd * sin_pos
    x_rot_odd = x_even * sin_pos + x_odd * cos_pos

    out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
    out = out.flatten(-2)

    return out 

def mha_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor,
) -> Tensor:
    assert d_model % num_heads == 0

    q = F.linear(in_features, q_proj_weight) # [... sequence_length d_model] @ [d_model d_model] -> [... sequence_length d_model]
    k = F.linear(in_features, k_proj_weight) # [... sequence_length d_model] @ [d_model d_model] -> [... sequence_length d_model]
    v = F.linear(in_features, v_proj_weight) # [... sequence_length d_model] @ [d_model d_model] -> [... sequence_length d_model]

    *batch_dims, sequence_length, input_dim = in_features.shape
    assert input_dim == d_model
    head_dim = d_model // num_heads

    q = q.reshape(*batch_dims, sequence_length, num_heads, head_dim).transpose(-3, -2) # [... sequence_length d_model] -> [... sequence_length num_heads head_dim] -> [... num_heads sequence_length head_dim] 
    k = k.reshape(*batch_dims, sequence_length, num_heads, head_dim).transpose(-3, -2) # [... sequence_length d_model] -> [... sequence_length num_heads head_dim] -> [... num_heads sequence_length head_dim] 
    v = v.reshape(*batch_dims, sequence_length, num_heads, head_dim).transpose(-3, -2) # [... sequence_length d_model] -> [... sequence_length num_heads head_dim] -> [... num_heads sequence_length head_dim] 
    
    q = rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=q, token_positions=token_positions)
    k = rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=k, token_positions=token_positions)
    #scores = q @ k.transpose(-2, -1) # [... num_heads sequence_length head_dim] @ [... num_heads head_dim sequence_length] -> [... num_heads sequence_length sequence_length]
    #scores = scores / math.sqrt(head_dim)
    #atten = softmax(scores, dim=-1) # [... num_heads sequence_length sequence_length]
     
    causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
    atten = scaled_dot_product_attention(q, k, v, causal_mask) # [... num_heads sequence_length sequence_length] @ [... num_heads sequence_length head_dim] -> [... num_heads sequence_length head_dim]

    atten = atten.transpose(-3, -2) # [... num_heads sequence_length head_dim] -> [... sequence_length num_heads head_dim]
    atten = atten.reshape(*batch_dims, sequence_length, d_model) # [... sequence_length num_heads head_dim] -> [... sequence_length d_model]

    return F.linear(atten, o_proj_weight)

def attention_sublayer(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
    token_positions: Tensor,
    eps: float = 1e-5
):
    h = rmsnorm(d_model, eps, weights["ln1.weight"], in_features)

    atten_out = mha_with_rope(d_model, num_heads, max_seq_len, theta,
                              weights["attn.q_proj.weight"],
                              weights["attn.k_proj.weight"],
                              weights["attn.v_proj.weight"],
                              weights["attn.output_proj.weight"],
                              h, token_positions)

    return in_features + atten_out

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    seq_len = in_features.shape[-2]
    token_positions = torch.arange(seq_len, device=in_features.device)

    x = attention_sublayer(d_model, num_heads, max_seq_len, theta,
                           weights, in_features, token_positions)
    
    h = rmsnorm(d_model, 1e-5, weights["ln2.weight"], x)
    x = x + swiglu(d_model, d_ff, weights["ffn.w1.weight"], weights["ffn.w2.weight"],
                   weights["ffn.w3.weight"], h)

    return x

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor
) -> Tensor:
    x = weights["token_embeddings.weight"][in_indices]

    for i in range(num_layers):
        weights_layers = {
            "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
            "ln1.weight": weights[f"layers.{i}.ln1.weight"],
            "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
            "ln2.weight": weights[f"layers.{i}.ln2.weight"]
        }
        
        x = transformer_block(d_model, num_heads, d_ff, context_length, rope_theta,
                          weights_layers, x)

    x = rmsnorm(d_model, 1e-5, weights["ln_final.weight"], x)
    out_emb = x @ weights["lm_head.weight"].T

    return out_emb