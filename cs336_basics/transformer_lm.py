from torch import nn
import torch
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import Rmsnorm
from cs336_basics.utils import softmax
from cs336_basics.Linear import Linear

class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float
    ) -> None:
        super().__init__()
        self.embd = Embedding(vocab_size, d_model)
        self.Tblock = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.norm = Rmsnorm(d_model)
        self.head = Linear(d_model, vocab_size)
        

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embd(tokens)
        x = self.Tblock(x)
        x = self.norm(x)
        x = self.head(x)
        
        return x