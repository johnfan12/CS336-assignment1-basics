import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from cs336_basics.utils import softmax
import math
import os
import regex as re

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    