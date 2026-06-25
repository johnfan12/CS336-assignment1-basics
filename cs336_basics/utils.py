import torch
import math
from jaxtyping import Float,Bool,Int
from torch import Tensor
from collections.abc import Iterable

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    max_val = torch.max(in_features, dim=dim, keepdim=True).values
    exp_val = torch.exp(in_features - max_val)
    sum_val = torch.sum(exp_val, dim=dim, keepdim=True)
    out_prob = exp_val / sum_val
    
    return out_prob

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # return torch.nn.functional.cross_entropy(inputs, targets)
    max_val = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_val

    lse = torch.logsumexp(shifted, dim=-1)
    idx = targets.unsqueeze(-1)
    correct = shifted.gather(dim=-1, index=idx)

    nll = lse - correct

    return nll.mean()

def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    