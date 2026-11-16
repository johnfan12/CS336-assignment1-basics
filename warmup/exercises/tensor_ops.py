"""
练习 1：张量操作（广播 / 索引 / einsum / einops）

每个函数都只需要几行代码。先读 notebook 中对应小节，再动手。
完成后运行：

    uv run pytest warmup/tests/test_tensor_ops.py -v

约束（为了练习，而不是为了刁难）：
- 不要写 Python for 循环去遍历张量元素，全部用张量运算完成。
- 能用 einops / torch.einsum 的地方，优先用它们来表达维度变换。
"""

from __future__ import annotations

import torch  # noqa: F401  (练习里会用到)
from einops import rearrange, reduce, repeat  # noqa: F401  (练习里会用到)
from jaxtyping import Bool, Float, Int
from torch import Tensor


def batched_outer(
    a: Float[Tensor, "... n"],
    b: Float[Tensor, "... m"],
) -> Float[Tensor, "... n m"]:
    """对任意多个前导 batch 维，计算 a 和 b 最后一维的外积。

    out[..., i, j] = a[..., i] * b[..., j]

    提示：可以用广播（unsqueeze / None 索引）或 einsum 两种方式实现，建议两种都试试。
    """
    raise NotImplementedError


def pairwise_sq_dist(
    x: Float[Tensor, "n d"],
    y: Float[Tensor, "m d"],
) -> Float[Tensor, "n m"]:
    """计算两组向量两两之间的欧氏距离平方。

    out[i, j] = sum_k (x[i, k] - y[j, k]) ** 2

    提示：想一想如何把 x 变成 (n, 1, d)、y 变成 (1, m, d)，让广播帮你做"两两"。
    """
    raise NotImplementedError


def batched_matvec(
    mats: Float[Tensor, "batch n d"],
    vecs: Float[Tensor, "batch d"],
) -> Float[Tensor, "batch n"]:
    """对每个 batch 元素，计算矩阵乘向量：out[b] = mats[b] @ vecs[b]。

    要求：用 einsum（torch.einsum 或 einops.einsum）写一次，并想想用 @ 怎么写。
    """
    raise NotImplementedError


def patchify(
    images: Float[Tensor, "batch channels height width"],
    patch_size: int,
) -> Float[Tensor, "batch num_patches patch_dim"]:
    """把图片切成不重叠的 patch_size x patch_size 小块，并展平。

    - num_patches = (H // p) * (W // p)，按"先行后列"的顺序排列
    - patch_dim   = p * p * C，每个 patch 内部按 (p_row, p_col, channel) 的顺序展平

    提示：一次 einops.rearrange 就够了。仔细想想括号里维度的顺序。
    """
    raise NotImplementedError


def split_groups(
    x: Float[Tensor, "batch seq dim"],
    num_groups: int,
) -> Float[Tensor, "batch num_groups seq group_dim"]:
    """把最后一维 dim 切成 num_groups 份（每份 group_dim = dim // num_groups），
    并把 group 维移到 seq 维之前。

    out[b, g, t, k] = x[b, t, g * group_dim + k]

    这个操作在后面实现多头注意力时会反复用到。
    """
    raise NotImplementedError


def merge_groups(
    x: Float[Tensor, "batch num_groups seq group_dim"],
) -> Float[Tensor, "batch seq dim"]:
    """split_groups 的逆操作。"""
    raise NotImplementedError


def masked_mean(
    x: Float[Tensor, "... n"],
    mask: Bool[Tensor, "... n"],
) -> Float[Tensor, ...]:
    """沿最后一维求均值，但只统计 mask 为 True 的位置。

    可以假设每一行至少有一个 True。
    提示：bool 张量可以直接 .to(x.dtype) 变成 0/1。
    """
    raise NotImplementedError


def fill_upper_triangle(
    scores: Float[Tensor, "... n n"],
    value: float,
) -> Float[Tensor, "... n n"]:
    """把最后两维中"严格上三角"（列号 j > 行号 i）的位置填成 value，其余保持不变。

    不要原地修改输入。
    提示：torch.triu / torch.tril / torch.ones(..., dtype=torch.bool)，以及 masked_fill。
    """
    raise NotImplementedError


def pick_per_row(
    x: Float[Tensor, "batch vocab"],
    idx: Int[Tensor, " batch"],
) -> Float[Tensor, " batch"]:
    """对每一行 b，取出 x[b, idx[b]]。

    提示：高级索引 x[rows, cols]，或者 torch.gather。两种都值得掌握。
    """
    raise NotImplementedError


def lookup_rows(
    table: Float[Tensor, "num_rows dim"],
    idx: Int[Tensor, ...],
) -> Float[Tensor, "... dim"]:
    """用任意形状的整数张量 idx 去查表：out[...] = table[idx[...]]。

    提示：试试直接 table[idx]，观察输出形状。
    """
    raise NotImplementedError
