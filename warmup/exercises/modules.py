"""
练习 2：继承 nn.Module、注册参数 / buffer / 子模块、实现 forward

完成后运行：

    uv run pytest warmup/tests/test_modules.py -v

注意：这里允许直接使用 torch.nn.Linear 等内置层——这只是热身。
正式作业里 Linear / Embedding 等需要你自己实现，届时套路和这里完全一样。
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn


class Affine(nn.Module):
    """逐特征的缩放与平移：y = x * scale + shift

    要求：
    - 两个可学习参数，属性名必须是 `scale` 和 `shift`，形状都是 (dim,)
    - scale 初始化为全 1，shift 初始化为全 0
    - 构造函数接受 device / dtype，并把它们传给参数的创建函数
    - forward 支持任意前导维度 (..., dim)
    """

    def __init__(self, dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        raise NotImplementedError


class TwoLayerMLP(nn.Module):
    """y = fc2(relu(fc1(x)))

    要求：
    - 子模块属性名必须是 `fc1`、`fc2`（都用 nn.Linear，带 bias）
    - fc1: d_in -> d_hidden, fc2: d_hidden -> d_out
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        raise NotImplementedError


class ResidualMLPStack(nn.Module):
    """由 num_blocks 个残差块串联而成：每个块计算 x = x + block(x)，
    block 是一个 TwoLayerMLP(dim, hidden, dim)。最后再接一个 Affine(dim)。

    要求：
    - 块存放在名为 `blocks` 的 nn.ModuleList 中
    - 最后的 Affine 属性名为 `out_affine`
    - 这样 state_dict 的 key 会形如 "blocks.0.fc1.weight"、"out_affine.scale"

    思考题：如果把 blocks 存在普通 Python list 里会发生什么？（测试里有一个检查专门针对它）
    """

    def __init__(self, dim: int, hidden: int, num_blocks: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        raise NotImplementedError


class RunningMean(nn.Module):
    """记录所有见过的输入在最后一维上的运行均值（不是可学习参数！）。

    要求：
    - 用 register_buffer 注册两个 buffer：
        `mean`：形状 (dim,)，初始全 0
        `count`：标量 long 张量，初始 0
    - forward(x) 接受 (..., dim)，把 x 里所有向量（展平前导维度后共 N 个）并入统计：
        new_mean = (mean * count + x_sum) / (count + N)
      然后原样返回 x。
    - 更新统计量时不应产生梯度（想想用什么上下文管理器）。
    - buffer 不应出现在 parameters() 里，但应出现在 state_dict() 里。
    """

    def __init__(self, dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        raise NotImplementedError
