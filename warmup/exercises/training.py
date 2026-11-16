"""
练习 3：autograd 与最小训练循环

完成后运行：

    uv run pytest warmup/tests/test_training.py -v
"""

from __future__ import annotations

import torch  # noqa: F401  (练习里会用到)
from jaxtyping import Float
from torch import Tensor, nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """返回模块的参数总数（元素个数之和）。

    trainable_only=True 时只统计 requires_grad=True 的参数。
    提示：module.parameters()、Tensor.numel()。
    """
    raise NotImplementedError


def freeze_(module: nn.Module) -> nn.Module:
    """原地冻结模块的全部参数（requires_grad=False），并返回该模块本身。"""
    raise NotImplementedError


def grad_of_sum_of_squares(x: Float[Tensor, ...]) -> Float[Tensor, ...]:
    """不手推公式，用 autograd 计算 f(x) = sum(x ** 2) 对 x 的梯度。

    要求：
    - 不能修改调用者传进来的 x（不要改它的 requires_grad）
    - 返回的梯度张量不应带有计算图（requires_grad=False）
    提示：x.detach().clone().requires_grad_()、loss.backward()、.grad
    """
    raise NotImplementedError


def fit_linear_regression(
    x: Float[Tensor, "n d_in"],
    y: Float[Tensor, "n d_out"],
    steps: int = 500,
    lr: float = 0.1,
    seed: int = 0,
) -> nn.Linear:
    """用 nn.Linear + MSE 损失 + torch.optim.SGD，做全批量梯度下降拟合 y ≈ x @ W.T + b。

    标准五步，每一步都要有：
      1. 前向得到预测
      2. 计算损失
      3. 清空旧梯度
      4. 反向传播
      5. 优化器更新参数

    在创建模型之前调用 torch.manual_seed(seed)，保证结果可复现。
    返回训练好的 nn.Linear。
    """
    raise NotImplementedError
