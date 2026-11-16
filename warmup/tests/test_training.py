"""热身练习 3 的测试：autograd 与最小训练循环。"""

import torch
from torch import nn

from warmup.exercises.training import count_parameters, fit_linear_regression, freeze_, grad_of_sum_of_squares


def test_count_parameters():
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2, bias=False))
    assert count_parameters(m) == 4 * 8 + 8 + 8 * 2
    m[0].bias.requires_grad_(False)
    assert count_parameters(m) == 4 * 8 + 8 * 2
    assert count_parameters(m, trainable_only=False) == 4 * 8 + 8 + 8 * 2


def test_freeze():
    m = nn.Linear(3, 3)
    out = freeze_(m)
    assert out is m
    assert all(not p.requires_grad for p in m.parameters())
    assert count_parameters(m) == 0


def test_grad_of_sum_of_squares():
    x = torch.randn(3, 4)
    g = grad_of_sum_of_squares(x)
    torch.testing.assert_close(g, 2 * x)
    assert not x.requires_grad, "不应修改调用者的张量"
    assert not g.requires_grad


def test_fit_linear_regression_recovers_true_params():
    torch.manual_seed(123)
    true_w = torch.tensor([[2.0, -3.0], [0.5, 1.0]])  # (d_out, d_in)
    true_b = torch.tensor([1.0, -2.0])
    x = torch.randn(256, 2)
    y = x @ true_w.T + true_b
    model = fit_linear_regression(x, y, steps=500, lr=0.1)
    assert isinstance(model, nn.Linear)
    torch.testing.assert_close(model.weight.detach(), true_w, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(model.bias.detach(), true_b, atol=1e-3, rtol=1e-3)
