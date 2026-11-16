"""热身练习 2 的测试：nn.Module 的参数、buffer、子模块、state_dict、forward。"""

import pytest
import torch
from torch import nn

from warmup.exercises.modules import Affine, ResidualMLPStack, RunningMean, TwoLayerMLP


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------------- Affine ----------------


def test_affine_parameters_registered():
    m = Affine(5)
    names = dict(m.named_parameters())
    assert set(names) == {"scale", "shift"}, "参数必须用 nn.Parameter 包装并赋值给 self.scale / self.shift"
    assert names["scale"].shape == (5,) and names["shift"].shape == (5,)
    torch.testing.assert_close(m.scale.data, torch.ones(5))
    torch.testing.assert_close(m.shift.data, torch.zeros(5))


def test_affine_forward_broadcast():
    m = Affine(4)
    with torch.no_grad():
        m.scale.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        m.shift.copy_(torch.tensor([0.5, 0.0, -1.0, 0.0]))
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(m(x), x * m.scale + m.shift)


def test_affine_device_dtype():
    m = Affine(3, dtype=torch.float64)
    assert m.scale.dtype == torch.float64 and m.shift.dtype == torch.float64


def test_affine_gradients_flow():
    m = Affine(3)
    m(torch.randn(4, 3)).sum().backward()
    assert m.scale.grad is not None and m.shift.grad is not None
    torch.testing.assert_close(m.shift.grad, torch.full((3,), 4.0))


# ---------------- TwoLayerMLP ----------------


def test_mlp_structure_and_forward():
    m = TwoLayerMLP(4, 8, 3)
    assert isinstance(m.fc1, nn.Linear) and isinstance(m.fc2, nn.Linear)
    assert m.fc1.weight.shape == (8, 4) and m.fc2.weight.shape == (3, 8)
    x = torch.randn(2, 5, 4)
    expected = m.fc2(torch.relu(m.fc1(x)))
    out = m(x)
    assert out.shape == (2, 5, 3)
    torch.testing.assert_close(out, expected)


# ---------------- ResidualMLPStack ----------------


def test_stack_state_dict_keys():
    m = ResidualMLPStack(dim=4, hidden=8, num_blocks=2)
    keys = set(m.state_dict().keys())
    expected = {f"blocks.{i}.{layer}.{p}" for i in range(2) for layer in ("fc1", "fc2") for p in ("weight", "bias")} | {
        "out_affine.scale",
        "out_affine.shift",
    }
    assert keys == expected, "如果 blocks 不在 state_dict 里，检查你是不是用了普通 list 而不是 nn.ModuleList"
    assert isinstance(m.blocks, nn.ModuleList)


def test_stack_forward_matches_manual():
    m = ResidualMLPStack(dim=4, hidden=8, num_blocks=3)
    x = torch.randn(2, 4)
    h = x
    for blk in m.blocks:
        h = h + blk(h)
    torch.testing.assert_close(m(x), m.out_affine(h))


def test_stack_load_state_dict_roundtrip():
    a = ResidualMLPStack(4, 8, 2)
    b = ResidualMLPStack(4, 8, 2)
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 4)
    torch.testing.assert_close(a(x), b(x))


# ---------------- RunningMean ----------------


def test_running_mean_buffers():
    m = RunningMean(3)
    assert len(list(m.parameters())) == 0, "统计量是 buffer，不是可学习参数"
    sd = m.state_dict()
    assert set(sd) == {"mean", "count"}
    assert sd["count"].dtype == torch.long and sd["count"].shape == ()


def test_running_mean_updates():
    m = RunningMean(3)
    x1 = torch.randn(4, 3)
    x2 = torch.randn(2, 5, 3)
    out = m(x1)
    assert out is x1 or torch.equal(out, x1)
    m(x2)
    expected = torch.cat([x1, x2.reshape(-1, 3)]).mean(0)
    torch.testing.assert_close(m.mean, expected)
    assert m.count.item() == 14


def test_running_mean_no_grad_graph():
    m = RunningMean(2)
    x = torch.randn(3, 2, requires_grad=True)
    m(x)
    assert not m.mean.requires_grad, "更新 buffer 时应该处于 torch.no_grad() 中"
