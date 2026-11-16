"""热身练习 1 的测试。参考答案都用最朴素的 for 循环写成，便于你对照理解语义。"""

import pytest
import torch

from warmup.exercises import tensor_ops as ops


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_batched_outer():
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 5)
    out = ops.batched_outer(a, b)
    assert out.shape == (2, 3, 4, 5)
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(out[i, j], torch.outer(a[i, j], b[i, j]))


def test_batched_outer_no_batch_dims():
    a, b = torch.randn(4), torch.randn(6)
    torch.testing.assert_close(ops.batched_outer(a, b), torch.outer(a, b))


def test_pairwise_sq_dist():
    x, y = torch.randn(5, 3), torch.randn(7, 3)
    out = ops.pairwise_sq_dist(x, y)
    assert out.shape == (5, 7)
    expected = torch.empty(5, 7)
    for i in range(5):
        for j in range(7):
            expected[i, j] = ((x[i] - y[j]) ** 2).sum()
    torch.testing.assert_close(out, expected)


def test_batched_matvec():
    mats, vecs = torch.randn(4, 3, 5), torch.randn(4, 5)
    out = ops.batched_matvec(mats, vecs)
    assert out.shape == (4, 3)
    for b in range(4):
        torch.testing.assert_close(out[b], mats[b] @ vecs[b])


def test_patchify():
    B, C, H, W, p = 2, 3, 4, 6, 2
    images = torch.randn(B, C, H, W)
    out = ops.patchify(images, p)
    assert out.shape == (B, (H // p) * (W // p), p * p * C)
    for b in range(B):
        k = 0
        for r in range(H // p):
            for c in range(W // p):
                patch = images[b, :, r * p : (r + 1) * p, c * p : (c + 1) * p]  # (C, p, p)
                expected = patch.permute(1, 2, 0).reshape(-1)  # (p_row, p_col, C) 顺序展平
                torch.testing.assert_close(out[b, k], expected)
                k += 1


def test_split_groups():
    B, T, G, D = 2, 5, 3, 4
    x = torch.randn(B, T, G * D)
    out = ops.split_groups(x, G)
    assert out.shape == (B, G, T, D)
    for g in range(G):
        torch.testing.assert_close(out[:, g], x[:, :, g * D : (g + 1) * D])


def test_merge_groups_roundtrip():
    x = torch.randn(2, 5, 12)
    torch.testing.assert_close(ops.merge_groups(ops.split_groups(x, 4)), x)


def test_masked_mean():
    x = torch.randn(3, 4, 6)
    mask = torch.rand(3, 4, 6) > 0.5
    mask[..., 0] = True  # 保证每行至少有一个 True
    out = ops.masked_mean(x, mask)
    assert out.shape == (3, 4)
    for i in range(3):
        for j in range(4):
            torch.testing.assert_close(out[i, j], x[i, j][mask[i, j]].mean())


def test_fill_upper_triangle():
    scores = torch.randn(2, 4, 4)
    original = scores.clone()
    out = ops.fill_upper_triangle(scores, float("-inf"))
    torch.testing.assert_close(scores, original, msg="不应原地修改输入")
    for i in range(4):
        for j in range(4):
            if j > i:
                assert torch.all(out[:, i, j] == float("-inf"))
            else:
                torch.testing.assert_close(out[:, i, j], scores[:, i, j])


def test_pick_per_row():
    x = torch.randn(5, 10)
    idx = torch.tensor([0, 9, 3, 3, 7])
    out = ops.pick_per_row(x, idx)
    assert out.shape == (5,)
    torch.testing.assert_close(out, torch.stack([x[b, idx[b]] for b in range(5)]))


def test_lookup_rows():
    table = torch.randn(10, 4)
    idx = torch.randint(0, 10, (2, 3))
    out = ops.lookup_rows(table, idx)
    assert out.shape == (2, 3, 4)
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(out[i, j], table[idx[i, j]])
