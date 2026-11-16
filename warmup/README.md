# 第 0 章：热身（PyTorch / einops / nn.Module）

这是在原版 CS336 作业 1 之外**额外添加**的入门章节，不修改任何原作业内容。
目标是在进入正式作业前，把后面反复用到的工具操作练熟。

## 目录结构

```
warmup/
├── README.md                       # 本文件
├── 00_pytorch_einops_primer.ipynb  # 讲解 notebook（含可运行示例）
├── exercises/                      # 你来实现（所有函数目前都 raise NotImplementedError）
│   ├── tensor_ops.py               # 广播、索引、einsum、einops
│   ├── modules.py                  # 继承 nn.Module、Parameter、buffer、ModuleList、forward
│   └── training.py                 # autograd、最小训练循环
└── tests/                          # 写好的测试
    ├── test_tensor_ops.py
    ├── test_modules.py
    └── test_training.py
```

## 建议流程

1. 运行 notebook（需要 Jupyter，可用 `uv run --with jupyter jupyter lab` 临时启动，不会改动项目依赖）
2. 按 `tensor_ops.py → modules.py → training.py` 的顺序完成练习
3. 自测：

   ```bash
   uv run pytest warmup/tests -v                     # 全部
   uv run pytest warmup/tests/test_modules.py -v     # 单个文件
   ```

4. 全部通过后，开始正式作业。

## 和正式作业的对应关系

| 热身内容 | 正式作业中会用到的地方 |
|---|---|
| 广播、`[..., None]` | 几乎所有模块，处理任意前导 batch 维 |
| `x @ W.T`、einsum | Linear、注意力分数、FFN |
| `table[idx]` | Embedding |
| `split_groups` / `merge_groups` | 多头注意力的拆头 / 合头 |
| `fill_upper_triangle`、`masked_fill` | 因果掩码 |
| `pick_per_row` / `gather` | 交叉熵损失 |
| `nn.Parameter`、`trunc_normal_`、device/dtype | 所有带参数的模块 |
| `nn.ModuleList`、`state_dict` 命名 | Transformer 层堆叠、测试 adapter 加载权重 |
| `register_buffer(persistent=False)` | RoPE 的 cos/sin 缓存 |
| `zero_grad → backward → step` | 训练循环、自定义优化器 |

## 关于 AI 助手

本章是"开放区"：AI 助手可以为这里的练习给出完整代码讲解（不过仍建议先自己尝试）。
正式作业部分的规则见仓库根目录的 `AGENTS.md`。
