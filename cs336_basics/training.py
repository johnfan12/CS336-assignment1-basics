import torch
from cs336_basics.utils import get_batch, learning_rate_schedule, cross_entropy, gradient_clip
import numpy as np


# 单步训练：把一次参数更新封装起来（内部的前向、损失、反向、裁剪、更新在前面几节都实现过了）
def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    max_l2_norm: float = 1.0,
) -> float:
    '''
    model: torch.nn.Module  语言模型
    optimizer: torch.optim.Optimizer  优化器
    x: (batch_size, context_length)  输入 token ID，需为 long 类型
    y: (batch_size, context_length)  目标 token ID，即 x 整体右移一格
    lr: float  本步使用的学习率（由余弦调度给出）
    max_l2_norm: float  M，梯度裁剪允许的最大 ℓ2 范数
    return: float  本步的损失值（标量）
    '''
    optimizer.param_groups[0]["lr"] = lr
    optimizer.zero_grad()

    logits = model(x)
    loss = cross_entropy(logits, y).mean()
    loss.backward()
    gradient_clip(model.parameters(), max_l2_norm)
    optimizer.step()

    return loss.item()


# 训练循环：反复调用 train_step
def train_llm(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    args,
) -> None:
    '''
    model / optimizer: 模型与优化器
    train_data / val_data: np.ndarray  token ID 数组，建议用 np.load(路径, mmap_mode='r') 打开
    args: 超参集合，建议用 argparse 以命令行参数的形式接收
    '''
    for i in range(args.total_iters):
        model.train()

        # 1. 取一批训练数据，并按余弦调度算出本步学习率
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        lr = learning_rate_schedule(i, args.max_lr, args.min_lr,
                                    args.warmup_iters, args.cosine_cycle_iters)

        # 2. 单步训练：前向 → 损失 → 反向 → 裁剪 → 更新
        loss = train_step(model, optimizer, x.long(), y.long(),
                          lr=lr, max_l2_norm=args.max_l2_norm)

        # 3. 每隔若干步：在验证集上评估一次、保存一次检查点、记录一次日志
        if i % args.val_interval == 0:
            ...