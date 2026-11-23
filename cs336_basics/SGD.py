import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

# 这套 API 约定同样适用于下一节的 AdamW
class SGD(torch.optim.Optimizer):
    '''
    PyTorch 中的优化器统一继承 torch.optim.Optimizer，需要实现两个方法：
    __init__(self, params, ...): 使用基类的 init 方法保存待优化参数和默认超参；
    step(self): 执行一次参数更新
    '''

    def __init__(self, params, lr=1e-3):
        """
        自定义SGD优化器构造函数
        :param params: 待优化的参数集合，可以是模型全部参数，也可以是分组参数
        :param lr: 学习率，超参数
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr} # 以字典的形式存储超参名（key）与超参默认值（value)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        执行单步参数更新，反向传播完成 grad 计算之后调用此函数
        :param closure: 可选的闭包函数，用于重新计算 loss 并返回 loss 值，部分优化器需要使用（我们不用）
        :return: 如果传入 closure，返回 loss，否则返回 None
        """
        loss = None if closure is None else closure()
        for group in self.param_groups: 
            lr = group["lr"] 
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] 
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

# 测试 SGD：对比不同学习率下 toy 损失的收敛情况
for lr in [1e1, 1e2, 1e3]: # 1*10 1*10^2 1*10^3
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    print(f"--- lr={lr} ---")
    for t in range(10):
        opt.zero_grad()             # 清空上一步的梯度
        loss = (weights**2).mean()  # 构造一个标量损失
        loss.backward()             # 反向传播，计算梯度
        opt.step()                  # 优化器更新参数
        if t % 3 == 0 or t == 9:
            print(f"  step {t:2d}: loss = {loss.item():.3e}")