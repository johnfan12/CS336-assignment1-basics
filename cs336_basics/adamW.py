import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

# 这套 API 约定同样适用于下一节的 AdamW
class AdamW(torch.optim.Optimizer):
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
        for group in self.param_groups: # 遍历不同的参数组: 不同组可以指定不同的超参数，比如第一层参数用小一点的 lr，最后一层用大一点的 lr
            lr = group["lr"] # group 也是一个字典，取其中的 lr
            for p in group["params"]: # 遍历该组内的每个待优化参数 p
                if p.grad is None: # 梯度还没有计算出来，grad 属性为 None，跳过更新
                    continue

                state = self.state[p]  # self.state 字典：保存每个参数对应的状态信息，如迭代步数、动量等；state[p] 获取参数 p 对应的状态字典
                t = state.get("t", 0)  # 读取迭代计数 t；如果key "t" 不存在，返回默认值 0
                grad = p.grad.data  # 获取损失 loss 关于参数 p 的梯度张量
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新参数：学习率随迭代步数做平方根衰减
                state["t"] = t + 1  # 更新状态里的迭代计数，步数 +1

        return loss

# 测试 SGD：对比不同学习率下 toy 损失的收敛情况
for lr in [1e1, 1e2, 1e3]: # 1*10 1*10^2 1*10^3
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=lr)
    print(f"--- lr={lr} ---")
    for t in range(10):
        opt.zero_grad()             # 清空上一步的梯度
        loss = (weights**2).mean()  # 构造一个标量损失
        loss.backward()             # 反向传播，计算梯度
        opt.step()                  # 优化器更新参数
        if t % 3 == 0 or t == 9:
            print(f"  step {t:2d}: loss = {loss.item():.3e}")