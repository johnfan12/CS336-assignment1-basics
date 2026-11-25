import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

# 这套 API 约定同样适用于下一节的 AdamW
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas:tuple[float, float]=(0.99, 0.999), eps:float=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data
                t += 1

                alpha_t = lr * ((1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t))

                p.data -= lr * weight_decay * p.data

                m = beta1 * m + (1 - beta1) * grad

                v = beta2 * v + (1 - beta2) * grad ** 2

                p.data -= alpha_t * m /(torch.sqrt(v) + eps)

                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss