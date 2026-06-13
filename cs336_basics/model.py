import torch
import torch.nn as nn

class Linear(torch.nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        w = torch.empty(d_out, d_in, device=device, dtype=dtype)
        self.W = nn.Parameter(w)
        sigma = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.W)
        
    