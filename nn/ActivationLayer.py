import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Layer import Layer

class ActivationLayer(nn.Module, Layer):
    def __init__(self, f='ReLU', params=[]):
        super(ActivationLayer, self).__init__()

        if f == 'ReLU':
            self.f = lambda x: torch.clamp(x, min=0)
        elif f == 'LReLU':
            self.f = lambda x: torch.where(x < 0, 0.01 * x, x)
        elif f == 'ELU':
            self.f = lambda x: torch.where(x < 0, torch.exp(x) - 1, x)
        elif f == 'softplus':
            self.f = F.softplus
        elif f == 'tanh':
            self.f = torch.tanh
        elif f == 'sigmoid':
            self.f = torch.sigmoid
        elif f == 'identity':
            self.f = lambda x: x
        else:
            if callable(f):
                self.f = f
            else:
                raise ValueError(f"Unknown activation function: {f}. Must be one of: 'ReLU', 'LReLU', 'ELU', 'softplus', 'tanh', 'sigmoid', 'identity', or a callable function.")

    def forward(self, input):
        return self.f(input)

    def cost(self, input):
        return torch.tensor(0.0)
