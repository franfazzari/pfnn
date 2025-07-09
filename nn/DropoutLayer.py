import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Layer import Layer

class DropoutLayer(nn.Module, Layer):
    def __init__(self, amount=0.7, rng=None):
        super(DropoutLayer, self).__init__()
        # amount is the keep probability (opposite of PyTorch's p which is drop probability)
        self.amount = amount
        self.keep_prob = amount
        self.drop_prob = 1.0 - amount

        # If rng is provided, use it to seed PyTorch's random state
        if rng is not None and hasattr(rng, 'randint'):
            # rng is a numpy RandomState, use it to seed PyTorch
            seed = rng.randint(2**31)
            torch.manual_seed(seed)

    def forward(self, input):
        if self.amount < 1.0:
            # Use PyTorch's built-in dropout which handles training/eval mode automatically
            return F.dropout(input, p=self.drop_prob, training=self.training)
        else:
            return input

    def cost(self, input):
        return torch.tensor(0.0)
