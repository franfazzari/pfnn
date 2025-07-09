import numpy as np
import torch
import torch.nn as nn

from Layer import Layer

class BiasLayer(nn.Module, Layer):
    def __init__(self, shape):
        super(BiasLayer, self).__init__()
        self.b = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.shape = shape

    def forward(self, input):
        # Create a view of the bias with proper broadcasting dimensions
        bias = self.b

        # Expand dimensions where shape is 1 to match input dimensions
        # We need to handle the case where input has batch dimension
        input_shape = input.shape
        bias_shape = list(self.shape)

        # If input has more dimensions than bias (e.g., batch dimension),
        # we need to add dimensions to bias for proper broadcasting
        while len(bias_shape) < len(input_shape):
            bias_shape.insert(0, 1)

        # Reshape bias to match the expected broadcasting shape
        bias = bias.view(bias_shape)

        return input + bias

    def cost(self, input):
        return torch.tensor(0.0)
