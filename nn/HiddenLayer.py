import numpy as np
import torch
import torch.nn as nn

from Layer import Layer

class HiddenLayer(nn.Module, Layer):
    def __init__(self, weights_shape, rng=None, gamma=0.01):
        super(HiddenLayer, self).__init__()

        # Handle different types of rng parameter
        if rng is None:
            rng = np.random
        elif hasattr(rng, 'uniform'):
            # rng is already a RandomState object, use it directly
            pass
        else:
            # rng is the np.random module, use it directly
            pass

        W_bound = np.sqrt(6. / np.prod(weights_shape[-2:]))
        W = torch.tensor(
            rng.uniform(low=-W_bound, high=W_bound, size=weights_shape),
            dtype=torch.float32)

        self.W = nn.Parameter(W)
        self.gamma = gamma

    def cost(self, input):
        return self.gamma * torch.mean(torch.abs(self.W))

    def forward(self, input):
        # Equivalent to theano's self.W.dot(input.T).T which is input @ W.T
        if len(self.W.shape) == 3:
            # W has shape (nslices, out_features, in_features)
            # input has shape (batch, in_features)
            # We want to apply each slice to the input
            # Result: (batch, nslices, out_features)
            batch_size = input.shape[0]
            input_expanded = input.unsqueeze(1).expand(-1, self.W.shape[0], -1)  # (batch, nslices, in_features)
            W_transposed = self.W.transpose(-2, -1)  # (nslices, in_features, out_features)

            # Batch matrix multiplication: (batch, nslices, in_features) @ (nslices, in_features, out_features)
            # We need to reshape for bmm
            input_flat = input_expanded.reshape(-1, input.shape[-1])  # (batch*nslices, in_features)
            W_flat = W_transposed.reshape(-1, W_transposed.shape[-1])  # (nslices*in_features, out_features)

            # Use einsum for tensor contraction: input(b,i) * W(s,o,i) -> output(b,s,o)
            result = torch.einsum('bi,soi->bso', input, self.W)
            return result
        else:
            # W has shape (out_features, in_features)
            # input has shape (batch, in_features)
            return torch.matmul(input, self.W.t())
