import sys
import numpy as np
# import theano
# import theano.tensor as T
# from theano.tensor.shared_randomstreams import RandomStreams
import torch
import torch.nn as nn
from datetime import datetime

class AdamTrainer:
    def __init__(self, rng=None, batchsize=16, epochs=100, alpha=0.001,
                 beta1=0.9, beta2=0.999, eps=1e-8, cost='mse'):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Handle different types of rng parameter
        if rng is None:
            self.rng = np.random
        elif hasattr(rng, 'randint') and hasattr(rng, 'shuffle'):
            # rng is a RandomState object
            self.rng = rng
        else:
            # rng is the np.random module or something else
            self.rng = rng if rng is not None else np.random

        self.epochs = epochs
        self.batchsize = batchsize

        if cost == 'mse':
            self.cost_fn = nn.MSELoss()
        elif cost == 'cross_entropy':
            self.cost_fn = nn.BCELoss()
        else:
            if callable(cost):
                self.cost_fn = cost
            else:
                raise ValueError(f"Unknown cost function: {cost}. Must be 'mse', 'cross_entropy', or a callable function.")

    def cost(self, network, x, y):
        pred = network(x)
        data_cost = self.cost_fn(pred, y)

        # Add regularization cost from network if it exists
        if hasattr(network, 'cost'):
            reg_cost = network.cost(x)
            return data_cost + reg_cost

        return data_cost

    def save(self, network, filename):
        torch.save(network.state_dict(), filename)

    def train(self, network, input_data, output_data, filename=None,
              restart=True, shuffle=True, silent=False):

        device = next(network.parameters()).device
        input_data = input_data.to(device)
        output_data = output_data.to(device)

        if restart or not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(
                network.parameters(),
                lr=self.alpha,
                betas=(self.beta1, self.beta2),
                eps=self.eps
            )

        n_batches = input_data.shape[0] // self.batchsize
        last_mean = 0.0

        for epoch in range(self.epochs):
            batchinds = np.arange(input_data.shape[0] // self.batchsize)
            print("Number of batches:", batchinds)
            costs = []

            for bii, bi in enumerate(batchinds):
                start_idx = bi * self.batchsize
                end_idx = start_idx + self.batchsize

                batch_input = input_data[start_idx:end_idx]
                batch_output = output_data[start_idx:end_idx]

                self.optimizer.zero_grad()
                cost = self.cost(network, batch_input, batch_output)
                cost.backward()
                self.optimizer.step()

                costs.append(cost.item())

                if not silent and bii % (max(n_batches // 1000, 1)) == 0:
                    sys.stdout.write('\r[Epoch %3i] % 3.1f%% mean %03.5f' %
                        (epoch, 100 * float(bii)/n_batches, np.mean(costs)))
                    sys.stdout.flush()

            curr_mean = np.mean(costs)
            if epoch > 0:
                diff_mean = curr_mean - last_mean
            else:
                diff_mean = 0
            last_mean = curr_mean

            sys.stdout.write('\r[Epoch %3i] 100.0%% mean %03.5f diff % .5f %s\n' %
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            sys.stdout.flush()

            if filename:
                self.save(network, filename)
