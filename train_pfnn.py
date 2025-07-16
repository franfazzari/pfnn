import sys
import numpy as np
#import theano
#import theano.tensor as T
#theano.config.allow_gc = True

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('./nn')

from Layer import Layer
from HiddenLayer import HiddenLayer
from BiasLayer import BiasLayer
from DropoutLayer import DropoutLayer
from ActivationLayer import ActivationLayer
from AdamTrainer import AdamTrainer

# HyperParameters
JOINT_IMPORTANCE = 0.1
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 10

rng = np.random.RandomState(23456)
# Add device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Using CPU instead.")
""" Load Data """
database = np.load('database_flat.npz')
X = torch.tensor(database['Xun'], dtype=torch.float32)
Y = torch.tensor(database['Yun'], dtype=torch.float32)
P = torch.tensor(database['Pun'], dtype=torch.float32)

print(X.shape, Y.shape)

""" Calculate Mean and Std """

# Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
# Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)
Xmean, Xstd = X.mean(dim=0).numpy(), X.std(dim=0).numpy()
Ymean, Ystd = Y.mean(dim=0).numpy(), Y.std(dim=0).numpy()

j = 31
w = ((60*2)//10)

Xstd[w*0:w* 1] = Xstd[w*0:w* 1].mean() # Trajectory Past Positions
Xstd[w*1:w* 2] = Xstd[w*1:w* 2].mean() # Trajectory Future Positions
Xstd[w*2:w* 3] = Xstd[w*2:w* 3].mean() # Trajectory Past Directions
Xstd[w*3:w* 4] = Xstd[w*3:w* 4].mean() # Trajectory Future Directions
Xstd[w*4:w*10] = Xstd[w*4:w*10].mean() # Trajectory Gait

""" Mask Out Unused Joints in Input """

joint_weights = np.array([
    1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)

Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w*10+j*3*1].mean() / (joint_weights * JOINT_IMPORTANCE) # Pos
Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w*10+j*3*2].mean() / (joint_weights * JOINT_IMPORTANCE) # Vel
Xstd[w*10+j*3*2:          ] = Xstd[w*10+j*3*2:          ].mean() # Terrain

Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
Ystd[3:4] = Ystd[3:4].mean() # Change in Phase
Ystd[4:8] = Ystd[4:8].mean() # Contacts

Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean() # Trajectory Future Positions
Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean() # Trajectory Future Directions

Ystd[8+w*2+j*3*0:8+w*2+j*3*1] = Ystd[8+w*2+j*3*0:8+w*2+j*3*1].mean() # Pos
Ystd[8+w*2+j*3*1:8+w*2+j*3*2] = Ystd[8+w*2+j*3*1:8+w*2+j*3*2].mean() # Vel
Ystd[8+w*2+j*3*2:8+w*2+j*3*3] = Ystd[8+w*2+j*3*2:8+w*2+j*3*3].mean() # Rot

""" Save Mean / Std / Min / Max """

# Xmean.astype(np.float32).tofile('./demo/network/pfnn/Xmean.bin')
# Ymean.astype(np.float32).tofile('./demo/network/pfnn/Ymean.bin')
# Xstd.astype(np.float32).tofile('./demo/network/pfnn/Xstd.bin')
# Ystd.astype(np.float32).tofile('./demo/network/pfnn/Ystd.bin')
Xmean.astype(np.float32).tofile('./demo/network/pfnn/Xmean.bin')
Ymean.astype(np.float32).tofile('./demo/network/pfnn/Ymean.bin')
Xstd.astype(np.float32).tofile('./demo/network/pfnn/Xstd.bin')
Ystd.astype(np.float32).tofile('./demo/network/pfnn/Ystd.bin')

""" Normalize Data """

# X = (X - Xmean) / Xstd
# Y = (Y - Ymean) / Ystd
X = (X - torch.from_numpy(Xmean)) / torch.from_numpy(Xstd)
Y = (Y - torch.from_numpy(Ymean)) / torch.from_numpy(Ystd)

""" Phase Function Neural Network """

class PhaseFunctionedNetwork(nn.Module):
    def __init__(self, input_shape=1, output_shape=1, dropout=0.7):
        super(PhaseFunctionedNetwork, self).__init__()
        self.nslices = 4
        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights and biases for all slices
        self.W0 = nn.Parameter(torch.randn(self.nslices, 512, input_shape-1) * 0.1)
        self.W1 = nn.Parameter(torch.randn(self.nslices, 512, 512) * 0.1)
        self.W2 = nn.Parameter(torch.randn(self.nslices, output_shape, 512) * 0.1)

        self.b0 = nn.Parameter(torch.zeros(self.nslices, 512))
        self.b1 = nn.Parameter(torch.zeros(self.nslices, 512))
        self.b2 = nn.Parameter(torch.zeros(self.nslices, output_shape))

        self.activation = nn.ELU()

    def cubic(self, y0, y1, y2, y3, mu):
        return ((-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu +
                (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu +
                (-0.5*y0+0.5*y2)*mu + y1)

    def forward(self, x):
        # Split input into data and phase
        data, phase = x[:, :-1], x[:, -1]

        pscale = self.nslices * phase
        pamount = pscale % 1.0

        pindex_1 = (pscale.long() % self.nslices).long()
        pindex_0 = (pindex_1-1) % self.nslices
        pindex_2 = (pindex_1+1) % self.nslices
        pindex_3 = (pindex_1+2) % self.nslices

        Wamount = pamount.unsqueeze(1).unsqueeze(2)
        bamount = pamount.unsqueeze(1)

        # Compute interpolated weights and biases
        W0 = self.cubic(self.W0[pindex_0], self.W0[pindex_1],
                       self.W0[pindex_2], self.W0[pindex_3], Wamount)
        W1 = self.cubic(self.W1[pindex_0], self.W1[pindex_1],
                       self.W1[pindex_2], self.W1[pindex_3], Wamount)
        W2 = self.cubic(self.W2[pindex_0], self.W2[pindex_1],
                       self.W2[pindex_2], self.W2[pindex_3], Wamount)

        b0 = self.cubic(self.b0[pindex_0], self.b0[pindex_1],
                       self.b0[pindex_2], self.b0[pindex_3], bamount)
        b1 = self.cubic(self.b1[pindex_0], self.b1[pindex_1],
                       self.b1[pindex_2], self.b1[pindex_3], bamount)
        b2 = self.cubic(self.b2[pindex_0], self.b2[pindex_1],
                       self.b2[pindex_2], self.b2[pindex_3], bamount)

        # Forward pass
        H0 = data
        H1 = self.activation(torch.bmm(W0, self.dropout(H0.unsqueeze(2))).squeeze(2) + b0)
        H2 = self.activation(torch.bmm(W1, self.dropout(H1.unsqueeze(2))).squeeze(2) + b1)
        H3 = torch.bmm(W2, self.dropout(H2.unsqueeze(2))).squeeze(2) + b2

        return H3

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


""" Function to Save Network Weights """

def save_network(network):
    # Similar function but using PyTorch tensors
    for i in range(50):
        pscale = network.nslices*(float(i)/50)
        pamount = pscale % 1.0

        pindex_1 = int(pscale) % network.nslices
        pindex_0 = (pindex_1-1) % network.nslices
        pindex_2 = (pindex_1+1) % network.nslices
        pindex_3 = (pindex_1+2) % network.nslices

        W0 = network.cubic(network.W0[pindex_0], network.W0[pindex_1],
                          network.W0[pindex_2], network.W0[pindex_3], pamount)
        W1 = network.cubic(network.W1[pindex_0], network.W1[pindex_1],
                          network.W1[pindex_2], network.W1[pindex_3], pamount)
        W2 = network.cubic(network.W2[pindex_0], network.W2[pindex_1],
                          network.W2[pindex_2], network.W2[pindex_3], pamount)

        b0 = network.cubic(network.b0[pindex_0], network.b0[pindex_1],
                          network.b0[pindex_2], network.b0[pindex_3], pamount)
        b1 = network.cubic(network.b1[pindex_0], network.b1[pindex_1],
                          network.b1[pindex_2], network.b1[pindex_3], pamount)
        b2 = network.cubic(network.b2[pindex_0], network.b2[pindex_1],
                          network.b2[pindex_2], network.b2[pindex_3], pamount)

        W0.cpu().detach().numpy().astype(np.float32).tofile(f'./demo/network/pfnn/W0_{i:03d}.bin')
        W1.cpu().detach().numpy().astype(np.float32).tofile(f'./demo/network/pfnn/W1_{i:03d}.bin')
        W2.cpu().detach().numpy().astype(np.float32).tofile(f'./demo/network/pfnn/W2_{i:03d}.bin')
        b0.cpu().detach().numpy().astype(np.float32).tofile(f'./demo/network/pfnn/b0_{i:03d}.bin')
        b1.cpu().detach().numpy().astype(np.float32).tofile(f'./demo/network/pfnn/b1_{i:03d}.bin')
        b2.cpu().detach().numpy().astype(np.float32).tofile(f'./demo/network/pfnn/b2_{i:03d}.bin')

""" Training Loop """

network = PhaseFunctionedNetwork(input_shape=X.shape[1]+1, output_shape=Y.shape[1], dropout=DROPOUT).to(device)
#network.load('./demo/network/pfnn/network.pt')
#save_network(network)
#sys.exit()
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

batchsize = 128

for me in range(EPOCHS):
    I = torch.randperm(len(X))
    print('\n[MacroEpoch] %03i' % me)
    for bi in range(10):
        start, stop = ((bi+0)*len(I))//10, ((bi+1)*len(I))//10
        idx = I[start:stop]
        print('Batch %03i' % bi)
        # Create input tensor by concatenating X and P
        input_data = torch.cat([X[idx], P[idx].unsqueeze(1)], dim=1).to(device)
        target = Y[idx].to(device)

        for i in range(0, len(idx), batchsize):
            batch_input = input_data[i:i+batchsize]
            batch_target = target[i:i+batchsize]

            optimizer.zero_grad()
            output = network(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

        # Save network
    network.save('./demo/network/pfnn/network%03d.pt' % me)
save_network(network)
