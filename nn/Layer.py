import numpy as np
import torch

class Layer(object):

    def cost(self, input):
        return torch.tensor(0.0)

    def load(self, database, prefix=''):
        # For PyTorch parameters, we need to load state dict items
        if hasattr(self, 'named_parameters') and callable(getattr(self, 'named_parameters')):
            for name, param in self.named_parameters():
                key = prefix + name
                if key in database:
                    param.data = torch.from_numpy(database[key].astype(np.float32))

    def save(self, database, prefix=''):
        # For PyTorch parameters, we save the numpy arrays
        if hasattr(self, 'named_parameters') and callable(getattr(self, 'named_parameters')):
            for name, param in self.named_parameters():
                database[prefix + name] = param.detach().cpu().numpy()
