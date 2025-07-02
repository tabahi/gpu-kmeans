import torch
import numpy as np
'''
This class implements a very large array that can grow dynamically.
It is designed to handle large datasets that may not fit into memory all at once.
It uses PyTorch tensors for efficient memory management and GPU support.
'''

class VeryLargeArray:
    
    def __init__(self, features_dim=None, init_capacity=1000, device='cpu', fill_value=None, reloaded_features=None):
        
        self.capacity = init_capacity
        self.increase_by = init_capacity
        self._fill_value = fill_value

        if reloaded_features is not None:
            if isinstance(reloaded_features, np.ndarray):
                reloaded_features = torch.tensor(reloaded_features, dtype=torch.float, device=self.device)
            
            self.features = reloaded_features
            self.device = reloaded_features.device
            self.features_dim = reloaded_features.shape[1]
            self.idx = reloaded_features.shape[0]
            return
        elif features_dim is None:
            raise ValueError("Either features_dim or reloaded_features must be provided")
        
        self.features_dim = features_dim
        self.device = device
        self.features = torch.zeros((self.capacity, self.features_dim), dtype=torch.float, device=self.device) #can use either torch or np
        
        if fill_value is not None:
            self.features.fill_(fill_value)
            
        self.idx = 0

    def increase_capacity(self):
        new_capacity = int(self.capacity + self.increase_by)
        features_new = torch.zeros((new_capacity, self.features_dim), dtype=torch.float, device=self.device)

        if self._fill_value is not None:
            features_new.fill_(self._fill_value)

        features_new[:self.capacity] = self.features
        self.features = features_new
        del features_new

        self.capacity = new_capacity

    def add(self, features):
        idx = self.idx
        self.features[idx,:] = features
        self.idx += 1

        if (self.idx == self.capacity): self.increase_capacity()


    def add_batch(self, features):
        batch_size = features.shape[0]
        if (self.idx + batch_size >= self.capacity): self.increase_capacity()

        self.features[self.idx:self.idx+batch_size,:] = features
        self.idx += batch_size
        if (self.idx > self.capacity):
            raise Exception("Error: VeryLargeArray capacity exceeded")
        
        if (self.idx + batch_size >= self.capacity): self.increase_capacity()

    def __len__(self):
        return self.idx
    
    def finalize(self):
        self.features = self.features[:self.idx]
        self.capacity = self.idx
        return self.features