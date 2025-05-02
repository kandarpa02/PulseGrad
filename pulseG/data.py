import numpy as np
import random

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.idx:self.idx+self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.idx += self.batch_size
        data, labels = zip(*batch)
        return np.stack(data), np.stack(labels)