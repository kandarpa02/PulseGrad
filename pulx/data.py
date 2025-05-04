import jax.numpy as jnp
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

        batch_indices = self.indices[self.idx:self.idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.idx += self.batch_size

        # Check if it's labeled data (tuple) or unlabeled (single item)
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            data, labels = zip(*batch)
            return jnp.array(data), jnp.array(labels)
        else:
            return jnp.array(batch)
