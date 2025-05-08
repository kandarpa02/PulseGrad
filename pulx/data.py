import jax
import jax.numpy as jnp

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = jax.random.PRNGKey(seed)
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size
        data, labels = zip(*[dataset[i] for i in range(len(dataset))])
        self.data = jnp.array(data)
        self.labels = jnp.array(labels)

    def __iter__(self):
        return self.get_batches()

    def get_batches(self):
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            indices = jax.random.permutation(subkey, self.num_samples)
        else:
            indices = jnp.arange(self.num_samples)

        data_shuffled = self.data[indices]
        labels_shuffled = self.labels[indices]

        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield data_shuffled[start:end], labels_shuffled[start:end]
