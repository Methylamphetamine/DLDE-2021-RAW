import jax.numpy as np
from jax import jit, random, grad
from functools import partial

class boundary_sampler:
    def __init__(self, batch_size, X, Y):
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        
        self.N = self.X.shape[0]
    @partial(jit, static_argnums=(0,))
    def sample(self, key):
        idx = random.choice(key, np.arange(self.N), shape = (self.batch_size,), replace = False)
        return (self.X[idx], self.Y[idx])

class residual_sampler:
    def __init__(self, batch_size, N):
        self.batch_size = batch_size
        self.N = N

    @partial(jit, static_argnums=(0,))
    def sample(self, key):
        X = random.uniform(key, (self.batch_size, 2)) * self.N

        return X

