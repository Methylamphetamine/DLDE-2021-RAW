import jax.numpy as np
from jax import random, grad, vmap, jit, lax, jacobian, eval_shape
from jax.tree_util import tree_map, tree_multimap, tree_reduce 
from jax.flatten_util import ravel_pytree
from jax.ops import index_update, index
from jax.experimental import optimizers
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import time
from functools import partial


def MLP(layers, activation, off_set = 1., b_init = 0.):
    '''
    Vanilla MLP with fan_in initialization
    '''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = off_set / np.sqrt(d_in)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = b_init * np.ones(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = inputs@W + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = inputs@W + b
        return outputs
    return init, apply

def Feature_Net(layers, activation, off_set = 1., b_init = 0.):
    '''
    Vanilla MLP with fan_in initialization
    '''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = off_set / np.sqrt(d_in)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = b_init * np.ones(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params:
            outputs = inputs@W + b
            inputs = activation(outputs)

        return inputs
    return init, apply


def init_net(params, activation):
    '''
    return output with fixed matrix directions but trainable scale and bias.
    '''
    def init(key):
        p_list = []
        keys = random.split(key)
        for (w, b) in params:
            
            init_w = np.exp(0.25 * random.normal(keys[0]))
            init_b = 0.1 * random.normal(keys[1])
            p_list.append((init_w, init_b))
            
            keys = random.split(keys[0])
        return p_list
    def apply(p_list, params, X):
        inputs = X
        
        for (w, b), (w_init, b_init) in zip(params, p_list):
            outputs = w_init * (inputs @ w) + b + b_init
            inputs = activation(outputs)
        
        w, b = params[-1]
        w_init, b_init = p_list[-1]
        
        return activation(w_init * (inputs @ w) + b + b_init)
    return init, apply

def parameter_scaling(params, scale_params):
    return [(w * w_init, b + b_init)for (w, b), (w_init, b_init) in zip(params, scale_params)]

def layer_weight_norm(params):
    return [np.linalg.norm(w) for (w, b) in params]

class initNet:
    def __init__(self, layers, activation, key = random.PRNGKey(0)):
        
        self.activation = activation
        self.layers = layers
        self.key = key
        self.net_init, self.net_apply = Feature_Net(self.layers, self.activation)
        self.net_params = self.net_init(self.key)
        
        self.key, _ = random.split(self.key)
        
        self.scale_init, self.scale_apply = init_net(self.net_params, self.activation)
        self.scale_params = self.scale_init(self.key)
        
        self.key, _ = random.split(self.key)

        
        
        
    def cosines(self, scale_params, params, X):
        out = self.scale_apply(scale_params, params, X)
        out = out / (np.linalg.norm(out, axis = 0) + 1e-7)
        c = out.T @ out
        c = index_update(c,index[np.arange(self.layers[-1]), np.arange(self.layers[-1])], 0)
        return np.clip(c, a_max = 1-1e-7, a_min = 1e-7 - 1)
    
    def angles(self, scale_params, params, X):
        return np.degrees(np.arccos(np.abs(self.cosines(scale_params, params, X))))
    
    @partial(jit, static_argnums=(0,))
    def logSineLoss(self, scale_params, params, X):
        sines_square = 1 - self.cosines(scale_params, params, X)**2
        return -np.log(1e-7 + sines_square).mean()
    @partial(jit, static_argnums=(0,))
    def regulated_logSineLoss(self, scale_params, params, X, lam = 0.001):
        ws = [w for (w, b) in scale_params]
        return self.logSineLoss(scale_params, params, X) + lam * optimizers.l2_norm(ws)**2
    
    
    def plot_degree(self, scale_params, params, X, figsize = (5,4), word = True):
        
        plt.figure(figsize=figsize)

        degree = self.angles(scale_params, params, X)
        
        m = plt.matshow(degree, fignum = 1, vmin = 0, vmax = 90)
        if word:

            for (x, y), value in onp.ndenumerate(degree):
                plt.text(x, y, f"{value:.1f}", va="center", ha="center", color = 'r')
        c = plt.colorbar(m)
        plt.title(f"Angle between basis. Log sine loss: {self.logSineLoss(scale_params, params, X):.2e}")

        plt.show()
        plt.close()
    
    @partial(jit, static_argnums=(0,))
    def hidden_output(self, params, X):
        '''
        params should be organized as ((w,b),...)
        '''
        p_list = []
        
        inputs = X
        for W, b in params:
            outputs = inputs@W + b
            inputs = self.activation(outputs)
            p_list.append(inputs)
        return p_list
    
class MLPRegression:
    def __init__(self, layers, activation, key = random.PRNGKey(1)):
        
        self.layers = layers
        self.net_init, self.net_apply = MLP(layers, activation=activation)
        self.net_params = self.net_init(key)
        self.key = key
        self.key,_ = random.split(self.key)
        
        self.activation = activation

    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        X, y = batch

        pred = self.net_apply(params, X).flatten()
        return ((pred - y.flatten())**2).mean()
    
    def scalar_out(self, params, x):
        X = np.array([x])
        return self.net_apply(params, X)[0,0]
    
    def scalar_grad(self, params, x):
        return ravel_pytree(grad(self.scalar_out)(params, x))[0]
    @partial(jit, static_argnums=(0,))
    def ntk(self, params, X):
        j = vmap(self.scalar_grad, in_axes = [None, 0])(params, X)
        return j@j.T
    @partial(jit, static_argnums=(0,))
    def l2_error(self, params, batch):
        X, Y = batch
        
        pred = self.net_apply(params, X)
        
        return np.linalg.norm(Y - pred) / np.linalg.norm(Y)
        
