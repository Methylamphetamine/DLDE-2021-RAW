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

# Architectures
def MLP(layers, activation, dist, off_set = 1., b_init = 0.):
    '''
    Vanilla MLP with fan_in initialization
    '''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = off_set / np.sqrt(d_in)
            W = glorot_stddev * dist(k1, (d_in, d_out))
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

def Feature_Net(layers, activation, dist, off_set = 1., b_init = 0.):
    '''
    Vanilla MLP with fan_in initialization. The output is post-activation last layer output.
    '''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = off_set / np.sqrt(d_in)
            W = glorot_stddev * dist(k1, (d_in, d_out))
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
    return output with fixed matrix directions but trainable scale and bias for feature_net.
    '''
    def init(key):
        p_list = []
        keys = random.split(key)
        for (w, b) in params:
            
            init_w = np.exp(0.25 * random.normal(keys[0]))
            init_b = 0.05 * random.normal(keys[1])
            p_list.append((init_w, init_b))
            
            keys = random.split(keys[0])
        return p_list
    def apply(p_list, params, X):
        inputs = X
        
        for (w, b), (w_init, b_init) in zip(params, p_list):
            outputs = w_init * (inputs @ w) + b + b_init
            inputs = activation(outputs)

        return inputs
    return init, apply

def parameter_scaling(params, scale_params):
    '''
    return feature_net parameter tree given the scaling parameters and the network parameters from Xavier.
    '''
    return [(w * w_init, b + b_init)for (w, b), (w_init, b_init) in zip(params, scale_params)]

def full_scaling(params, scale_params):
    '''
    scale the hidden layers parameters before the output layer of a normal MLP.
    '''
    return parameter_scaling(params[:-1], scale_params) + [params[-1]]


# Tree utilities
def cat_fn(*x):
    '''
    concatenate the given arrays.
    '''
    return np.concatenate(x)
def repeat_fn(x, rep):
    '''
    repeat the given arrays.
    '''
    return np.array([x for _ in range(rep)])

def stack_fn(*x):
    '''
    stack the given arrays
    '''
    return np.array(x)


def layer_weight_norm(params):
    '''
    return the norm of layer weights matrices.
    '''
    return [np.linalg.norm(w) for (w, b) in params]

def layer_bias_norm(params):
    '''
    return the norm of layer bias vector.
    '''
    return [np.linalg.norm(b) for (w, b) in params]

class initNet:
    '''
    class object for pre-training.
    '''
    def __init__(self, layers, activation, dist = lambda key, shape: random.normal(key, shape), key = random.PRNGKey(0)):
        
        self.activation = activation
        self.layers = layers
        self.key = key
        self.net_init, self.net_apply = Feature_Net(self.layers, self.activation, dist = dist)
        self.net_params = self.net_init(self.key)
        
        self.key, _ = random.split(self.key)
        
        self.scale_init, self.scale_apply = init_net(self.net_params, self.activation)
        self.scale_params = self.scale_init(self.key)
        
        self.key, _ = random.split(self.key)

        
        
        
    def cosines(self, scale_params, params, X):
        '''
        return cosine matrix of last hidden layer output features on data distribution.
        '''
        out = self.scale_apply(scale_params, params, X)
        out = out / (np.linalg.norm(out, axis = 0) + 1e-7)
        c = out.T @ out
        c = index_update(c,index[np.arange(self.layers[-1]), np.arange(self.layers[-1])], 0)
        return np.clip(c, a_max = 1-1e-7, a_min = 1e-7 - 1)
    
    def angles(self, scale_params, params, X):
        return np.degrees(np.arccos(np.abs(self.cosines(scale_params, params, X))))
    
    @partial(jit, static_argnums=(0,))
    def logSineLoss(self, scale_params, params, X):
        '''
        average of log sine value of the cosine matrix
        '''
        sines_square = 1 - self.cosines(scale_params, params, X)**2
        return -np.log(1e-7 + sines_square).mean()
    @partial(jit, static_argnums=(0,))
    def regulated_logSineLoss(self, scale_params, params, X, lam = 0.001):
        '''
        regularized log sine loss
        '''
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
        return hidden layer output
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
    '''
    class object for regression
    '''
    def __init__(self, layers, activation, dist = lambda key, shape: random.normal(key, shape), key = random.PRNGKey(1)):
        
        self.layers = layers
        self.net_init, self.net_apply = MLP(layers, activation=activation, dist = dist)
        self.net_params = self.net_init(key)
        self.key = key
        self.key,_ = random.split(self.key)
        
        self.activation = activation

    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        X, y = batch

        pred = self.net_apply(params, X).flatten()
        return ((pred - y.flatten())**2).mean()
    # NTK utilities
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
        
        pred = self.net_apply(params, X).flatten()
        
        return np.linalg.norm(Y.flatten() - pred) / np.linalg.norm(Y.flatten())

class fitPinns:
    '''
    class object for regression
    '''
    def __init__(self, layers, activation, mu_X, sigma_X, pde, pde_args, dist = lambda key, shape: random.normal(key, shape), key = random.PRNGKey(1)):
        '''
        pde is a functional that takes in a function f(params, x) and returns the residual function with arguments (params, x). 
        f is a function that returns output for one point.
        '''
        
        self.layers = layers
        self.net_init, net_apply = MLP(layers, dist = dist, activation=activation)
        self.net_params = self.net_init(key)
        self.key = key
        self.key,_ = random.split(self.key)
        self.activation = activation
        self.mu_X, self.sigma_X = mu_X, sigma_X
        self.net_apply = lambda params, X: net_apply(params, (X - self.mu_X) / self.sigma_X)
        self.residual = pde(self.scalar_out, *pde_args)
        

    @partial(jit, static_argnums=(0,))
    def boundary_loss(self, params, batch):
        X, y = batch

        pred = self.net_apply(params, X).flatten()
        return ((pred - y.flatten())**2).mean()
    @partial(jit, static_argnums=(0,))
    def residual_loss(self, params, X):
        residual = vmap(self.residual, in_axes = [None, 0])(params, X)
        return np.mean(residual**2)
    # NTK utilities
    def scalar_out(self, params, x):
        X = np.array([x])
        return self.net_apply(params, X)[0]
    def loss(self, params, batch, ws):
        (X, y), res_X = batch
        bcs_loss = self.boundary_loss(params, (X, y))
        rcs_loss = self.residual_loss(params, res_X)
        return bcs_loss * ws[0] + rcs_loss * ws[1]
    

    @partial(jit, static_argnums=(0,))
    def l2_error(self, params, batch):
        X, Y = batch
        
        pred = self.net_apply(params, X).flatten()
        
        return np.linalg.norm(Y.flatten() - pred) / np.linalg.norm(Y.flatten())
    
class normInitNet(initNet):
    def __init__(self, layers, activation, mu_X, sigma_X, dist = lambda key, shape: random.normal(key, shape), key = random.PRNGKey(0)):
        super().__init__(layers, activation, dist = dist, key = random.PRNGKey(0))
        self.mu_X, self.sigma_X = mu_X, sigma_X
        _, scale_apply = init_net(self.net_params, self.activation)
        _, net_apply = Feature_Net(self.layers, self.activation, dist = dist)
        
        self.net_apply = lambda params, X: net_apply(params, (X - self.mu_X) / self.sigma_X)
        self.scale_apply = lambda scale_params, params, X: scale_apply(scale_params, params, (X - self.mu_X) / self.sigma_X)
        
