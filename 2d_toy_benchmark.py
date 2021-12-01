
import numpy as onp
from numpy import fft
import jax.numpy as np
from jax import random, grad, vmap, jit, lax, pmap
from jax.experimental import optimizers
from jax.experimental.optimizers import make_schedule
from jax.experimental.ode import odeint
from jax.nn import relu, leaky_relu, swish, sigmoid
from jax.config import config

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from jax import jacobian, eval_shape
from jax.tree_util import tree_map, tree_multimap, tree_reduce 
from jax.flatten_util import ravel_pytree
from jax.ops import index_update, index

from jaxpinns.optimizers import mdmm_adam

import neural_tangents as nt
import operator

from jax.scipy.stats.norm import logpdf, pdf


from tqdm import trange, tqdm

import time
from IPython.display import clear_output


# In[4]:


path = '2d_toy/'




# In[6]:


from helpers import *


# In[7]:


class sampler:
    def __init__(self, X, Y, batch_size):

        self.X = X
        self.Y = Y
        self.batch_size = batch_size
    @partial(jit, static_argnums=(0,))
    def sample(self, key):
        idx = random.choice(key, np.arange(self.X.shape[0]), shape = (batch_size,), replace = False)
        return (self.X[idx], self.Y[idx])


# In[8]:


key = random.PRNGKey(int(time.time()*100))


# In[9]:


def target_fn_gen(max_freq, key = random.PRNGKey(0)):
    k1, k2 = random.split(key)
    As = random.uniform(k1, shape = (4,2)) * 2
    fs = random.uniform(k2, shape = (4, 4)) * max_freq
    
    def target_fn(X):
        cs = []
        for A, f in zip(As, fs):
            cs.append(A[0] * np.sin(np.pi * f[0] * X[:, 0]) * np.cos(np.pi * f[1] * X[:,1]) + A[1] * np.sin(np.pi * f[2] * X[:, 0]) * np.cos(np.pi * f[3] * X[:,1]))
        return np.array(cs).sum(0)
    return target_fn


# In[10]:


max_freq = 20
target_fn = target_fn_gen(max_freq, key = key)


# In[11]:


X_raw = random.uniform(key, shape = (2**16, 2), minval = -1, maxval = 1)
mu_X, sigma_X = X_raw.mean(0), X_raw.std(0)
X = (X_raw - mu_X) / sigma_X
X_test_raw = np.array(np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))).transpose((1,2,0)).reshape(-1,2)
X_test = (X_test_raw - mu_X) / sigma_X

Y = target_fn(X_raw)
Y = (Y - Y.mean(0)) / Y.std(0)


depth = 16

layers = [X_raw.shape[1], *[256 for _ in range(depth)], 1]
lams = [1e-5, 1e-4, 1e-3, 1e-2]
activation = np.tanh
print(f'Candidate lam: {lams}')
for lam in lams:
    
    fit_model = MLPRegression(layers, activation = activation)


    # In[14]:


    batch_size = 2048
    data_sampler = sampler(X, Y, batch_size)


    # In[15]:


    init_model = initNet(layers[:-1], activation = activation)
    #init_model.plot_degree(init_model.scale_params, fit_model.net_params[:-1], X, word = False)


    # In[16]:


    lr = 5e-3
    init_fn, update_fn, get_params = optimizers.adam(lr)
    # init_fn, update_fn, get_params = optimizers.nesterov(lr, 0.9)

    
    @jit
    def step(i, state, X):
        key = random.PRNGKey(i)
        g = grad(init_model.regulated_logSineLoss)(get_params(state), init_model.net_init(key), X, lam = lam)
        return update_fn(i, g, state)
    print(f'Important: confirm lam {lam} before running!!!!!!!!!!!!!!!')


    # In[17]:


    opt_state = init_fn(init_model.scale_params)
    init_nIter = 20000
    pbar = trange(init_nIter)
    scale_stor = []
    bias_stor = []

    for i in pbar:
        mini_batch = data_sampler.sample(random.PRNGKey(i))[0]
        opt_state = step(i, opt_state, mini_batch)

        if i % 100 == 0:
            params = get_params(opt_state)
            pbar.set_postfix({'Log sine loss': init_model.logSineLoss(get_params(opt_state), init_model.net_init(random.PRNGKey(i)), mini_batch)})
            scale_stor.append([w for (w, b) in params])
            bias_stor.append([b for (w, b) in params])


    scale_stor = np.array(scale_stor)
    bias_stor = np.array(bias_stor)






    scale_params = params





    fit_model.scaled_net_params = parameter_scaling(fit_model.net_params[:-1], scale_params) + [fit_model.net_params[-1]]





    normal_init_params = vmap(fit_model.net_init)(random.split(key, 4))
    scale_init_params = vmap(full_scaling, in_axes = (0, None))(normal_init_params, scale_params)

    reps = 4

    fixed_repeat_fn = lambda x: repeat_fn(x, rep = reps)
    init_params = tree_multimap(stack_fn, tree_map(fixed_repeat_fn, normal_init_params), tree_map(fixed_repeat_fn, scale_init_params))


    # In[24]:


    key = random.PRNGKey(int(time.time()*100))


    # In[25]:


    # lr = 1e-3
    # init_fn, update_fn, get_params = optimizers.nesterov(lr, 0.9)
    decay_steps = 400
    base_lr = 5e-4
    init_fn, update_fn, get_params = optimizers.adam(optimizers.exponential_decay(base_lr, decay_steps = decay_steps, decay_rate = 0.99))

    @jit
    def step(i, state, batch):
        g = grad(fit_model.loss)(get_params(state), batch)
        return update_fn(i, g, state)


    # In[26]:


    nIter = 120000


    # In[ ]:


    batch_size = 2048

    max_freqs = np.arange(1, 11) * 5

    loss_stor = []
    pvv_step = pmap(vmap(vmap(step, in_axes = [None, 0, None]), in_axes = (None, 0, 0)), in_axes = (None, 0, None))

    print(f'Start training : depth = {depth}, batch_size = {batch_size}, learning rate : {base_lr}, decay steps = {decay_steps}')
    for j, max_freq in enumerate(max_freqs):

        Xs = []
        Ys = []
        Xs_test = []
        Ys_test = []
        for _ in range(reps):


            key,_ = random.split(key)


            target_fn = target_fn_gen(max_freq, key = key)

            Y = target_fn(X_raw)
            Y = Y + random.normal(random.split(key)[0], shape = Y.shape)*Y.std(0)*0.01
            mu_Y, sigma_Y = Y.mean(0), Y.std(0)
            Y = (Y - mu_Y) / sigma_Y


            Y_test = target_fn(X_test_raw)

            Y_test = (Y_test - mu_Y) / sigma_Y

            Xs.append(X)
            Ys.append(Y)

            Xs_test.append(X_test)
            Ys_test.append(Y_test)

        Xs = np.array(Xs)
        Ys = np.array(Ys)

        Xs_test = np.array(Xs_test)
        Ys_test = np.array(Ys_test)





        opt_state = pmap(vmap(vmap(init_fn)))(init_params)
        pbar = trange(nIter)
        if j > 0:
            pbar.set_postfix({'Max Freq.': max_freq, 'Previous normal': loss_stor[-1][0][0,:,:].mean(), 'Previous scaled': loss_stor[-1][0][1,:,:].mean()})
        for i in pbar:
            key,_ = random.split(key)
            idx = random.choice(key, np.arange(X_raw.shape[0]), shape = (batch_size,), replace = False)
            opt_state = pvv_step(i, opt_state, (Xs[:, idx], Ys[:, idx]))

        opt_params = pmap(vmap(vmap(get_params)))(opt_state)

        freq_train_loss = pmap(vmap(vmap(fit_model.l2_error, in_axes = [0, None]), in_axes = [0, 0]), in_axes = (0, None))(opt_params, (Xs, Ys))
        freq_test_loss = pmap(vmap(vmap(fit_model.l2_error, in_axes = [0, None]), in_axes = [0, 0]), in_axes = (0, None))(opt_params, (Xs_test, Ys_test))


        loss_stor.append((freq_train_loss, freq_test_loss))
        np.save(path+f'{reps}_toy_loss_{batch_size}_{lam}_{base_lr}_{decay_steps}', np.array(loss_stor))
        np.save(path+f'{reps}_max_freq_{batch_size}_{lam}_{base_lr}_{decay_steps}', np.array(max_freqs))
    

    
        


