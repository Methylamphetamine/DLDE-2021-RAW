#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse

parser = argparse.ArgumentParser(description='Sampling ratio')


parser.add_argument('-r', type=float, dest = 'ratio')

ratio = vars(parser.parse_args())['ratio']




# In[4]:


import numpy as onp
from numpy import fft
import jax.numpy as np
from jax import random, grad, vmap, jit, lax
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
from matplotlib import image
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



# In[5]:


source_path = ''
save_path = 'regularized/color/workshop_final/'


# In[6]:


plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16,
                        'lines.linewidth': 2,
                        'axes.labelsize': 10,
                        'axes.titlesize': 16,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'legend.fontsize': 10,
                        'axes.linewidth': 2})


# In[7]:


exec(open("../helpers.py").read())
exec(open("../regularized_image_regression_helpers.py").read())



def d1_regularization(f):
    def pde(params, x):
        f_r = lambda params, x: f(params, x)[0]
        f_g = lambda params, x: f(params, x)[1]
        f_b = lambda params, x: f(params, x)[2]
        return np.array([grad(f_r, argnums = 1)(params, x), grad(f_g, argnums = 1)(params, x),                         grad(f_b, argnums = 1)(params, x)])
    return pde


# In[8]:


key = random.PRNGKey(1234)


# In[9]:


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], np.array([0.2989, 0.5870, 0.1140])).reshape(rgb.shape[:2] + (1,))
file_name = 'ucd.jpg'
img = image.imread(source_path + file_name)
# Y_raw = np.array(rgb2gray(img))
Y_raw = np.array(img)
Y_raw = Y_raw[::3][:,::3]
Y_raw = Y_raw[:min(Y_raw.shape[:2])][:,:min(Y_raw.shape[:2])]
Y_raw.shape


# In[10]:


X_raw = np.array(np.meshgrid(np.arange(Y_raw.shape[0]), np.arange(Y_raw.shape[1]))).reshape(2,-1).T


# In[11]:


# ratio = 0.99
train_idx = random.choice(key, np.arange(X_raw.shape[0]), shape = (int(X_raw.shape[0] * ratio), ), replace = False)

X_train = X_raw[train_idx]
mu_X, sigma_X = X_train.mean(0), X_train.std(0)

Y_train = Y_raw.reshape(-1,Y_raw.shape[-1])[train_idx]
mu_Y, sigma_Y = Y_train.mean(0), Y_train.std(0)
Y_train = (Y_train - mu_Y) / sigma_Y

print(f'Ratio of pixel used: {ratio}')


# In[12]:


mask = np.zeros(Y_raw.shape).reshape(-1,Y_raw.shape[-1])
mask = index_update(mask, train_idx, 1).reshape(Y_raw.shape)


# In[13]:


batch_size = 512
# batch_size = 2048
rcs_sampler = residual_sampler(batch_size, Y_raw.shape[0])


# In[14]:


# For colored
width = 512
depth = 8

layers = [2, *[width for _ in range(depth)], Y_raw.shape[-1]]

# Pure sine function, not scaled sine
activation = np.sin
save_path = save_path + 'SIREN_'

dist = lambda key, shape: random.uniform(key, shape, minval = -np.sqrt(6), maxval = np.sqrt(6))
init_model = normInitNet(layers[:-1], activation, mu_X, sigma_X, dist = dist)


# In[15]:


lr = 5e-3
init_fn, update_fn, get_params = optimizers.adam(lr)
# init_fn, update_fn, get_params = optimizers.nesterov(lr, 0.9)
lam = 1e-4
@jit
def step(i, state, X):
    key = random.PRNGKey(i)
    g = grad(init_model.regulated_logSineLoss)(get_params(state), init_model.net_init(key), X, lam = lam)
    return update_fn(i, g, state)
print(f'Important: confirm lam {lam} before running!!!!!!!!!!!!!!!')


# In[16]:


opt_state = init_fn(init_model.scale_params)
init_nIter = 20000
pbar = trange(init_nIter)
scale_stor = []
bias_stor = []

for i in pbar:
    mini_batch = rcs_sampler.sample(random.PRNGKey(i))
    opt_state = step(i, opt_state, mini_batch)
    
    if i % 100 == 0:
        params = get_params(opt_state)
        pbar.set_postfix({'Log sine loss': init_model.logSineLoss(get_params(opt_state), init_model.net_init(random.PRNGKey(i)), mini_batch)})
        scale_stor.append([w for (w, b) in params])
        bias_stor.append([b for (w, b) in params])
        

scale_stor = np.array(scale_stor)
bias_stor = np.array(bias_stor)
        


# In[18]:


scale_params = params


# In[20]:


bcs_sampler = boundary_sampler(batch_size, X_train, Y_train)
fit_model = fitPinns(layers, activation, mu_X, sigma_X, d1_regularization, (), dist = dist)


# In[24]:


fit_model.scaled_net_params = parameter_scaling(fit_model.net_params[:-1], scale_params) + [fit_model.net_params[-1]]
original_scale_parameters = [[1., 0.] for _ in scale_params]
original_scale_parameters[0][0] = 30.
fit_model.orginal_scaled_net_params = parameter_scaling(fit_model.net_params[:-1], original_scale_parameters) + [fit_model.net_params[-1]]


# In[23]:


key = random.PRNGKey(1234)


# In[24]:


# lr = 1e-3
# init_fn, update_fn, get_params = optimizers.nesterov(lr, 0.9)

init_fn, update_fn, get_params = optimizers.adam(optimizers.exponential_decay(1e-4, decay_steps = 1000, decay_rate = 0.99))


# In[25]:


nIter = 400000
losses = []
regularizer = 1e-2
more_regularizer = 1.


# In[26]:


@jit
def step(i, state, batch):
    g1 = grad(fit_model.boundary_loss)(get_params(state), batch[0])

    return update_fn(i, g1, state)

pbar = trange(nIter)
log = 1000
# stack the parameter trees at each leaf node for vmap
init_paramses = tree_multimap(stack_fn, fit_model.net_params,
                              fit_model.scaled_net_params,
                              fit_model.orginal_scaled_net_params)
opt_state = vmap(init_fn)(init_paramses)
# compile the vmap optimization step
v_step = jit(vmap(step, in_axes = (None, 0, None)))
loss_stor = []

rcs_place_holder = rcs_sampler.sample(key)
for i in pbar:
    key,_ = random.split(key)
    mini_batch = (bcs_sampler.sample(key), rcs_place_holder)
    opt_state = v_step(i, opt_state, mini_batch)
    if i % log == 0:
        params = vmap(get_params)(opt_state)

        bcs_loss = vmap(fit_model.boundary_loss, in_axes = [0, None])(params, mini_batch[0])
        rcs_loss = vmap(fit_model.residual_loss, in_axes = [0, None])(params, mini_batch[1])
        loss_stor.append((bcs_loss, rcs_loss))
        
        pbar.set_postfix({'boundary loss': f'{bcs_loss[0]:.2e} {bcs_loss[1]:.2e} {bcs_loss[2]:.2e}',                         'residual loss': f'{rcs_loss[0]:.2e} {rcs_loss[1]:.2e}, {rcs_loss[2]:.2e}'})
losses.append(loss_stor) 
        

opt_params = vmap(get_params)(opt_state)
normal_opt_params = tree_map(lambda x: x[0], opt_params)
scaled_opt_params = tree_map(lambda x: x[1], opt_params)
original_opt_params = tree_map(lambda x: x[2], opt_params)

normal_pred_img = (fit_model.net_apply(normal_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
scaled_pred_img = (fit_model.net_apply(scaled_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
original_pred_img = (fit_model.net_apply(original_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)


# In[31]:


@jit
def step(i, state, batch, w):
    g1 = grad(fit_model.boundary_loss)(get_params(state), batch[0])
    g2 = grad(fit_model.residual_loss)(get_params(state), batch[1])
    g = tree_multimap(lambda x,y: w[0] * x + y * w[1], g1, g2)
    return update_fn(i, g, state)

pbar = trange(nIter)
# stack the parameter trees at each leaf node for vmap
init_paramses = tree_multimap(stack_fn, fit_model.net_params,
                              fit_model.scaled_net_params,
                              fit_model.orginal_scaled_net_params)
opt_state = vmap(init_fn)(init_paramses)
# compile the vmap optimization step
v_step = jit(vmap(step, in_axes = (None, 0, None, None)))
loss_stor = []
weight = [1., regularizer]
for i in pbar:
    key,_ = random.split(key)
    mini_batch = (bcs_sampler.sample(key), rcs_sampler.sample(key))
    opt_state = v_step(i, opt_state, mini_batch, weight)
    if i % log == 0:
        params = vmap(get_params)(opt_state)

        bcs_loss = vmap(fit_model.boundary_loss, in_axes = [0, None])(params, mini_batch[0])
        rcs_loss = vmap(fit_model.residual_loss, in_axes = [0, None])(params, mini_batch[1])
        loss_stor.append((bcs_loss, rcs_loss))
        
        pbar.set_postfix({'boundary loss': f'{bcs_loss[0]:.2e} {bcs_loss[1]:.2e} {bcs_loss[2]:.2e}',                         'residual loss': f'{rcs_loss[0]:.2e} {rcs_loss[1]:.2e}, {rcs_loss[2]:.2e}'})
losses.append(loss_stor) 
        

opt_params = vmap(get_params)(opt_state)
regularized_normal_opt_params = tree_map(lambda x: x[0], opt_params)
regularized_scaled_opt_params = tree_map(lambda x: x[1], opt_params)
regularized_original_opt_params = tree_map(lambda x: x[2], opt_params)

regularized_normal_pred_img = (fit_model.net_apply(regularized_normal_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
regularized_scaled_pred_img = (fit_model.net_apply(regularized_scaled_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
regularized_original_pred_img = (fit_model.net_apply(regularized_original_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)


# In[36]:


@jit
def step(i, state, batch, w):
    g1 = grad(fit_model.boundary_loss)(get_params(state), batch[0])
    g2 = grad(fit_model.residual_loss)(get_params(state), batch[1])
    g = tree_multimap(lambda x,y: w[0] * x + y * w[1], g1, g2)
    return update_fn(i, g, state)

pbar = trange(nIter)
# stack the parameter trees at each leaf node for vmap
init_paramses = tree_multimap(stack_fn, fit_model.net_params,
                              fit_model.scaled_net_params,
                              fit_model.orginal_scaled_net_params)
opt_state = vmap(init_fn)(init_paramses)
# compile the vmap optimization step
v_step = jit(vmap(step, in_axes = (None, 0, None, None)))
loss_stor = []
weight = [1., more_regularizer]
for i in pbar:
    key,_ = random.split(key)
    mini_batch = (bcs_sampler.sample(key), rcs_sampler.sample(key))
    opt_state = v_step(i, opt_state, mini_batch, weight)
    if i % log == 0:
        params = vmap(get_params)(opt_state)

        bcs_loss = vmap(fit_model.boundary_loss, in_axes = [0, None])(params, mini_batch[0])
        rcs_loss = vmap(fit_model.residual_loss, in_axes = [0, None])(params, mini_batch[1])
        loss_stor.append((bcs_loss, rcs_loss))
        
        pbar.set_postfix({'boundary loss': f'{bcs_loss[0]:.2e} {bcs_loss[1]:.2e} {bcs_loss[2]:.2e}',                         'residual loss': f'{rcs_loss[0]:.2e} {rcs_loss[1]:.2e}, {rcs_loss[2]:.2e}'})
losses.append(loss_stor) 
        

opt_params = vmap(get_params)(opt_state)
more_regularized_normal_opt_params = tree_map(lambda x: x[0], opt_params)
more_regularized_scaled_opt_params = tree_map(lambda x: x[1], opt_params)
more_regularized_original_opt_params = tree_map(lambda x: x[2], opt_params)

more_regularized_normal_pred_img = (fit_model.net_apply(more_regularized_normal_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
more_regularized_scaled_pred_img = (fit_model.net_apply(more_regularized_scaled_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
more_regularized_original_pred_img = (fit_model.net_apply(more_regularized_original_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)


# In[ ]:


losses = np.array(losses)
np.save(save_path + file_name[:-4] + f'_width_{width}_depth_{depth}_multi_regularized_{batch_size}_{ratio}_losses',        losses)
try:
    np.save(save_path + file_name[:-4] + f'_width_{width}_depth_{depth}_multi_regularized_{batch_size}_{ratio}_normalization', np.array((mu_X, sigma_X)))
except:
    print(np.array((mu_X, sigma_X)))
# In[43]:


np.save(save_path + file_name[:-4] +         f'_width_{width}_depth_{depth}_multi_regularized_{batch_size}_{ratio}_flat_params',        np.array((ravel_pytree(normal_opt_params)[0],
                  ravel_pytree(scaled_opt_params)[0],
                  ravel_pytree(original_opt_params)[0],
                  ravel_pytree(regularized_normal_opt_params)[0],
                  ravel_pytree(regularized_scaled_opt_params)[0],
                  ravel_pytree(regularized_original_opt_params)[0],
                  ravel_pytree(more_regularized_normal_opt_params)[0],
                  ravel_pytree(more_regularized_scaled_opt_params)[0],
                  ravel_pytree(more_regularized_original_opt_params)[0]))
                 )


# In[44]:



flat_params = np.load(save_path + file_name[:-4] +         f'_width_{width}_depth_{depth}_multi_regularized_{batch_size}_{ratio}_flat_params.npy')
_, unravel = ravel_pytree(fit_model.scaled_net_params)

normal_opt_params = unravel(flat_params[0])
scaled_opt_params = unravel(flat_params[1])
original_opt_params = unravel(flat_params[2])
regularized_normal_opt_params = unravel(flat_params[3])
regularized_scaled_opt_params = unravel(flat_params[4])
regularized_original_opt_params = unravel(flat_params[5])
more_regularized_normal_opt_params = unravel(flat_params[6])
more_regularized_scaled_opt_params = unravel(flat_params[7])
more_regularized_original_opt_params = unravel(flat_params[8])

normal_pred_img = (fit_model.net_apply(normal_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
scaled_pred_img = (fit_model.net_apply(scaled_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
original_pred_img = (fit_model.net_apply(original_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)

regularized_normal_pred_img = (fit_model.net_apply(regularized_normal_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
regularized_scaled_pred_img = (fit_model.net_apply(regularized_scaled_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
regularized_original_pred_img = (fit_model.net_apply(regularized_original_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)

more_regularized_normal_pred_img = (fit_model.net_apply(more_regularized_normal_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
more_regularized_scaled_pred_img = (fit_model.net_apply(more_regularized_scaled_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)
more_regularized_original_pred_img = (fit_model.net_apply(more_regularized_original_opt_params, X_raw) * sigma_Y + mu_Y).reshape(Y_raw.shape)


# In[45]:


SNR = lambda pred, y: -np.log(np.linalg.norm(pred - y) / np.linalg.norm(y)) / np.log(10) * 10 * 2

print(f'SNR of Xavier initialization: {SNR(normal_pred_img, Y_raw)}, {SNR(regularized_normal_pred_img, Y_raw)}, {SNR(more_regularized_normal_pred_img, Y_raw)}')
print(f'SNR of scaled initialization: {SNR(scaled_pred_img, Y_raw)}, {SNR(regularized_scaled_pred_img, Y_raw)}, {SNR(more_regularized_scaled_pred_img, Y_raw)}')
print(f'SNR of original initialization: {SNR(original_pred_img, Y_raw)}, {SNR(regularized_original_pred_img, Y_raw)}, {SNR(more_regularized_original_pred_img, Y_raw)}')


# In[46]:


with open(save_path + file_name[:-4] +           f'_width_{width}_depth_{depth}_multi_regularized_{batch_size}_{ratio}_SNR.txt', "w") as text_file:
    text_file.write(f'regularization: 0, {regularizer}, {more_regularizer}\n')
    text_file.write(f'SNR of Xavier initialization: {SNR(normal_pred_img, Y_raw)}, {SNR(regularized_normal_pred_img, Y_raw)}, {SNR(more_regularized_normal_pred_img, Y_raw)}\n')
    text_file.write(f'SNR of scaled initialization: {SNR(scaled_pred_img, Y_raw)}, {SNR(regularized_scaled_pred_img, Y_raw)}, {SNR(more_regularized_scaled_pred_img, Y_raw)}\n')
    text_file.write(f'SNR of original initialization: {SNR(original_pred_img, Y_raw)}, {SNR(regularized_original_pred_img, Y_raw)}, {SNR(more_regularized_original_pred_img, Y_raw)}\n')

