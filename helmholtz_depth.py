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

import matplotlib
matplotlib.use('Agg')


from scipy.interpolate import griddata
from scipy.stats import arcsine

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

# In[4]:



test_errors = []
path = f'helmholtz/{int(width)}/'
print(f'depths: {ds}')

k = 1
a1 = 12
a2 = 12

print('k = ', k, 'a1 = ', a1, 'a2 = ', a2)

batch_size = 512

nIter = 400000

for d in ds:

    layers = [2, *[width for _ in range(d)], 1]
    print('==============================================')
    print(f'depth is: {d}, width = {layers[1]}')
    print('==============================================')
    exec(open("helmholtz_subroutine.py").read())
    test_errors.append((normal_test_error, scaled_test_error))
    np.save(path + f'helmholtz_test_errors_{k}_{a1}_{a2}_{layers[1]}_{ds}.npy', np.array(test_errors))

