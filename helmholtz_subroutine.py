from helpers import *


# In[7]:


def laplacian(f, params, s):
    x, y = s
    f_aux = lambda x, y: f(params, np.array([x, y]))[0]
    f_xx = grad(grad(f_aux, argnums = 0), argnums = 0)(x, y)
    f_yy = grad(grad(f_aux, argnums = 1), argnums = 1)(x, y)
    
    return f_xx + f_yy

def helmholtz(f, k, q_fn):
    def pde(params, x):
        return laplacian(f, params, x) + k**2 * f(params, x) - q_fn(x)
    return pde


# In[8]:


class boundary_sampler:
    def __init__(self, batch_size, f):
        self.batch_size = batch_size // 4
        self.f = f
    @partial(jit, static_argnums=(0,))
    def sample(self, key):
        xs = random.uniform(key, (4, self.batch_size), maxval = 1., minval = -1.)
        y = np.array((np.ones((self.batch_size,)), xs[0])).T
        y = np.concatenate((y, np.array((-1. * np.ones((self.batch_size,)), xs[1])).T))
        y = np.concatenate((y, np.array((xs[2], -1. * np.ones((self.batch_size,)))).T))
        y = np.concatenate((y, np.array((xs[3], np.ones((self.batch_size,)))).T))
        return (y, self.f(y.T).reshape(y.shape[0], -1))
class residual_sampler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

#     @partial(jit, static_argnums=(0,))
    def sample(self, key):
        x = arcsine.rvs(size = (self.batch_size,2)) * 2 - 1

        return np.array(x)
        


# In[9]:


key = random.PRNGKey(int(time.time()*100))


# In[10]:

rcs_sampler = residual_sampler(batch_size)

mu_X = rcs_sampler.sample(key).mean(0)
sigma_X= rcs_sampler.sample(key).std(0)


# In[11]:



activation = np.tanh
init_model = normInitNet(layers[:-1], activation, mu_X, sigma_X)


# In[12]:


lr = 5e-3
init_fn, update_fn, get_params = optimizers.adam(lr)
# init_fn, update_fn, get_params = optimizers.nesterov(lr, 0.9)
lam = 1e-3
@jit
def step(i, state, X):
    key = random.PRNGKey(i)
    g = grad(init_model.regulated_logSineLoss)(get_params(state), init_model.net_init(key), X, lam = lam)
    return update_fn(i, g, state)
print(f'Important: confirm lam {lam} before running!!!!!!!!!!!!!!!')


# In[13]:


opt_state = init_fn(init_model.scale_params)
init_nIter = 20000
pbar = trange(init_nIter)
scale_stor = []
bias_stor = []

for i in pbar:
    mini_batch = rcs_sampler.sample(random.PRNGKey(i))
    opt_state = step(i, opt_state, mini_batch)




scale_params = get_params(opt_state)



# In[17]:


u_fn = lambda x : np.sin(a1 * np.pi * x[0]) * np.sin(a2 * np.pi * x[1])
q_fn = lambda x:  (-np.pi**2*(a1 ** 2 + a2 ** 2) + k**2) * np.sin(a1 * np.pi * x[0]) * np.sin(a2 * np.pi * x[1])

bcs_sampler = boundary_sampler(batch_size, u_fn)
mu_Y, sigma_Y = bcs_sampler.sample(key)[1].mean(), bcs_sampler.sample(key)[1].std()

X_test = np.array(np.meshgrid(np.linspace(-1,1, 100), np.linspace(-1,1, 100))).transpose(1,2,0).reshape(-1, 2)
Y_test = u_fn(X_test.T)

fit_model = fitPinns(layers, activation, mu_X, sigma_X, helmholtz, (k, q_fn))

# In[18]:


fit_model.scaled_net_params = parameter_scaling(fit_model.net_params[:-1], scale_params) + [fit_model.net_params[-1]]


key = random.PRNGKey(int(time.time()*100))


# In[21]:


# lr = 1e-3
# init_fn, update_fn, get_params = optimizers.nesterov(lr, 0.9)

init_fn, update_fn, get_params = optimizers.adam(optimizers.exponential_decay(1e-4, decay_steps = 1000, decay_rate = 0.99))

@jit
def step(i, state, batch, ws):
    g1 = grad(fit_model.boundary_loss)(get_params(state), batch[0])
    g2 = grad(fit_model.residual_loss)(get_params(state), batch[1])
    g = tree_multimap(lambda x,y: ws[0] * x + ws[1] * y, g1, g2)
    return update_fn(i, g, state)


# In[22]:



# In[ ]:


pbar = trange(nIter)
log = 500
# stack the parameter trees at each leaf node for vmap
init_paramses = tree_multimap(stack_fn, fit_model.net_params, fit_model.scaled_net_params)
opt_state = vmap(init_fn)(init_paramses)
# compile the vmap optimization step
v_step = jit(vmap(step, in_axes = (None, 0, None, None)))
loss_stor = []
losses_stor = []
weight = [50., 1.]
for i in pbar:
    key,_ = random.split(key)
    mini_batch = (bcs_sampler.sample(key), rcs_sampler.sample(key))
    opt_state = v_step(i, opt_state, mini_batch, weight)
    if i % log == 0:
        params = vmap(get_params)(opt_state)
        loss_val = vmap(fit_model.loss, in_axes = [0, None, None])(params, mini_batch, weight)
        
        loss_stor.append(loss_val)
        
        bcs_loss = vmap(fit_model.boundary_loss, in_axes = [0, None])(params, mini_batch[0])
        rcs_loss = vmap(fit_model.residual_loss, in_axes = [0, None])(params, mini_batch[1])
        losses_stor.append((bcs_loss, rcs_loss))
        
        pbar.set_postfix({'boundary loss': f'{bcs_loss[0]:.2e} {bcs_loss[1]:.2e}',                         'residual loss': f'{rcs_loss[0]:.2e} {rcs_loss[1]:.2e}'})
        


# In[ ]:


loss_stor = np.array(loss_stor)
losses_stor = np.array(losses_stor)
opt_params = vmap(get_params)(opt_state)

# Save the individual losses
np.save(path + f'loss_{k}_{a1}_{a2}.npy', loss_stor)

np.save(path + f'losses_{k}_{a1}_{a2}.npy', losses_stor)
np.save(path + f'flat_paramses_{k}_{a1}_{a2}.npy', ravel_pytree(opt_params)[0])
normal_opt_params = tree_map(lambda x: x[0], opt_params)
scaled_opt_params = tree_map(lambda x: x[1], opt_params)


# In[ ]:


plt.figure(figsize = (4,3))
plt.plot(layer_weight_norm(normal_opt_params), label = 'Xavier, trained')
plt.plot(layer_weight_norm(scaled_opt_params), label = 'Scaled, trained')
plt.plot(layer_weight_norm(fit_model.net_params), linewidth = 4, alpha = 0.5, linestyle = '--', label = 'Xavier init.')
plt.plot(layer_weight_norm(fit_model.scaled_net_params), linewidth = 4, alpha = 0.5, linestyle = '--', label = 'Scaled init.')
plt.legend()
plt.xlabel('layer')
plt.ylabel(r'Frobenius norm')
plt.ylim([0, 90])
plt.tight_layout()
plt.savefig(path + f'norm_{k}_{a1}_{a2}_depth_{d}.pdf', dpi = 200)
plt.close()


# In[ ]:


plt.figure(figsize = (15,3))
plt.subplot(1,4,1)
c = plt.tricontourf(X_test[:,0], X_test[:,1], Y_test)
plt.colorbar(c)
plt.title('Ground Truth')
plt.tight_layout()

plt.subplot(1,4,2)
c = plt.tricontourf(X_test[:,0], X_test[:,1], fit_model.net_apply(scaled_opt_params, X_test).flatten())
plt.colorbar(c)
plt.title('Prediction')
plt.tight_layout()

plt.subplot(1,4,3)
c = plt.tricontourf(X_test[:,0], X_test[:,1], Y_test - fit_model.net_apply(scaled_opt_params, X_test).flatten(), cmap = 'jet')
plt.colorbar(c)
# plt.title(r'Test Rel. $L_2$ : ' + f'{fit_model.l2_error(scaled_opt_params, (X_test, Y_test)):.1e}')
plt.title('Error')

plt.subplot(1,4,4)
c = plt.tricontourf(X_test[:,0], X_test[:,1], vmap(fit_model.residual, in_axes = [None, 0])(scaled_opt_params, X_test).flatten(), cmap = 'jet')
plt.colorbar(c)
# plt.title(r'Test Rel. $L_2$ : ' + f'{fit_model.l2_error(scaled_opt_params, (X_test, Y_test)):.1e}')
plt.title('Residual')

plt.tight_layout()

plt.savefig(path + f'scaled_{k}_{a1}_{a2}_depth_{d}.pdf', dpi = 200)

plt.close()


# In[ ]:


plt.figure(figsize = (15,3))
plt.subplot(1,4,1)
c = plt.tricontourf(X_test[:,0], X_test[:,1], Y_test)
plt.colorbar(c)
plt.title('Ground Truth')
plt.tight_layout()

plt.subplot(1,4,2)
c = plt.tricontourf(X_test[:,0], X_test[:,1], fit_model.net_apply(normal_opt_params, X_test).flatten())
plt.colorbar(c)
plt.title('Prediction')
plt.tight_layout()

plt.subplot(1,4,3)
c = plt.tricontourf(X_test[:,0], X_test[:,1], Y_test - fit_model.net_apply(normal_opt_params, X_test).flatten(), cmap = 'jet')
plt.colorbar(c)
# plt.title(r'Test Rel. $L_2$ : ' + f'{fit_model.l2_error(scaled_opt_params, (X_test, Y_test)):.1e}')
plt.title('Error')

plt.subplot(1,4,4)
c = plt.tricontourf(X_test[:,0], X_test[:,1], vmap(fit_model.residual, in_axes = [None, 0])(normal_opt_params, X_test).flatten(), cmap = 'jet')
plt.colorbar(c)
# plt.title(r'Test Rel. $L_2$ : ' + f'{fit_model.l2_error(scaled_opt_params, (X_test, Y_test)):.1e}')
plt.title('Residual')

plt.tight_layout()

plt.savefig(path + f'normal_{k}_{a1}_{a2}_depth_{d}.pdf', dpi = 200)

plt.close()


# In[ ]:


plt.figure(figsize = (4,3))
plt.plot(log * np.arange(loss_stor.shape[0]), loss_stor[:,1], label = 'Scaled initialization')
plt.plot(log * np.arange(loss_stor.shape[0]), loss_stor[:,0], label = 'Xavier initialization')
plt.yscale('log')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(path + f'loss_{k}_{a1}_{a2}_depth_{d}.pdf', dpi = 200)
plt.close()


# In[ ]:


plt.figure(figsize = (8,3))
plt.subplot(1,2,1)
plt.plot(log * np.arange(loss_stor.shape[0]), losses_stor[:,0,1], label = 'Boundary, scaled')
plt.plot(log * np.arange(loss_stor.shape[0]), losses_stor[:,0,0], label = 'Boundary, Xavier')
plt.yscale('log')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(log * np.arange(loss_stor.shape[0]), losses_stor[:,1,1], label = 'Residual, scaled')
plt.plot(log * np.arange(loss_stor.shape[0]), losses_stor[:,1,0], label = 'Residual, Xavier')
plt.yscale('log')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(path + f'losses_{k}_{a1}_{a2}_depth_{d}.pdf', dpi = 200)
plt.close()


# In[ ]:

normal_test_error = fit_model.l2_error(normal_opt_params, (X_test, Y_test))
print('normal test error: ' + str(normal_test_error))
scaled_test_error = fit_model.l2_error(scaled_opt_params, (X_test, Y_test))
print('scaled test error: ' + str(scaled_test_error))

