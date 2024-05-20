import jax
import jax.numpy as jnp
import numpy as np
import optax

import flax.linen as nn
from flax.jax_utils import replicate
from functools import partial
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from typing import Any
from tqdm import tqdm, trange

from checkpointer import Checkpointer
from diffusion import DDPM, Config
from datasets import generate_gmm
from models import MLP

# Initialisatoin of DDPM
T = 100
config = Config(T=T)
ddpm = DDPM(config)

# Parameters for the Gaussian Mixture Model (GMM)
prior = [0.5, 0.5]  # Priors for the two components
means = [-1, 1]     # Means for each component
stds = [0.5, 0.5]       # Standard deviations for each component
train_samples = 100_000    # Number of train samples to generate

# Generate samples from the GMM
x, _ = generate_gmm(prior, means, stds, train_samples)

# Standarise samples
scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

    
key = jax.random.PRNGKey(0)

t_n = jnp.linspace(start=0, stop=T, num=T)
score = MLP(hidden_dim=128, out_dim=1, n_layers=5)
params = score.init(key, jnp.concatenate([x[:T], jnp.expand_dims(t_n/T, axis=-1)], axis=1))
    
@partial(jax.jit, static_argnums=(3,))
def diffusion_loss(params, x, t, score, key):
    eps = jax.random.normal(key, shape=x.shape)
    x_t = ddpm.add_noise(x, t, key)
    # t is between 0 and T-1, both included. Therefore, t/(T-1) is in the interval [0,1]
    eps_pred = score.apply(params, jnp.concatenate([x_t, t/(T-1)], -1))
    # TODO: Add weights for different augmentation steps
    return jnp.mean((eps - eps_pred) ** 2)

@partial(jax.jit, static_argnums=(3,))
def weighted_diffusion_loss(params, x, t, score, key):
    eps = jax.random.normal(key, shape=x.shape)
    x_t = ddpm.add_noise(x, t, key)
    # t is between 0 and T-1, both included. Therefore, t/(T-1) is in the interval [0,1]
    eps_pred = score.apply(params, jnp.concatenate([x_t, t/(T-1)], -1))
    # TODO: Add weights for different augmentation steps
    return jnp.sum(((1.-ddpm.gamma_squared[t])/ddpm.gamma_squared[t])**2/(1.-ddpm.prod_gamma_squared_prev[t])*jnp.mean((eps - eps_pred) ** 2, axis=1))

opt = optax.adam(learning_rate=3e-4)
opt_state = opt.init(params)

n_steps = 5000
n_batch = 128
T = 100
best_loss = float('inf')

checkpointer = Checkpointer('./checkpoint.pkl')
# Checkpoint load reads parameters from disk.
# params= checkpointer.load()

with trange(n_steps) as steps:
    for step in steps:

        # Draw a random batches from x
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(key, x.shape[0], shape=(n_batch,))
        
        x_batch = x[idx]
        t_n_batch = jax.random.choice(key, T, shape=(x_batch.shape[0], 1))

        loss, grads = jax.value_and_grad(diffusion_loss)(params, x_batch, t_n_batch, score, key)
        updates, opt_state = opt.update(grads, opt_state, params)

        if loss < best_loss:
            best_loss = loss
            checkpointer.save(params)

        params = optax.apply_updates(params, updates)

        steps.set_postfix(val=loss)

checkpointer.save(params)
# Checkpoint load reads parameters from disk.
#params= checkpointer.load()

n_samples = 100_000
x_t = jax.random.normal(key, shape=(n_samples, 1))

samples_per_step = [x_t]

for t in tqdm(range(T-1, -1, -1)):
   t_array = jnp.repeat(jnp.atleast_1d(t), n_samples, axis=0)
   eps_pred = score.apply(params, jnp.concatenate([x_t, jnp.expand_dims(t_array/(T-1), axis=-1)], axis=-1))
   key, subkey = jax.random.split(key)
   x_t = ddpm.sample_previous_eps_pred(x_t, eps_pred, t, key)
   # Add to the samples-per-step
   samples_per_step.append(x_t)

def gmm_density(x, prior, means, stds):
  # Iterate over GMM components
  density = prior[0]*norm.pdf(x, loc=means[0], scale=stds[0])
  for i in range(1, len(prior)):
    density += prior[i]*norm.pdf(x, loc=means[i], scale=stds[i])
  return density

def log_gmm_density(x, prior, means, stds):
  # Iterate over GMM components
  log_density = jnp.logaddexp([jnp.log(prior[i])+norm.logpdf(x, loc=means[i], scale=stds[i]) for i in range(len(prior))])
  return log_density


print(ddpm.log_w(samples_per_step, score, params, lambda x : gmm_density(x, prior, means, stds)))


# To reverse the transformation
x_t = scaler.inverse_transform(x_t)

# Generate a range of x values
x_values = np.linspace(-4, 4, 1000)


# Calculate the density for each x value
density = gmm_density(x_values, prior, means, stds)

plt.hist(np.array(x_t), density=True, bins=50)
plt.plot(x_values, density, label='Density Function', color='blue')
plt.show()