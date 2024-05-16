from functools import partial

import jax
import jax.numpy as jnp
import flax
import numpy as np
import flax.linen as nn
from flax.jax_utils import replicate
import optax
import orbax
import orbax.checkpoint
from sklearn import datasets, preprocessing
from jax.scipy.stats import norm
from typing import Any
from tqdm import tqdm

import matplotlib.pyplot as plt
from tqdm import trange

import scipy.integrate as integrate
import scipy.special as special

import pickle

class Checkpointer:
  def __init__(self, path):
    self.path = path

  def save(self, params):
    params = jax.device_get(params)
    with open(self.path, 'wb') as fp:
      pickle.dump(params, fp)

  def load(self):
    with open(self.path, 'rb') as fp:
      params = pickle.load(fp)
    return jax.device_put(params)

n_samples = 100_0000

global T
T = 16

x, _ = datasets.make_moons(n_samples=n_samples, noise=.06, random_state=0)

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

#plt.hist2d(x[:, 0], x[:, 1], bins=100)
#plt.xlim(-2 ,2)
#plt.ylim(-2, 2)
#plt.show()

# Parameters for the Gaussian Mixture Model (GMM)
prior = [0.5, 0.5]  # Priors for the two components
means = [-1, 1]     # Means for each component
stds = [0.5, 0.5]       # Standard deviations for each component
train_samples = n_samples    # Number of train samples to generate

np.random.seed(0)

def generate_samples(prior, means, stds, n_samples):
  # Generate the samples from the GMM
  # Determine component choices based on priors
  elements_per_component = np.random.multinomial(n_samples, prior, size=1,)[0]

  samples = [
      means[i]+stds[i]*jax.random.normal(jax.random.PRNGKey(0), shape=(num_elements,))
      for i, num_elements in enumerate(elements_per_component)
  ]
  print(samples)
  labels = tuple([
      [i]*num_elements
      for i, num_elements in enumerate(elements_per_component)
  ])
  # Return samples
  return jnp.array(np.concatenate(samples)), jnp.array(np.concatenate(labels))

def gmm_density(x, prior, means, stds):
  # Iterate over GMM components
  density = prior[0]*norm.pdf(x, loc=means[0], scale=stds[0])
  for i in range(1, len(prior)):
    density += prior[i]*norm.pdf(x, loc=means[i], scale=stds[i])
  return density

def unnormalised_gmm_density(x, prior, means, stds):
    # Supposing stds are the same for all
    return gmm_density(x, prior, means, stds) * jnp.sqrt(jnp.pi)

def log_gmm_density(x, prior, means, stds):
    # Iterate over GMM components
    log_components = jnp.array([
        jnp.log(prior[i]) 
        + norm.logpdf(x, loc=means[i], scale=stds[i]) for i in range(len(prior))
    ])
    return jax.nn.logsumexp(log_components)

# Shuffle arrays
x, _ = generate_samples(prior, means, stds, n_samples)
x = jnp.expand_dims(jax.random.permutation(jax.random.PRNGKey(0), x), axis=-1)

# Shuffle arrays
stds_q = [1.0, 1.0]
x_q, _ = generate_samples(prior, means, stds_q, n_samples)
x_q = jnp.expand_dims(jax.random.permutation(jax.random.PRNGKey(0), x_q), axis=-1)

print(f"Z={jnp.mean(gmm_density(x_q, prior, means, stds)/gmm_density(x_q, prior, means, stds_q))}")

result = integrate.quad(lambda x: gmm_density(x, prior, means, stds), -np.inf, np.inf)
print(result)
print(jnp.mean(x),jnp.var(x))
print(f"Z={jnp.mean(gmm_density(x, prior, means, stds))}")

# Generate a range of x values
x_values = np.linspace(-4, 4, 1000)

# Calculate the density for each x value
density = gmm_density(x_values, prior, means, stds)
unnormalised_density = unnormalised_gmm_density(x_values, prior, means, stds)

"""
plt.hist(np.array(x), density=True, bins=30)
plt.plot(x_values, density, label='Normalised Density Function', color='blue')
plt.plot(x_values, unnormalised_density, label='Unnormalised Density Function', color='red')
plt.legend()
plt.show()
"""

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

# Following (more or less) Rich's notation
# q(x_{t}|x_{t-1}) = N(x_{t}; gamma_{t}*x_{t-1}; sigma^{2}_{t}I)
# For a variance-preserving AR(1), gamma_{t}=sqrt(1-sigma^{2}_{t})
# Under this setting, it is verified that
# q(x_{t}|x_{0})=N(x_{t}; sqrt(alpha_bar_{t}x_{0}), (1-alpha_bar_{t})I)=
# where alpha_bar_{t}=prod(sqrt(1-sigma^{2}_{t}))

# Let's suppose that the number of timesteps is discrete

def variance_schedule(t, t_min=1e-4, t_max=0.02):
    """ 
    Linear variance schedule
    """
    return t_min + (t_max - t_min) * t

#print(f'variance_schedule(1)={variance_schedule(1)}')
#print(f'variance_schedule(2)={variance_schedule(2)}')
#print(f'variance_schedule(10)={variance_schedule(10)}')

def sigma_squared_t(t, schedule=variance_schedule):
    """ 
    Linear variance schedule
    """
    return schedule(t)

def gamma_t(t):
    """ 
    \gamma_{t} = \sqrt{1-sigma^{2}_{t}} to ensure that we have a AR(1) variance-preserving process
    """
    return jnp.sqrt(gamma_squared_t(t))

def gamma_squared_t(t):
    """ 
    \gamma_{t} = \sqrt{1-sigma^{2}_{t}} to ensure that we have a AR(1) variance-preserving process
    """
    return 1-sigma_squared_t(t)

def weight_t(t):
    return 1.

#print(f'gamma_t(1)={gamma_t(1)}')
#print(f'gamma_t(2)={gamma_t(2)}')
#print(f'gamma_t(10)={gamma_t(10)}')

def prod_gamma_t(t):
    # t denotes the current time, and T the number of timesteps
    return jnp.sqrt(prod_gamma_squared_t(t))

def prod_gamma_squared_t(t):
    # t denotes the current time, and T the number of timesteps
    gamma_t_ary = jax.vmap(lambda t: 1 - sigma_squared_t(t))(jnp.arange(T)/T)
    return jnp.cumprod(gamma_t_ary)[t]

def compute_x_t_from_x_0(x_0, eps, t):
    x_t = prod_gamma_t(t)*x_0 + jnp.sqrt(1-prod_gamma_squared_t(t))*eps
    return x_t

#print(f'variance_schedule(1)={variance_schedule(1)}')
#print(f'gamma_t(1)={gamma_t(1)}')
#print(f'prod_gamma_t(1)={prod_gamma_t(1)}')
#print(f'1-prod_gamma_squared_t(1)={1-prod_gamma_squared_t(1)}')

class MLP(nn.Module):
    """ A simple MLP in Flax. This is the noise-prediction or score function.
    """
    hidden_dim: int = 32
    out_dim: int = 1
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x
    
@partial(jax.jit, static_argnums=(3,))
def loss_fn(params, x, t, score, key):
    eps = jax.random.normal(key, shape=x.shape)
    #jax.debug.print("{eps}", eps=eps)
    x_t = compute_x_t_from_x_0(x, eps, t)
    eps_pred = score.apply(params, jnp.concatenate([x_t, t/T], -1))    
    #print(jnp.concatenate([x_t, t/T], -1))
    return weight_t(t) * jnp.mean((eps - eps_pred) ** 2)
    
key = jax.random.PRNGKey(0)
t_n = jnp.arange(T)[:, None]

score = MLP(hidden_dim=128, out_dim=1, n_layers=5)
params = score.init(key, jnp.concatenate([x[:T], t_n/T], axis=1))


print(loss_fn(params, x[:T], t_n[:T], score, key))

key = jax.random.PRNGKey(0)

@flax.struct.dataclass
class Store:
  params: jnp.ndarray
  state: Any
  rng: Any
  step: int = 0

# we'll use adamw with some linear warmup and a cosine decay.
opt = optax.adam(learning_rate=3e-4)

#store = Store(params, opt.init(params), key, 0)
#pstore = replicate(store)

opt = optax.adam(learning_rate=3e-4)
opt_state = opt.init(params)

# Setup checkpoint manager

n_steps = 5000
n_batch = 128
T = 100
best_loss = float('inf')

checkpointer = Checkpointer('./checkpoint.pkl')
# Checkpoint load reads parameters from disk.
params= checkpointer.load()

"""
with trange(n_steps) as steps:
    for step in steps:

        # Draw a random batches from x
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(key, x.shape[0], shape=(n_batch,))
        
        x_batch = x[idx]
        t_n_batch = jax.random.choice(key, T, shape=(x_batch.shape[0], 1))

        loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, t_n_batch, score, key)
        updates, opt_state = opt.update(grads, opt_state, params)

        if loss < best_loss:
            best_loss = loss

        params = optax.apply_updates(params, updates)

        steps.set_postfix(val=loss)
    
checkpointer.save(params)
"""

def single_sample_fn(score, params, key, lambda_ddpm=1.):
    """ Ancestral sampling with `t_N` timesteps.
        `lambda_ddpm = 1` corresponds to the DDPM, `lambda_ddpm = 0` corresponds to the DDIM.
    """

    x_t = jax.random.normal(key, (1,))

    for t_i in range(T, 0, -1):

        t = (t_i / T)

        t = jnp.atleast_1d(t)
        
        key, _ = jax.random.split(key)
        eps = jax.random.normal(key, shape=x_t.shape) 
        eps_pred = score.apply(params, jnp.concatenate([x_t, t], axis=-1))

        # Often just sigma_t = beta_t is taken for simplicity
        sigma_t = jnp.sqrt((1 - prod_gamma_squared_t(t_i - 1)) / (1 - prod_gamma_squared_t(t_i)) * sigma_squared_t(t))

        x_t = 1 / gamma_t(t) * (x_t - sigma_squared_t(t) / (jnp.sqrt(1 - prod_gamma_squared_t(t_i))) * eps_pred) + lambda_ddpm * sigma_t * eps
        
    return x_t

def single_sample_and_log_w_fn(score, params, log_density, key, lambda_ddpm=1.):
    """ Ancestral sampling with `t_N` timesteps.
        `lambda_ddpm = 1` corresponds to the DDPM, `lambda_ddpm = 0` corresponds to the DDIM.
    """

    x_t = jax.random.normal(key, (1,))

    samples_t = [x_t]

    for t_i in range(T, 0, -1):

        t = (t_i / T)

        t = jnp.atleast_1d(t)
        
        key, _ = jax.random.split(key)
        eps = jax.random.normal(key, shape=x_t.shape) 
        eps_pred = score.apply(params, jnp.concatenate([x_t, t], axis=-1))

        # Often just sigma_t = beta_t is taken for simplicity
        sigma_t = jnp.sqrt((1 - prod_gamma_squared_t(t_i - 1)) / (1 - prod_gamma_squared_t(t_i)) * sigma_squared_t(t))

        x_t = 1 / gamma_t(t) * (x_t - sigma_squared_t(t) / (jnp.sqrt(1 - prod_gamma_squared_t(t_i))) * eps_pred) + lambda_ddpm * sigma_t * eps

        samples_t.append(x_t)

    # At this point, samples containts the augmentations (x^(T), ..., x^(0))

    x_0 = samples_t[-1]
    x_0_dot = jnp.dot(x_0, x_0)
    
    # log(q(x^{(T)}|x_{(0)})/p(x^{T}))
    x_T = samples_t[0]
    d = x_T.shape[0]
    # log q(x^{(0)})
    log_importance_weight = log_density(x_0)

    var_x_T = 1 - prod_gamma_squared_t(T)
    mean_x_T = prod_gamma_t(T)
    log_var_x_T = jnp.log(var_x_T)
    log_importance_weight += -0.5 * (d * log_var_x_T + (jnp.square(mean_x_T)*x_0_dot - 2. * mean_x_T * jnp.dot(x_T, x_0))/var_x_T)

    x_t = x_T

    # log(q(x^{(t-1)}|x^{(t)},x_{(0)})/p(x^{(t-1)}|x^{(t)}))
    for t_i in tqdm(range(T, 0, -1)):
        t = (t_i / T)
        t = jnp.atleast_1d(t)

        x_t_minus_1 = samples_t[t_i-1]

        var_x_t_minus_1 = (1 - prod_gamma_squared_t(t_i - 1)) / (1 - prod_gamma_squared_t(t_i)) * sigma_squared_t(t)
        a_t_minus_1 = prod_gamma_t(t_i-1)*(1 - gamma_squared_t(t_i))/(1 - gamma_squared_t(t_i))
        b_t_minus_1 = ((1 - gamma_squared_t(t_i-1))/(1 - gamma_squared_t(t_i)))*gamma_t(t)
        c_t = prod_gamma_t(t_i)
        d_t = jnp.sqrt(1 - gamma_squared_t(t_i))

        """
        jax.debug.print("var_x_t_minus_1={var_x_t_minus_1}", var_x_t_minus_1=var_x_t_minus_1)
        jax.debug.print("a_t_minus_1={a_t_minus_1}", a_t_minus_1=a_t_minus_1)
        jax.debug.print("b_t_minus_1={b_t_minus_1}", b_t_minus_1=b_t_minus_1)
        jax.debug.print("gamma_squared_t_1={gamma_squared_t}", gamma_squared_t=gamma_squared_t(t_i-1))
        jax.debug.print("gamma_squared_t={gamma_squared_t}", gamma_squared_t=gamma_squared_t(t_i))
        jax.debug.print("c_t={c_t}", c_t=c_t)
        jax.debug.print("d_t={d_t}", d_t=d_t)
        """

        eps_pred = score.apply(params, jnp.concatenate([x_t, t], axis=-1))
        x_0_pred_t = (x_t - d_t*eps_pred)/c_t

        #jax.debug.print("x_0={x_0}", x_0=x_0)
        #jax.debug.print("x_0_pred={x_0_pred}", x_0_pred=x_0_pred_t)

        x_0_pred_t_dot = jnp.dot(x_0_pred_t, x_0_pred_t)


        log_importance_weight += -0.5 * a_t_minus_1 *(
            a_t_minus_1*(x_0_dot - x_0_pred_t_dot) 
            - 2.*b_t_minus_1*jnp.dot(x_t, x_0 - x_0_pred_t)
            - 2. * jnp.dot(x_t_minus_1, x_0 - x_0_pred_t)
        ) / var_x_t_minus_1

        x_t = x_t_minus_1

    return samples_t[-1],log_importance_weight


sample_fn = partial(single_sample_and_log_w_fn, score, params, lambda x: log_gmm_density(x, prior, means, stds))
#sample_fn = partial(single_sample_fn, score, params)

n_samples = 100
sample_key = jax.random.split(key, n_samples ** 2)
print(f"T={T}")
x_sample = jax.vmap(sample_fn)(sample_key)
x_sample, log_importance_weight = jax.vmap(sample_fn)(sample_key)

# To reverse the transformation
x_sample = scaler.inverse_transform(x_sample)

print(log_importance_weight)
print(jax.nn.softmax(log_importance_weight - jnp.max(log_importance_weight)))

print(jnp.mean(log_importance_weight), jnp.std(log_importance_weight))

plt.hist(np.array(x_sample), density=True, bins=30)
plt.plot(x_values, density, label='Unnormalised Density Function', color='blue')
plt.show()

#plt.hist2d(x_sample[:, 0], x_sample[:, 1], bins=100)
#plt.xlim(-2 ,2)
#plt.ylim(-2, 2)

#plt.show()