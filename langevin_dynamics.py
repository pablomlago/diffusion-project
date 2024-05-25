import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm

from datasets import generate_gmm, log_gmm_density, gmm_density, score_gmm
from schedulers import linear_schedule
from langevin import LangevinConfig, LangevinDiffusion


# Generate RandomKey
key = jax.random.PRNGKey(0)

# Configuration for the Langevin Diffuser
T = 1000
langevin_config = LangevinConfig(
    T=T, step_size=0.01
)
# x_0 is N(2, 0.01)
# Start the diffusion sampling from a standard normal
num_samples = 10_000
_mean_x_0 = 2.
_std_x_0 = 0.1
x_0 = 2. + 0.1*jax.random.normal(key, shape=(num_samples, 1))

# Score function based on noise
_sigma_squared = linear_schedule(T)
_gamma_squared = 1. - _sigma_squared
_gamma = jnp.sqrt(_gamma_squared)
_prod_gamma = jnp.cumprod(_gamma)
_prod_gamma_squared = jnp.cumprod(_gamma_squared)

# Score function for initial normal distribution
def score_fn(x_t: jnp.array, t: int):
    return -(x_t - _prod_gamma[t]*_mean_x_0)/((_prod_gamma[t]*_std_x_0)**2 + (1. - _prod_gamma_squared[t]))

# Score function for initial normal distribution
def score_fn_reversed(x_t: jnp.array, t: int):
    return -(x_t - _prod_gamma[T-t-1]*_mean_x_0)/((_prod_gamma[T-t-1]*_std_x_0)**2 + (1. - _prod_gamma_squared[T-t-1]))

langevin_diffuser = LangevinDiffusion(langevin_config)

# Reversed diffusion
# x_T is N(0, 1)
key, _ = jax.random.split(key)
x_T = jax.random.normal(key, shape=(num_samples, 1))
# Split the random key
key, _ = jax.random.split(key)
x_0, log_w = langevin_diffuser.cmcd(
    x_T, 
    score_fn_reversed,
    lambda x : norm.logpdf(x, loc=_mean_x_0, scale=_std_x_0),
    lambda x : norm.logpdf(x),
    key,)

print(f"Expected mean: {_mean_x_0}, Sample mean: {jnp.mean(x_0)}, IS mean: {jnp.mean(jnp.exp(log_w)*x_0)}")
print(f"Expected Z: {1.}, Sample Z: {jnp.mean(norm.pdf(x_0, loc=_mean_x_0, scale=_std_x_0))}, IS mean: {jnp.mean(norm.pdf(jnp.exp(log_w)*x_0, loc=_mean_x_0, scale=_std_x_0))}")

x_values = np.linspace(-4, 4, 1000)
density_x_0 = norm.pdf(x_values, loc=_mean_x_0, scale=_std_x_0)
density_x_T = norm.pdf(x_values, loc=_mean_x_0*_prod_gamma[T-1], scale=jnp.sqrt(_prod_gamma[T-1]*_std_x_0)**2 + (1. - _prod_gamma_squared[T-1]))
density_standard = norm.pdf(x_values)
plt.hist(np.array(x_0), density=True, bins=50, color='blue')
plt.hist(np.array(x_T), density=True, bins=50, color='red')
plt.plot(x_values, np.array(density_x_0), label='Density x_0', color='blue')
plt.plot(x_values, np.array(density_x_T), label='Density x_T', color='red')
plt.plot(x_values, np.array(density_standard), label='Density Standard', color='green')
plt.show()

plt.hist(np.exp(np.array(log_w)))
plt.yscale('log')
plt.show()