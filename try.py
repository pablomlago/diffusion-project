import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

def geometric_progression(a, l, T):
    if T == 1:
        # If there is only one term, both a and l must be the same
        return [a]
    
    # Calculate the common ratio
    r = (l / a) ** (1 / (T - 1))
    
    # Generate the terms in the geometric progression
    gp = [a * (r ** i) for i in range(T)]
    return gp

# Example usage:
a = 2   # Initial value
l = 32  # End value
T = 5   # Total number of terms

sequence = geometric_progression(a, l, T)
print("Geometric Progression Sequence:", sequence)

# Generate RandomKey
key = jax.random.PRNGKey(0)

n_samples = 100_000

samples_q = jax.random.normal(key, shape=(n_samples,))
log_density_fn_q = lambda x : norm.logpdf(x, loc=0., scale=1.)

key, _ = jax.random.split(key)
samples_p = 1. + jax.random.normal(key, shape=(n_samples,))
log_density_fn_p = lambda x : norm.logpdf(x, loc=1., scale=1.)

log_r = log_density_fn_q(samples_p) - log_density_fn_p(samples_p)
print(log_r[:10])
r = jnp.exp(log_r)
print(r[:10])

k1 = -log_r
k2 = (r-1) - log_r
k3 = 0.5*(log_r)**2


def kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Compute the KL divergence between two normal distributions.

    Args:
    mu1 (float): Mean of the first normal distribution.
    sigma1 (float): Standard deviation of the first normal distribution.
    mu2 (float): Mean of the second normal distribution.
    sigma2 (float): Standard deviation of the second normal distribution.

    Returns:
    float: The KL divergence between the two distributions.
    """
    term1 = jnp.log(sigma2 / sigma1)
    term2 = (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
    return term1 + term2 - 0.5

# Example usage
mu1 = 0.0
sigma1 = 1.0
mu2 = 1.0
sigma2 = 1.0

kl_div = kl_divergence(mu1, sigma1, mu2, sigma2)
print(f"The KL divergence is: {kl_div}")

print(jnp.mean(k1), jnp.var(k1))
print(jnp.mean(k2), jnp.var(k2))
print(jnp.mean(k3), jnp.var(k3))

plt.hist(k1, label='k1', alpha=0.5, bins=50)
plt.hist(k2, label='k2', alpha=0.5, bins=50)
plt.show()


