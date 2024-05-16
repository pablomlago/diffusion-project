import jax.numpy as jnp

def log_ratio_gaussians(x, mu1, a1, mu2, a2):
    d = x.shape[0]
    diff_log_var = d * jnp.log(a1 / a2)
    distance_1 = jnp.sum((x - mu1) ** 2, axis=1) / a1
    distance_2 = jnp.sum((x - mu2) ** 2, axis=1) / a2
    log_ratio = -0.5 * (diff_log_var + distance_1 - distance_2)
    return log_ratio

def log_ratio_gaussians_same_variance(x, mu1, mu2, var):
    log_ratio = -0.5 * (jnp.sum(mu1 ** 2) - jnp.sum(mu2 ** 2) - 2*jnp.sum(x * (mu1 - mu2), axis=1)) / var
    return log_ratio

# Example usage
x = jnp.array([[1.0, 2.0],[1.0, 2.0]])
mu1 = jnp.array([0.5, 1.5])
mu2 = jnp.array([1.5, 2.5])
a1 = jnp.array([2.0])
a2 = jnp.array([3.0])

result = log_ratio_gaussians(x, mu1, a1, mu2, a2)
print(result)