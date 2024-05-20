import jax.numpy as jnp

def log_ratio_normal(x, mean_1, var_1, mean_2, var_2):
   k = x.shape[0]
   return - 0.5 * (k * (jnp.log(var_1) - jnp.log(var_2)) + jnp.sum((x - mean_1)**2, axis=1)/var_1 - jnp.sum((x-mean_2)**2, axis=1)/var_2)

def log_ratio_normal_same_var(x, mean_1, mean_2, var):
   return - 0.5 * (jnp.sum((x - mean_1)**2, axis=1) - jnp.sum((x-mean_2)**2, axis=1)) / var