import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Callable

def log_ratio_normal(x, mean_1, var_1, mean_2, var_2):
   k = x.shape[0]
   return - 0.5 * (k * (jnp.log(var_1) - jnp.log(var_2)) + jnp.sum((x - mean_1)**2, axis=1)/var_1 - jnp.sum((x-mean_2)**2, axis=1)/var_2)

def log_ratio_normal_same_var(x, mean_1, mean_2, var):
    return (
        jnp.sum(x*(mean_1 - mean_2), axis=1, keepdims=True) 
        - 0.5*(
            jnp.sum(mean_1**2, axis=1, keepdims=True) -
            jnp.sum(mean_2**2, axis=1, keepdims=True) 
        )
    )/var

# Wrap a model with fixed params to use in diffusion
def wrap_score_model_params(params: Dict, model: nn.Module, T: int, prod_gamma_squared: jnp.array,  reversed : bool = False) -> Callable:
    def _apply_model(x: jnp.array, t: int):
        # Allows to do reverse diffusions
        t = t if not reversed else T-t-1
        t_array = jnp.repeat(jnp.atleast_1d(t), x.shape[0], axis=0)
        x_input = jnp.concatenate([x, jnp.expand_dims(t_array/(T-1), axis=-1)], axis=-1)
        # Model predicts noise, score is computed based on this.
        return -model.apply(params, x_input)
    return _apply_model

# Wrap a model with fixed params to use in diffusion
def wrap_fn(fn: Callable, *args) -> Callable:
    return lambda x : fn(x, *args)

# Wrap a model with fixed params to use in diffusion
def wrap_dsm_model_params(params: Dict, model: nn.Module, T : int, sigma_schedule: jnp.array, reversed : bool = False) -> Callable:
    def _apply_model(x: jnp.array, t: int):
        # Allows to do reverse diffusions
        t = t if not reversed else T-t-1
        t_array = jnp.expand_dims(jnp.repeat(jnp.atleast_1d(t), x.shape[0], axis=0), axis=-1)
        # Model predicts noise, score is computed based on this.
        return model.apply(params, jnp.concatenate([x, sigma_schedule[t_array]], axis=-1)) / sigma_schedule[t_array]
    return _apply_model