import chex
import jax.numpy as jnp

_Array = chex.Array

def linear_schedule(
    T: int,
    t_min: float = 1e-4, 
    t_max: float = 0.02
) -> _Array:
    return t_min + (t_max - t_min) * jnp.linspace(start=0, stop=T, num=T) / T

