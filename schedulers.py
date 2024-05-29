import chex
import jax.numpy as jnp
import numpy as np

_Array = chex.Array

def linear_schedule(
    T: int,
    t_min: float = 1e-4, 
    t_max: float = 0.02
) -> _Array:
    return t_min + (t_max - t_min) * jnp.linspace(start=0, stop=T, num=T) / T

def geometric_progression_schedule(
    T: int,
    sigma_start: float = 20., 
    sigma_end: float = 1.,
) -> _Array:
    # Base case
    if T == 1:
        return jnp.array([sigma_start], dtype=np.float32)
    # Calculate the common ratio
    r = (sigma_end / sigma_start) ** (1 / (T - 1))
    return jnp.array([sigma_start * (r ** i) for i in range(T)], dtype=np.float32)


