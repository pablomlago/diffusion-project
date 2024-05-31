import chex
import jax.numpy as jnp
from jax.nn import sigmoid
import numpy as np
from typing import Callable

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

def simple_linear_schedule(t: jnp.array, clip_min=1e-6):
  return jnp.clip(t, clip_min, 1.)

def sigmoid_schedule(t: jnp.array, start=-3, end=3, tau=1.0, clip_min=1e-6):
  v_start = sigmoid(start / tau)
  v_end = sigmoid(end / tau)
  output = sigmoid((t * (end - start) + start) / tau)
  output = (v_end - output) / (v_end - v_start)
  return jnp.clip(1.-output, clip_min, 1.)

def cosine_schedule(t: jnp.array, start=0, end=1, tau=1, clip_min=1e-6):
  v_start = jnp.cos(start * jnp.pi / 2) ** (2 * tau)
  v_end = jnp.cos(end * jnp.pi / 2) ** (2 * tau)
  output = jnp.cos((t * (end - start) + start) * jnp.pi / 2) ** (2 * tau)
  output = (v_end - output) / (v_end - v_start)
  return jnp.clip(1. - output, clip_min, 1.)

class NoiseScheduler:
    def __init__(self, scheduler: Callable = simple_linear_schedule):
        self.scheduler = scheduler

    def sigma_squared(self, t: jnp.array):
        return self.scheduler(t)

    def sigma(self, t: jnp.array):
        return jnp.sqrt(self.scheduler(t))

    def alpha(self, t: jnp.array):
        return jnp.sqrt(1. - self.scheduler(t))
