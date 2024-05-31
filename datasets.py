from typing import List

import abc
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from jax.nn import softmax
from schedulers import NoiseScheduler
from typing import Callable

class DataModel(abc.ABC):

  def __init__(self, noise_scheduler: NoiseScheduler, *args, **kwargs):
    self._noise_scheduler = noise_scheduler

  @abc.abstractmethod
  def generate_data(self, key, n_samples: int):
      pass

  @abc.abstractmethod
  def density(self, x: jnp.array):
    pass

  @abc.abstractmethod
  def log_density(self, x: jnp.array):
    pass

  @abc.abstractmethod
  def score(self, x: jnp.array):
    pass

  @abc.abstractmethod
  def noisy_density(self, x: jnp.array, t: jnp.array):
    pass

  @abc.abstractmethod
  def noisy_log_density(self, x: jnp.array, t: jnp.array):
    pass

  @abc.abstractmethod
  def noisy_score(self, x: jnp.array, t: jnp.array):
    pass

  # Method to do reverse Langevin dynamics
  def noisy_score_reversed(self, x: jnp.array, t: jnp.array):
    return self.noisy_score(x, 1. - t)
  
def load_model(noise_scheduler: NoiseScheduler, model_name: str ="gmm"):
  if model_name == "gmm":
    return GMM(noise_scheduler)
  elif model_name == "normal":
    return Normal(noise_scheduler)
  else:
    ValueError("Undefined data model")
    
class GMM(DataModel):
  def __init__(
      self, 
      noise_scheduler: NoiseScheduler, 
      prior: jnp.array = jnp.array([0.5,0.5]), 
      means: jnp.array = jnp.array([[-1.],[1.]]), 
      stds:  jnp.array = jnp.array([[0.2],[0.2]]), 
    ):
    super().__init__(noise_scheduler)
    # Initialise GMM parameters
    self.prior = prior
    self.means = means
    self.stds = stds

  def generate_data(self, key, n_samples: int):
    key, subkey = jax.random.split(key)
    samples = []
    # Sample from the GMM components based on the mixture weights
    for i in range(n_samples):
        # Sample a component index based on the mixture weights
        component_idx = jax.random.choice(
          key + i, len(self.prior), p=self.prior
        )
        # Sample from the chosen component
        component_mean = self.means[component_idx]
        component_std = self.stds[component_idx]
        sample = component_mean + component_std*jax.random.normal(subkey + i)
        samples.append(sample)
    return jnp.stack(samples)

  def density(self, x: jnp.array):
    return jnp.sum(
      jnp.concatenate([self.prior[i]*norm.pdf(x, loc=self.means[i], scale=self.stds[i]) for i in range(len(self.prior))], axis=-1), 
      axis=-1, 
      keepdims=True
    )
  
  def log_density(self, x: jnp.array):
    return logsumexp(
      jnp.concatenate([jnp.log(self.prior[i])+norm.logpdf(x, loc=self.means[i], scale=self.stds[i]) for i in range(len(self.prior))], axis=-1), 
      axis=-1, 
      keepdims=True
    )
  
  def score(self, x: jnp.array):
    # Iterate over GMM components
    responsibilities = softmax(jnp.concatenate([jnp.log(self.prior[i])+norm.logpdf(x, loc=self.means[i], scale=self.stds[i]) for i in range(len(self.prior))], axis=-1))
    mean_diffs = jnp.concatenate([(self.means[i]-x)/(self.stds[i]**2) for i in range(len(self.prior))], axis=-1)
    return jnp.sum(responsibilities * mean_diffs, axis=-1, keepdims=True)
  
  def noisy_density(self, x: jnp.array, t: jnp.array):
    return jnp.sum(
      jnp.concatenate([
        self.prior[i]*
        norm.pdf(x, 
          loc=self.means[i]*self._noise_scheduler.alpha(t), 
          scale=jnp.sqrt((self._noise_scheduler.alpha(t)*self.stds[i])**2 + self._noise_scheduler.sigma_squared(t)),
        ) for i in range(len(self.prior))], axis=-1), 
      axis=-1, 
      keepdims=True
    )

  def noisy_log_density(self, x: jnp.array, t: jnp.array):
    return logsumexp(
      jnp.concatenate([
        jnp.log(self.prior[i])+
        norm.logpdf(x, 
          loc=self.means[i]*self._noise_scheduler.alpha(t), 
          scale=jnp.sqrt((self._noise_scheduler.alpha(t)*self.stds[i])**2 + self._noise_scheduler.sigma_squared(t)),
        ) for i in range(len(self.prior))], axis=-1), 
      axis=-1, 
      keepdims=True
    )
  
  def noisy_score(self, x: jnp.array, t: jnp.array):
    # Iterate over GMM components
    responsibilities = softmax(
      jnp.concatenate([jnp.log(self.prior[i])+
        norm.logpdf(x, 
          loc=self.means[i]*self._noise_scheduler.alpha(t), 
          scale=jnp.sqrt((self._noise_scheduler.alpha(t)*self.stds[i])**2 + self._noise_scheduler.sigma_squared(t))
        ) 
      for i in range(len(self.prior))], axis=-1)
    )
    mean_diffs = jnp.concatenate([
      (self.means[i]*self._noise_scheduler.alpha(t)-x)/(
        (self._noise_scheduler.alpha(t)*self.stds[i])**2 + self._noise_scheduler.sigma_squared(t)
      ) for i in range(len(self.prior))
    ], axis=-1)
    return jnp.sum(responsibilities * mean_diffs, axis=-1, keepdims=True)

# Plan to remove it as the GMM model can be used in this simpler scenario
class Normal(DataModel):
  def __init__(
      self, 
      noise_scheduler: NoiseScheduler, 
      mean: jnp.array = jnp.array([[2.]]), 
      std: jnp.array = jnp.array([[0.1]]),
    ):
    super().__init__(noise_scheduler)
    # Initialise Normal parameters
    self.mean = mean
    self.std = std

  def generate_data(self, key, n_samples):
    return self.mean + self.std * jax.random.normal(key, shape=(n_samples, 1))
  
  def density(self, x: jnp.array):
    return norm.pdf(x, loc=self.mean, scale=self.std)
  
  def log_density(self, x: jnp.array):
    return norm.logpdf(x, loc=self.mean, scale=self.std)

  def score(self, x: jnp.array):
    return -(x - self.mean)/(self.std**2)
  
  def noisy_density(self, x: jnp.array, t: jnp.array):
    return norm.pdf(
      x, 
      loc=self.mean*self._noise_scheduler.alpha(t), 
      scale=jnp.sqrt((self._noise_scheduler.alpha(t)*self.std)**2 + self._noise_scheduler.sigma_squared(t))
    )
  
  def noisy_log_density(self, x: jnp.array, t: jnp.array):
    return norm.pdf(
      x, 
      loc=self.mean*self._noise_scheduler.alpha(t), 
      scale=jnp.sqrt((self._noise_scheduler.alpha(t)*self.std)**2 + self._noise_scheduler.sigma_squared(t))
    )
  
  def noisy_score(self, x: jnp.array, t: jnp.array):
    return -(x - self._noise_scheduler.alpha(t)*self.mean)/(
      (self._noise_scheduler.alpha(t)*self.std)**2 + self._noise_scheduler.sigma_squared(t)
    )