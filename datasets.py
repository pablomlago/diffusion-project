from typing import List

import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from jax.nn import softmax

def generate_gmm(prior: List, means: List, stds: List, n_samples: int):
  # Generate the samples from the GMM
  # Determine component choices based on priors
  elements_per_component = np.random.multinomial(n_samples, prior, size=1,)[0]

  samples = [
      means[i]+stds[i]*jax.random.normal(jax.random.PRNGKey(0), shape=(num_elements,))
      for i, num_elements in enumerate(elements_per_component)
  ]
  labels = tuple([
      [i]*num_elements
      for i, num_elements in enumerate(elements_per_component)
  ])

  samples = jnp.array(np.concatenate(samples))
  samples = jnp.expand_dims(jax.random.permutation(jax.random.PRNGKey(0), samples), axis=-1)
  labels = jnp.array(np.concatenate(labels))
  # Return samples and labels
  return samples, labels

def gmm_density(x, prior, means, stds):
  # Iterate over GMM components
  density = prior[0]*norm.pdf(x, loc=means[0], scale=stds[0])
  for i in range(1, len(prior)):
    density += prior[i]*norm.pdf(x, loc=means[i], scale=stds[i])
  return density

def log_gmm_density(x, prior, means, stds):
  # Iterate over GMM components
  return logsumexp(
    jnp.concatenate([jnp.log(prior[i])+norm.logpdf(x, loc=means[i], scale=stds[i]) for i in range(len(prior))], axis=-1), 
    axis=-1, 
    keepdims=True
)

def score_gmm(x, prior, means, stds):
    # Iterate over GMM components
    responsibilities = softmax(jnp.concatenate([jnp.log(prior[i])+norm.logpdf(x, loc=means[i], scale=stds[i]) for i in range(len(prior))], axis=-1))
    mean_diffs = jnp.concatenate([(means[i]-x)/(stds[i]**2) for i in range(len(prior))], axis=-1)
    return jnp.sum(responsibilities * mean_diffs, axis=-1, keepdims=True)

class Normal1D:
  def __init__(self, mean: jnp.array, std: jnp.array, noise_schedule: jnp.array):
    self.mean = mean
    self.std = std
    self._noise_schedule = noise_schedule

    self._gamma_squared = 1. - self._noise_schedule
    self._gamma = jnp.sqrt(self._gamma_squared)
    self._prod_gamma = jnp.cumprod(self._gamma)
    self.prod_gamma_squared = jnp.cumprod(self._gamma_squared)

  def generate_samples(self, key, n_samples):
    return self.mean + self.std * jax.random.normal(key, shape=(n_samples, 1))
  
  def density(self, x: jnp.array):
    return norm.pdf(x, loc=self.mean, scale=self.std)
  
  def log_density(self, x: jnp.array):
    return norm.logpdf(x, loc=self.mean, scale=self.std)

  def score(self, x: jnp.array):
    return -(x - self.mean)/(self.std**2)
  
  def noisy_density(self, x: jnp.array, t: int):
    # Verify the step
    assert t < len(self._noise_schedule)

    return norm.pdf(
      x, 
      loc=self.mean*self._prod_gamma[t], 
      scale=jnp.sqrt((self._prod_gamma[t]*self.std)**2 + (1. - self.prod_gamma_squared[t]))
    )
  
  def noisy_log_density(self, x: jnp.array, t: int):
    # Verify the step
    assert t < len(self._noise_schedule)

    return norm.pdf(
      x, 
      loc=self.mean*self._prod_gamma[t], 
      scale=jnp.sqrt((self._prod_gamma[t]*self.std)**2 + (1. - self.prod_gamma_squared[t]))
    )
  
  def noisy_score(self, x: jnp.array, t: int):
    # Verify the step
    assert t < len(self._noise_schedule)

    return -(x - self._prod_gamma[t]*self.mean)/(
      (self._prod_gamma[t]*self.std)**2+(1. - self.prod_gamma_squared[t])
    )
  
  def noisy_score_reversed(self, x: jnp.array, t: int):
    # Verify the step
    assert t < len(self._noise_schedule)
    T = len(self._noise_schedule)
    # Score for reversed diffusion dynamics
    return self.noisy_score(x, T-t-1)
