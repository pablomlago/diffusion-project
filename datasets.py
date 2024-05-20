from typing import List

import numpy as np
import jax.numpy as jnp
import jax

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
