"""Unit tests for `datasets.py`."""

from absl.testing import absltest

import chex
from datasets import score_gmm, log_gmm_density, gmm_density
import jax.numpy as jnp
import jax
import numpy as np

class GMMTest(absltest.TestCase):
    
    # GMM parameters
    prior = (0.5, 0.5)
    means = (-1, 1)
    stds = (0.2, 0.2)

    def test__schedule(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, shape=(10,1))

        gmm_density_fn = lambda x : gmm_density(x, self.prior, self.means, self.stds)
        log_gmm_density_fn = lambda x : log_gmm_density(x, self.prior, self.means, self.stds)
        score_gmm_fn = lambda x : score_gmm(x, self.prior, self.means, self.stds)

        score_gmm_grad = jax.grad(lambda x : log_gmm_density_fn(x)[0])
        expected_grad = jnp.expand_dims(jnp.concatenate([score_gmm_grad(x[i]) for i in range(x.shape[0])], axis=0), axis=-1)
    
        chex.assert_trees_all_close(jnp.log(gmm_density_fn(x)), log_gmm_density_fn(x))
        chex.assert_trees_all_close(expected_grad, score_gmm_fn(x))

if __name__ == "__main__":
  absltest.main()