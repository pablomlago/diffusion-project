"""Unit tests for `utils.py`."""

from absl.testing import absltest

import jax
import chex
import jax.numpy as jnp
import numpy as np

from utils import log_ratio_normal, log_ratio_normal_same_var

class DiffusionTest(absltest.TestCase):

    def test_log_ratio_normal(self):
        pass

    def test_log_ration_normal_same_var(self):
        pass

if __name__ == "__main__":
  absltest.main()