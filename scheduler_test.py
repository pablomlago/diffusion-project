"""Unit tests for `schedulers.py`."""

from absl.testing import absltest

import chex
from schedulers import linear_schedule
import jax.numpy as jnp
import numpy as np

class SchedulersTest(absltest.TestCase):

  def test_linear_schedule(self):
    T = 10
    expected = jnp.linspace(start=0, stop=T, num=T)/T
    out = linear_schedule(T, t_min=0.0, t_max=1.0)
    chex.assert_trees_all_close(expected, out)

if __name__ == "__main__":
  absltest.main()