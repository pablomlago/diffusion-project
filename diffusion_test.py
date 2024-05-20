"""Unit tests for `diffusion.py`."""

from absl.testing import absltest

import jax
import chex
import jax.numpy as jnp
import numpy as np

from diffusion import Config, DDPM

class DiffusionTest(absltest.TestCase):

    def test_ddpm_initalisation(self):
        # Steps are [0.0, 0.5, 1.0]
        config = Config(3, lambda _ : jnp.array([0., 5./9., 8./9.]))
        ddpm = DDPM(config)

        expected_sigma_squared = jnp.array([0., 5./9., 8./9.])
        expected_sigma = jnp.array([0., jnp.sqrt(5.)/3., jnp.sqrt(8.)/3.])
        expected_gamma = jnp.array([1., 2./3., 1./3.])
        expected_prod_gamma = jnp.array([1., 2./3., 2./9.])
        expected_gamma_squared = jnp.array([1., 4./9., 1./9.])
        expected_prod_gamma_squared = jnp.array([1., 4./9., 4./81.])
        expected_std_marginal_x_t = jnp.array([0., jnp.sqrt(5.)/3., jnp.sqrt(77.)/9.])

        chex.assert_trees_all_close(expected_sigma_squared, ddpm.sigma_squared)
        chex.assert_trees_all_close(expected_sigma, ddpm.sigma)
        chex.assert_trees_all_close(expected_gamma_squared, ddpm.gamma_squared)
        chex.assert_trees_all_close(expected_gamma, ddpm.gamma)
        chex.assert_trees_all_close(expected_prod_gamma, ddpm.prod_gamma)
        chex.assert_trees_all_close(expected_prod_gamma_squared, ddpm.prod_gamma_squared)
        chex.assert_trees_all_close(expected_std_marginal_x_t, ddpm.std_marginal_x_t)

        x = [jnp.array([[0.5]]),jnp.array([[1.0]]),jnp.array([[1.5]]),jnp.array([[2.0]])]
        expected_log_w = jnp.array([-0.5 * (jnp.log(77./81.) + 81./77.*(0.5-4./9.)**2 - 0.5**2)])
        # chex.assert_trees_all_close(expected_log_w, ddpm.log_w(x))

        expected_q_mean_2 = jnp.array([24./77. + 385./648.])
        #ddpm_q_mean_2, ddpm_q_var_2 = ddpm.q_mean_var(x[0],x[-1],3)
        # chex.assert_trees_all_close(expected_q_mean_2, ddpm_q_mean_2)

    def test_add_noise(self):
        config = Config(3, lambda _ : jnp.array([0., 5./9., 8./9.]))
        ddpm = DDPM(config)

        key = jax.random.PRNGKey(0)
        # Epsilon for the given seed
        eps = jnp.array([[1.8160863 , -0.75488514],[0.33988908, -0.53483534]])
        # (Batch Dimension, Feature Dimension)
        x_0 = jnp.ones((2, 2), dtype=np.float32)
        x_1 = ddpm.add_noise(x_0, 0, key)
        x_2 = ddpm.add_noise(x_0, 1, key)
        x_3 = ddpm.add_noise(x_0, 2, key)
        # No noise is added in the first step
        chex.assert_trees_all_close(x_0, x_1)
        chex.assert_trees_all_close(2./3.*x_0+jnp.sqrt(5.)/3.*eps, x_2)
        chex.assert_trees_all_close(2./9.*x_0+jnp.sqrt(77.)/9.*eps, x_3)

    def test_q_mean_var_and_p_mean_var(self):
        config = Config(3, lambda _ : jnp.array([0., 5./9., 8./9.]))
        ddpm = DDPM(config)

        key = jax.random.PRNGKey(0)
        # Epsilon for the given seed
        eps = jnp.array([[1.8160863 , -0.75488514],[0.33988908, -0.53483534]])
        # (Batch Dimension, Feature Dimension)
        x_0 = jnp.ones((2, 2), dtype=np.float32)
        x_1 = ddpm.add_noise(x_0, 0, key)
        x_2 = ddpm.add_noise(x_0, 1, key)
        eps_2 = (x_2 - 2.*x_0/3.)*3./jnp.sqrt(5.)
        x_3 = ddpm.add_noise(x_0, 2, key)
        eps_3 = (x_3 - 2.*x_0/9.)*9./jnp.sqrt(77.)

        # q(x^2|x^3,x^0)
        expected_q_mean_2 = 81./77.*(16./27.*x_0 + 5./27.*x_3)
        expected_q_var_2 = 40./77.
        q_mean_2, q_var_2 = ddpm.q_mean_var(x_3, x_0, 2)
        # q(x^2|x^1, x^0)
        expected_q_mean_1 = x_0
        expected_q_var_1 = 0.0
        q_mean_1, q_var_1 = ddpm.q_mean_var(x_2, x_0, 1)

        # p(x^2|x^3)
        p_mean_2, p_var_2 = ddpm.p_mean_var(x_3, eps_3, 2)
        # p(x^1|x^2)
        p_mean_1, p_var_1 = ddpm.p_mean_var(x_2, eps_2, 1)
        # p(x^0|x^1)
        p_mean_0, p_var_0 = ddpm.p_mean_var(x_1, eps, 0)
        # Check q distributions
        chex.assert_trees_all_close(expected_q_mean_2, q_mean_2)
        chex.assert_trees_all_close(expected_q_var_2, q_var_2)
        chex.assert_trees_all_close(expected_q_mean_1, q_mean_1)
        chex.assert_trees_all_close(expected_q_var_1, q_var_1)
        # Check p distributions
        chex.assert_trees_all_close(expected_q_mean_2, p_mean_2)
        chex.assert_trees_all_close(expected_q_var_2, p_var_2)
        chex.assert_trees_all_close(expected_q_mean_1, p_mean_1)
        chex.assert_trees_all_close(expected_q_var_1, p_var_1)
        chex.assert_trees_all_close(x_1, p_mean_0)
        chex.assert_trees_all_close(jnp.array(0.0), p_var_0)

    def test_log_w(self):
        config = Config(2, lambda _ : jnp.array([5./9., 8./9.]))
        ddpm = DDPM(config)

        key = jax.random.PRNGKey(0)
        # Epsilon for the given seed
        eps = jnp.array([[1.8160863 , -0.75488514],[0.33988908, -0.53483534]])
        # (Batch Dimension, Feature Dimension)
        x_0 = jnp.ones((2, 2), dtype=np.float32)
        x_1 = ddpm.add_noise(x_0, 0, key)
        eps_1 = (x_1 - 2.*x_0/3.)*3./jnp.sqrt(5.)
        x_2 = ddpm.add_noise(x_0, 2, key)
        eps_2 = (x_2 - 2.*x_0/9.)*9./jnp.sqrt(77.)

        # Check log_w
        class DummyScoreNet:
            def apply(self, params, input):
                t = input[0,-1]
                if jnp.abs(t - 0.0) < 1e-3:
                    return eps_1
                elif jnp.abs(t - 1.0) < 1e-3:
                    return eps_2
        dummy_model = DummyScoreNet()

        dummy_model = DummyScoreNet()
        expected_log_w = - 0.5 * (2*jnp.log(77./81.) + jnp.sum((x_2 - 2./9.*x_0)**2, axis=-1)*81./77. - jnp.sum(x_2**2, axis=-1))
        # Call to log_w
        log_w = ddpm.log_w([x_2, x_1, x_0], dummy_model, {}, lambda x: jnp.zeros((x.shape[0],)))
        chex.assert_trees_all_close(expected_log_w, log_w, atol=1e-3)


if __name__ == "__main__":
  absltest.main()