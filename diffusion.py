import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from typing import List, Dict

from schedulers import linear_schedule
from utils import log_ratio_normal, log_ratio_normal_same_var

class Config:
    def __init__(self, T: int, scheduler = linear_schedule):
        self.T = T
        self.scheduler = scheduler

class DDPM:
    def __init__(self, config : Config):
        self.T = config.T
        # The scheduler gives the variance that is added at each step
        self.scheduler = config.scheduler
        
        self.sigma_squared = self.scheduler(self.T)
        self.sigma = jnp.sqrt(self.sigma_squared)
        # Variance-preserving AR(1)
        self.gamma_squared = 1. - self.sigma_squared
        self.gamma = jnp.sqrt(self.gamma_squared)
        self.prod_gamma = jnp.cumprod(self.gamma)
        self.prod_gamma_squared = jnp.cumprod(self.gamma_squared)
        self.prod_gamma_squared_prev = jnp.pad(self.prod_gamma_squared, (1, 0), constant_values=0.)

        # Saved to avoid recomputing sqrt(1-\bar{\alpha})
        self.std_marginal_x_t = jnp.sqrt(1 - self.prod_gamma_squared)

        # Weights to each term in the loss
        self.loss_weights = jnp.ones((self.T,), dtype=np.float32)

    def add_noise(self, x_0: jnp.ndarray, t: jnp.array, key):
        # Generate noise
        eps = jax.random.normal(key, shape=x_0.shape)
        # Add noise to samples
        return self.prod_gamma[t]*x_0 + self.std_marginal_x_t[t]*eps
    
    def q_mean_var(self, x_t: jnp.array, x_0: jnp.array, t: int):
        # Quick sanity check
        assert t >= 1 and t < self.T
        # q(x^{t-1};x_{t},x_{0})
        a_t = self.prod_gamma[t-1]*(1-self.gamma_squared[t])
        b_t = (1-self.gamma_squared[t-1])*self.gamma[t]
        mean = (a_t*x_0 + b_t*x_t)/(1-self.prod_gamma_squared[t])
        var = (1 - self.prod_gamma_squared[t-1]) / (1 - self.prod_gamma_squared[t]) * self.sigma_squared[t]
        return mean, var
        
    def p_mean_var(self, x_t: jnp.array, eps_pred: jnp.array, t: int):
        # Quick sanity check
        assert t >= 0 and t < self.T
        # p(x^{t-1}; x_{0})
        if t >= 1:
            mean = (x_t - self.sigma_squared[t] / self.std_marginal_x_t[t] * eps_pred) / self.gamma[t]
            var = (1 - self.prod_gamma_squared[t-1]) / (1 - self.prod_gamma_squared[t]) * self.sigma_squared[t]
        elif t == 0:
            mean = (x_t - self.sigma[t]*eps_pred)/self.gamma[t]
            var = jnp.array(0.0)
        return mean, var
    
    def log_w(self, x: List[jnp.array], model: nn.Module, params: Dict, log_unnormalised_density_fn):
        # log(N(x^T; self.prod_gamma[T-1]x_0, (1-self.prod_gamma_squared[T-1])I)/N(x^T; 0, I))
        x_T = x[0] # List is reversed
        x_0 = x[-1]
        log_w = log_ratio_normal(
            x_T, 
            self.prod_gamma[self.T-1]*x_0, 
            1-self.prod_gamma_squared[self.T-1],
            jnp.zeros_like(x_0), 
            jnp.ones_like(self.prod_gamma_squared[self.T-1])
        )
        # log(N(x^T; self.prod_gamma[T-1]x_0, (1-self.prod_gamma_squared[T-1])I)/N(x^T; 0, I))
        x_t_prev = x_T
        for t in range(self.T-1, 0, -1):
            x_t = x[self.T-t]
            mean_q, var_q = self.q_mean_var(x_t_prev, x_0, t)
            t_array = jnp.repeat(jnp.atleast_1d(t), x_t.shape[0], axis=0)
            eps_pred = model.apply(params, jnp.concatenate([x_t_prev, jnp.expand_dims(t_array/(self.T-1), axis=-1)], axis=-1))
            mean_p, _ = self.p_mean_var(x_t_prev, eps_pred, t)
            log_w += log_ratio_normal_same_var(x_t, mean_q, mean_p, var_q)
            x_t_prev = x_t
        # p(x^{0}|x^{1})=1, since the noise is deterministic. q^()
        log_w += jnp.squeeze(log_unnormalised_density_fn(x_0))
        return log_w
    
    def sample_previous_eps_pred(self, x_t: jnp.ndarray, eps_pred: jnp.ndarray, t: int, key, lambda_ddpm=1.):
        """
        `lambda_ddpm = 1` corresponds to the DDPM, `lambda_ddpm = 0` corresponds to the DDIM.
        """
        assert t >= 0 and t < self.T
        # Case distinction to simplify equations. The main motivation for this is that
        # the ScoreNet takes a number between [0.0, 1.0] as input.
        if t > 0:
            # Sample noise
            eps = jax.random.normal(key, shape=x_t.shape)
            # Standard deviation ancestral normal, could also be set to sigma
            sigma_t = jnp.sqrt((1 - self.prod_gamma_squared[t-1]) / (1 - self.prod_gamma_squared[t]) * self.sigma_squared[t])
            # Sampling of previous augmentation
            x_t = (x_t - self.sigma_squared[t] / self.std_marginal_x_t[t] * eps_pred) / self.gamma[t] + lambda_ddpm * sigma_t * eps
        else:
            x_t = (x_t - self.sigma[t]*eps_pred) / self.gamma[t]
        # Previous sample
        return x_t

