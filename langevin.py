import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm, trange
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict
import functools

from typing import List, Dict, Callable
import jax
from schedulers import linear_schedule
from utils import log_ratio_normal, log_ratio_normal_same_var

@chex.dataclass
class _CMCDScanState:
  x_t_prev: chex.Array
  mean_x_t_prev: chex.Array
  drift_correction_x_t_prev: chex.Array
  log_w: chex.Array
  key: chex.Array

class LangevinConfig:
    def __init__(self, T: int, step_size: float):
        self.T = T
        self.step_size = step_size

class LangevinDiffusion:
    def __init__(self, config : LangevinConfig):
        self.T = config.T
        self.step_size = config.step_size

    def ula_diffusion(self, x_0: jnp.array, score: Callable, key, sigma : float = 1.0):
        # sigma can be tweaked to perform overdamped Langevin dynamics
        # Run ULA for the specified number of steps
        x_t = x_0
        for _ in tqdm(range(self.T)):
            key, _ = jax.random.split(key)
            # Brownian motion
            eps = jax.random.normal(key, shape=x_0.shape)
            # ULA-step
            x_t = x_t + (sigma**2)*self.step_size * score(x_t) + jnp.sqrt(2*self.step_size*sigma)*eps
        return x_t
    
    def mala_diffusion(self, x_0: jnp.array, log_density: Callable, score: Callable, key, sigma : float = 1.0):
        # sigma can be tweaked to perform overdamped Langevin dynamics
        # Run ULA for the specified number of steps
        def log_ratio_proposal(x_t: jnp.array, x_t_prev: jnp.array, score_x_t_prev: jnp.array):
            return (
                log_density(x_t_prev) - 
                jnp.sum(
                    (x_t - x_t_prev + (sigma**2)*self.step_size * score_x_t_prev)**2, axis=-1, keepdims=True
                ) / (4. * self.step_size)
            )
        # Running mean acceptance probability
        running_mean_acceptance_probability = 0.0
        # Initialisation
        x_t_prev = x_0
        score_x_t_prev = score(x_t_prev)
        # MALA loop
        with trange(self.T) as steps:
            for i in steps:
                key, _ = jax.random.split(key)
                # ULA update
                eps = jax.random.normal(key, shape=x_0.shape)
                x_t = x_t_prev + (sigma**2)*self.step_size * score_x_t_prev + jnp.sqrt(2*self.step_size*sigma)*eps
                # Compute and save score to avoid computing it twice
                score_x_t = score(x_t)
                # Compute acceptance threshold
                log_acceptance_probability = log_ratio_proposal(x_t_prev, x_t, score_x_t) - log_ratio_proposal(x_t, x_t_prev, score_x_t_prev)
                log_acceptance_probability = jnp.minimum(log_acceptance_probability, jnp.zeros_like(log_acceptance_probability))
                log_alpha = jnp.log(jax.random.uniform(key, shape=log_acceptance_probability.shape))
                # Select based on acceptance probability, also update scores
                acceptances = log_alpha <= log_acceptance_probability
                x_t = jnp.where(acceptances, x_t, x_t_prev)
                score_x_t = jnp.where(acceptances, score_x_t, score_x_t_prev)
                # For the next step save the appropiate elements
                x_t_prev = x_t
                score_x_t_prev = score_x_t

                # Useful logging for debugging
                running_mean_acceptance_probability = float(running_mean_acceptance_probability*i/(i+1) + jnp.sum(acceptances)/(x_0.shape[0]*(i+1)))
                steps.set_postfix(acc=running_mean_acceptance_probability)
        # Return last element
        return x_t
    
    def annealed_langevin(self, x_0: jnp.array, score: Callable, steps_per_level: int, key, sigma : jnp.array = None):
        x_t = x_0
        with trange(self.T) as steps:
            for t in steps:
                annealing_factor = 1.0 if sigma is None else sigma[t]/sigma[self.T-1]
                annealed_step_size = self.step_size * annealing_factor
                for _ in range(steps_per_level):
                    # ULA update
                    key, _ = jax.random.split(key)
                    eps = jax.random.normal(key, shape=x_0.shape)
                    x_t = x_t + annealed_step_size * score(x_t, t) + jnp.sqrt(2*annealed_step_size)*eps
        # Return latest step
        return x_t
        
    def cmcd(
            self, 
            x_0: 
            jnp.array, 
            score: Callable,
            log_density_target : Callable,
            log_density_sample: Callable, 
            key, 
            sigma : jnp.array = None,
        ):
        # Auxiliar function to simplify computations
        def compute_cmcd_terms(x_t, sigma, score, t):
            annealing_factor = 1.0 if sigma is None else sigma[t]/sigma[self.T-1]
            annealed_step_size = self.step_size * annealing_factor
            score_x_t = score(x_t, t)
            mean_x_t = x_t + annealed_step_size * score_x_t
            return annealed_step_size, mean_x_t
        # Use value on initialisation as first iterate
        x_t_prev = x_0
        annealed_step_size_prev, mean_x_t_prev = compute_cmcd_terms(x_t_prev, sigma, score, 0)
        # Log-density for the sampling distribution
        log_w = - log_density_sample(x_t_prev)

        with trange(1,self.T) as steps:
            for t in steps:
                # Generate random noise
                key, _ = jax.random.split(key)
                eps = jax.random.normal(key, shape=x_0.shape)
                # Generate next iterate using 
                x_t = mean_x_t_prev + jnp.sqrt(2*annealed_step_size_prev)*eps
                # Compute terms for log_w
                annealed_step_size, mean_x_t = compute_cmcd_terms(x_t, sigma, score, t)
                # Update log_w
                log_w += -0.5 * (
                    x_t.shape[1]*(jnp.log(2*annealed_step_size)-jnp.log(2*annealed_step_size_prev)) +
                    jnp.sum((x_t_prev - mean_x_t)**2, axis=1, keepdims=True)/(2*annealed_step_size) - 
                    jnp.sum((x_t - mean_x_t_prev)**2, axis=1, keepdims=True)/(2*annealed_step_size_prev)
                )
                # Book-keeping for next step
                x_t_prev, annealed_step_size_prev, mean_x_t_prev = x_t, annealed_step_size, mean_x_t
        # Log-density for the target distribution
        log_w += log_density_target(x_t)
        # Return latest step
        return x_t, log_w
    
    def _cmcd_step(
        self,
        cmcd_state: _CMCDScanState,
        i: int,
        first_step: bool,
        params: Dict,
        drift_correction: nn.Module,
        score: Callable,
    ):
        # Unpack terms needed for computing the CMCD loss
        x_t_prev = cmcd_state.x_t_prev
        # These should be initialised to zero in the first step
        mean_x_t_prev = cmcd_state.mean_x_t_prev
        drift_correction_x_t_prev = cmcd_state.drift_correction_x_t_prev
        # Initialised to the log density of the sample dsitribution in the first step
        log_w = cmcd_state.log_w
        # Unpack random key
        key = cmcd_state.key
        # Prepare steps to be passed to the drift correction network
        t_input = jnp.expand_dims(jnp.repeat(jnp.atleast_1d(i)/(self.T-1), x_t_prev.shape[0], axis=0), axis=-1)

        if first_step:
            x_t = x_t_prev
            # Evaluate score at starting step
            score_x_t = score(x_t, 0.)
            mean_x_t = x_t + self.step_size * score_x_t
            # Prepare input to the drift correction network
            x_t_input = jnp.concatenate([x_t, t_input], axis=-1)
            # Compute drift correction
            drift_correction_x_t = drift_correction.apply(params, x_t_input) * self.step_size
        else:
            # Split key and pass to the next step
            key, subkey = jax.random.split(key)
            # Generate random noise to perform the step
            eps = jax.random.normal(subkey, shape=x_t_prev.shape)
            # Generate next iterate
            x_t = mean_x_t_prev + drift_correction_x_t_prev + jnp.sqrt(2*self.step_size)*eps
            # Compute CMCD terms to pass to the next iterate
            score_x_t = score(x_t, i/(self.T-1))
            mean_x_t = x_t + self.step_size * score_x_t
            # Prepare input for the drift correction network
            x_t_input = jnp.concatenate([x_t, t_input], axis=-1)
            # Compute drift correction term
            drift_correction_x_t = drift_correction.apply(params, x_t_input) * self.step_size
            # Update log_w
            log_w += -0.5 * (
                jnp.sum((x_t_prev - mean_x_t + drift_correction_x_t)**2, axis=1, keepdims=True) - 
                jnp.sum((x_t - mean_x_t_prev - drift_correction_x_t_prev)**2, axis=1, keepdims=True)
            ) / (2.*self.step_size)
        # Prepare updated CMCD state
        new_cmcd_state = _CMCDScanState(
            x_t_prev=x_t,
            mean_x_t_prev=mean_x_t,
            drift_correction_x_t_prev=drift_correction_x_t,
            log_w=log_w,
            key=key,
        )
        # Do not use accum_state, as we are interested in the last samples only
        return new_cmcd_state, None
    
    def cmcd_diffusion(
        self,
        params: jnp.array,
        x_0: jnp.array,
        drift_correction: nn.Module,
        score: Callable,
        log_density_target : Callable,
        log_density_sample: Callable,
        key: jnp.array,
    ):
        # Prepare common arguments for the CMCD step function
        common_args = dict(
            params=params,
            drift_correction=drift_correction,
            score=score,
        )
        # Initial state for the first iteration
        initial_cmcd_state = _CMCDScanState(
            x_t_prev=x_0,
            mean_x_t_prev=None,
            drift_correction_x_t_prev=None,
            log_w = - log_density_sample(x_0),
            key = key,
        )
        # First step needs special treatment
        cmcd_state, _ = self._cmcd_step(
            initial_cmcd_state,
            i=0,
            first_step=True,
            **common_args,
        )
        # Then scan through the rest.
        cmcd_step_fn = functools.partial(
            self._cmcd_step,
            first_step=False,
            **common_args
        )
        # Perform the CMCD step iterations
        cmcd_state, _ = jax.lax.scan(
            cmcd_step_fn,
            cmcd_state,
            jnp.arange(self.T - 1) + 1,
            length = self.T - 1,
        )
        # Recover last samples
        x_T = cmcd_state.x_t_prev
        # Add log-density of the target
        log_w = cmcd_state.log_w + log_density_target(x_T)
        # Return latest step and log_w
        return x_T, log_w

    def cmcd_train_loss(
        self,
        params: jnp.array,
        x_0: jnp.array,
        drift_correction: nn.Module,
        score: Callable,
        log_density_target : Callable,
        log_density_sample: Callable,
        key: jnp.array,
    ):
        _, log_w = self.cmcd_diffusion(
            params,
            x_0,
            drift_correction,
            score,
            log_density_target,
            log_density_sample,
            key
        )
        # Ensure that loss is not proportional to the number of steps
        # so it is more easily comparable accross runs
        return jnp.mean(-log_w) / self.T
    
    def cmcd_kl_loss(
        self,
        params: jnp.array,
        x_0: jnp.array,
        drift_correction: nn.Module,
        score: Callable,
        log_density_target : Callable,
        log_density_sample: Callable,
        key: jnp.array,
    ):
        _, log_w = self.cmcd_diffusion(
            params,
            x_0,
            drift_correction,
            score,
            log_density_target,
            log_density_sample,
            key
        )
        # We are estimating the KL divergence, which is positive,
        # in here we use an alternative estimator for the KL divergence
        return jnp.mean((jnp.exp(log_w)-1.)-log_w) / self.T



            
