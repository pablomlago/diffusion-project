from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import orbax
import orbax.checkpoint
from sklearn import datasets, preprocessing
from jax.scipy.stats import norm

import matplotlib.pyplot as plt
from tqdm import trange


def positional_encoding_for_list(positions_list, d_model):
    # Calculate the positional encodings for specific indices directly
    angle_rates = 1 / jnp.power(10000, (2 * (jnp.arange(d_model) // 2)) / d_model)
    angle_rads = jnp.outer(positions_list, angle_rates)  # Outer product to get angles for each position
    sines = jnp.sin(angle_rads[:, 0::2])
    cosines = jnp.cos(angle_rads[:, 1::2])
    pos_encoding = jnp.empty((len(positions_list), d_model))
    pos_encoding = pos_encoding.at[:, 0::2].set(sines)
    pos_encoding = pos_encoding.at[:, 1::2].set(cosines)
    return pos_encoding

prior = [0.5, 0.5]  # Priors for the two components
means = [-1, 1]     # Means for each component
stds = [0.5, 0.5]       # Standard deviations for each component

def log_gmm_density(x, prior, means, stds):
    # Iterate over GMM components
    log_components = jnp.array([
        jnp.log(prior[i]) 
        + norm.logpdf(x, loc=means[i], scale=stds[i]) for i in range(len(prior))
    ])
    return jax.nn.logsumexp(log_components)

print(log_gmm_density(jnp.array([[-1.0],[5.0],[0.0]]), prior, means, stds))
