from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from tqdm import trange
import optax
import wandb
from absl import app, flags

from PIL import Image

from datasets import generate_gmm, log_gmm_density, gmm_density, score_gmm, Normal1D
from models import MLP
from schedulers import linear_schedule
from langevin import LangevinConfig, LangevinDiffusion
from checkpointer import Checkpointer
from visualisation_utils import plt_to_image, visualise_samples_density
from utils import wrap_fn

# Define flags
FLAGS = flags.FLAGS

# Diffusion params
flags.DEFINE_integer("T", 50, "Number of diffusion steps")
flags.DEFINE_float("step_size", 0.01, "Size of each diffusion step")
# Target distribution params
flags.DEFINE_float("mean_target", 2.0, "Mean of the target distribution")
flags.DEFINE_float("std_target", 0.1, "Standard deviation of the target distribution")
# Training params
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("n_steps", 0, "Number of training steps")
flags.DEFINE_integer("n_batch", 128, "Batch size for training")
flags.DEFINE_float("lr", 3e-4, "Adam optimiser LR")
# Test hyperparameters
flags.DEFINE_integer("num_samples", 10_000, "Number of samples to generate")
# Model hyperparameters
flags.DEFINE_integer("hidden_dim", 128, "Dimension of hidden layers in the model")
flags.DEFINE_integer("n_layers", 3, "Number of layers in the model")
# Define number of validation steps
flags.DEFINE_integer("n_steps_eval", 10, "Frequency for qualitative evaluation")
flags.DEFINE_integer("n_samples_eval", 4096, "Number of samples in qualitative evaluation")

def main(argv):
    # Initialize wandb and save hyperparameters
    wandb.init(
        project="master-project",
        name="cmcd-gaussian",
        config={
            # Diffusion params
            "T": FLAGS.T,
            "step_size": FLAGS.step_size,
            # Target distribution params
            "mean_target": FLAGS.mean_target,
            "std_target": FLAGS.std_target,
            # Training params
            "seed": FLAGS.seed,
            "n_steps": FLAGS.n_steps,
            "n_batch": FLAGS.n_batch,
            "lr": FLAGS.lr,
            # Test hyperparameters
            "num_samples": FLAGS.num_samples,
            # Model hyperparameters
            "hidden_dim": FLAGS.hidden_dim,
            "n_layers": FLAGS.n_layers,
            # Num steps eval
            "n_steps_eval": FLAGS.n_steps_eval,
            "n_samples_eval": FLAGS.n_samples_eval,
        }
    )
    config = wandb.config

    # Generate RandomKey
    key = jax.random.PRNGKey(config.seed)

    # Configuration for the Langevin Diffuser
    T = config.T
    langevin_config = LangevinConfig(
        T=T, step_size=config.step_size,
    )
    langevin_diffuser = LangevinDiffusion(langevin_config)

    # Initialise distribution
    normal_distribution = Normal1D(
        mean=jnp.array([[config.mean_target]]),
        std=jnp.array([[config.std_target]]),
        noise_schedule=linear_schedule(T)
    )
    # Initialise drift correction model
    drift_correction = MLP(hidden_dim=config.hidden_dim, out_dim=1, n_layers=config.n_layers)

    # Sample elements for initialisation
    key, _ = jax.random.split(key)
    x_T = jax.random.normal(key, shape=(T, 1))
    t_T = jnp.expand_dims(jnp.linspace(start=0, stop=T, num=T)/T, axis=-1)
    # Init with an element for all steps
    params = drift_correction.init(key, jnp.concatenate([x_T, t_T], axis=-1))
    
    # Optimiser initialisation
    opt = optax.adam(learning_rate=config.lr)
    opt_state = opt.init(params)

    # Training parameters
    n_steps = config.n_steps
    n_batch = config.n_batch

    checkpointer = Checkpointer('./cmcd_checkpoint.pkl')
    # Best loss
    best_loss = float('inf')

    # Run training loop
    with trange(n_steps) as steps:
        for step in steps:
            key, subkey = jax.random.split(key)
            x_T = jax.random.normal(key, shape=(n_batch, 1))
            # Update key before running an iteration
            key, subkey = jax.random.split(key)
            train_loss_fn = jax.jit(langevin_diffuser.cmcd_train_loss, static_argnums=(2,3,4,5))
            loss, grads = jax.value_and_grad(train_loss_fn)(
                params,
                x_T,
                drift_correction,
                normal_distribution.noisy_score_reversed,
                normal_distribution.log_density,
                lambda x : norm.logpdf(x),
                (1. - normal_distribution.prod_gamma_squared),
                key,
            )
            # Save model params
            if loss < best_loss:
                # Update best loss
                best_loss = loss
                # Save checkpoint
                checkpointer.save(params)
            # Update model params
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            # wandb logging
            wandb.log({"Train Loss": loss})
            # Compute histogram of log_w
            if step % config.n_steps_eval == 0:
                key, subkey = jax.random.split(key)
                x_T = jax.random.normal(key, shape=(config.n_samples_eval, 1))
                x_0, log_w = langevin_diffuser.cmcd_train(
                    params,
                    x_T,
                    drift_correction,
                    normal_distribution.noisy_score_reversed,
                    normal_distribution.log_density,
                    lambda x : norm.logpdf(x),
                    (1. - normal_distribution.prod_gamma_squared),
                    key,
                )
                # Create histogram for -log_w
                wandb.log({"Log w": wandb.Histogram(np.array(-log_w))})
                # Samples log-likelihood
                log_likelihood = np.mean(normal_distribution.log_density(x_0))
                # Log-likelihood should increase during training
                wandb.log({"Mean log-likelihood": log_likelihood})
                # Computation of effective sample size
                adjusted_sample_size = config.n_samples_eval/(1. + jnp.var(jnp.exp(log_w - logsumexp(log_w, axis=0, keepdims=True) + jnp.log(config.n_samples_eval))))
                # Log-likelihood should increase during training
                wandb.log({"Adjusted sample size": adjusted_sample_size})
            # Update progress bar
            steps.set_postfix(val=loss)

    # Qualitative model evaluation
    num_samples = config.num_samples
    # Use for sampling
    key, subkey = jax.random.split(key)
    x_T = jax.random.normal(key, shape=(num_samples, 1))
    x_0, log_w = langevin_diffuser.cmcd_train(
        params,
        x_T,
        drift_correction,
        normal_distribution.noisy_score_reversed,
        normal_distribution.log_density,
        lambda x : norm.logpdf(x),
        (1. - normal_distribution.prod_gamma_squared),
        key,
    )

    # Compute densities
    fig, ax = visualise_samples_density([x_T, x_0], [
        wrap_fn(normal_distribution.noisy_density, T-1), normal_distribution.density, norm.pdf,
    ], ["sample", "target", "standard"])
    wandb.log({"Trained Model Samples": wandb.Image(plt_to_image(fig))})
    # Create final histogram
    wandb.log({"Test Log w": wandb.Histogram(np.array(-log_w))})
    # Finish
    wandb.finish()

if __name__ == '__main__':
    app.run(main)