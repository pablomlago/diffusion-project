from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from absl import app, flags
from flax.jax_utils import replicate
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import optax
from sklearn import preprocessing
from tqdm import tqdm, trange
from typing import Any
import wandb

from checkpointer import Checkpointer
from datasets import generate_gmm, gmm_density, score_gmm
from diffusion import DDPM, Config
from langevin import LangevinConfig, LangevinDiffusion
from models import MLP
from schedulers import geometric_progression_schedule
from visualisation_utils import visualise_samples_density, plt_to_image, visualise_densities_1d
from utils import wrap_dsm_model_params, wrap_fn
# Define flags
FLAGS = flags.FLAGS

# Diffusion params
flags.DEFINE_integer("T", 10, "Number of diffusion steps")
flags.DEFINE_integer("ddim_step", 5, "DDIM step size")
flags.DEFINE_float("step_size", 0.1, "Size of each diffusion step")
# Target distribution params
flags.DEFINE_list("gmm_prior", [0.5, 0.5], 'GMM prior')
flags.DEFINE_list("gmm_means", [-1., 1.], 'GMM component means')
flags.DEFINE_list("gmm_stds", [0.5, 0.5], 'GMM component standard deviations')
# Varianche scheduling
flags.DEFINE_float("initial_std", 1., "Standard deviation for the highest noise level")
flags.DEFINE_float("end_std", 0.1, "Standard deviation for the lowest noise level")
# Training params
flags.DEFINE_string("checkpoint_dir", "./sm-checkpoint.pkl", "Directory to save checkpoint")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("n_steps", 10_000, "Number of training steps")
flags.DEFINE_integer("n_batch", 128, "Batch size for training")
flags.DEFINE_float("lr", 3e-4, "Adam optimiser LR")
flags.DEFINE_boolean("use_weighted_loss", False, "Use VAE loss or unweighted loss")
# Test hyperparameters
flags.DEFINE_integer("n_train_samples", 500_000, "Number of training samples to generate")
flags.DEFINE_integer("n_test_samples", 10_000, "Number of test samples to generate")
# Model hyperparameters
flags.DEFINE_integer("hidden_dim", 128, "Dimension of hidden layers in the model")
flags.DEFINE_integer("n_layers", 3, "Number of layers in the model")
# Define number of validation steps
flags.DEFINE_integer("n_steps_eval", 10, "Frequency for qualitative evaluation")
flags.DEFINE_integer("n_samples_eval", 4096, "Number of samples in qualitative evaluation")


def main(argv):

    wandb.init(
        project="master-project",
        name="score-matching-gmm",
        mode="disabled",
        config={
            # Diffusion params
            "T": FLAGS.T,
            "step_size": FLAGS.step_size,
            "ddim_step": FLAGS.ddim_step,
            # Target distribution params
            "gmm_prior": FLAGS.gmm_prior,
            "gmm_means": FLAGS.gmm_means,
            "gmm_stds": FLAGS.gmm_stds,
            # Variance scheduling
            "initial_std": FLAGS.initial_std,
            "end_std": FLAGS.end_std,
            # Training params
            "checkpoint_dir": FLAGS.checkpoint_dir,
            "seed": FLAGS.seed,
            "n_steps": FLAGS.n_steps,
            "n_batch": FLAGS.n_batch,
            "lr": FLAGS.lr,
            "use_weighted_loss": FLAGS.use_weighted_loss,
            # Test hyperparameters
            "n_train_samples": FLAGS.n_train_samples,
            "n_test_samples": FLAGS.n_test_samples,
            # Model hyperparameters
            "hidden_dim": FLAGS.hidden_dim,
            "n_layers": FLAGS.n_layers,
            # Num steps eval
            "n_steps_eval": FLAGS.n_steps_eval,
            "n_samples_eval": FLAGS.n_samples_eval,
        }
    )
    config = wandb.config

    # Initialisatoin of DDPM
    T = config.T
    sigma_schedule = geometric_progression_schedule(T, sigma_start=config.initial_std, sigma_end=config.end_std)

    @partial(jax.jit, static_argnums=(3,))
    def dsm_loss(params, x, t, score, key):
        eps = jax.random.normal(key, shape=x.shape)
        x_t = x + sigma_schedule[t]*eps
        score_pred = score.apply(params, jnp.concatenate([x_t, sigma_schedule[t]], axis=-1))
        return jnp.mean(jnp.sum((score_pred + eps)**2, axis=1))
  
    # Create lambda for GMM density
    gmm_density_fn = wrap_fn(gmm_density, config.gmm_prior, config.gmm_means, config.gmm_stds)
    # Generate samples from the GMM
    x, _ = generate_gmm(config.gmm_prior, config.gmm_means, config.gmm_stds, config.n_train_samples)
    # Standarise samples
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)

    # Set random seeds
    key = jax.random.PRNGKey(config.seed)

    # Initialise score network
    score = MLP(hidden_dim=128, out_dim=1, n_layers=config.n_layers)
    t_array = jnp.expand_dims(jnp.arange(T), axis=-1)
    # Initialise for all levels of noise
    params = score.init(key, jnp.concatenate([x[:T], sigma_schedule[t_array]], axis=-1))
        
    opt = optax.adam(learning_rate=config.lr)
    opt_state = opt.init(params)

    best_loss = float('inf')
    checkpointer = Checkpointer(config.checkpoint_dir)

    # Allow to use a different optimisation objective
    loss_fn = dsm_loss

    with trange(config.n_steps) as steps:
        for step in steps:
            # Draw a random batches from x
            key, _ = jax.random.split(key)
            idx = jax.random.choice(key, x.shape[0], shape=(config.n_batch,))
            
            x_batch = x[idx]
            t_n_batch = jax.random.choice(key, T, shape=(x_batch.shape[0], 1))

            loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, t_n_batch, score, key)
            updates, opt_state = opt.update(grads, opt_state, params)

            if loss < best_loss:
                best_loss = loss
                checkpointer.save(params)

            params = optax.apply_updates(params, updates)
            # Log losses to wandb
            wandb.log({"Train Loss": loss})
            # Add loss to error bar
            steps.set_postfix(val=loss)

    # Load best checkpoint
    params = checkpointer.load()
    # Save checkpoint
    wandb.save(config.checkpoint_dir)

    """Langevin sampling"""

    # Use langevin diffuser for sampling
    score_fn = wrap_dsm_model_params(params, score, config.T, sigma_schedule)
    langevin_config = LangevinConfig(T = config.T, step_size=config.step_size)
    langevin_diffuser = LangevinDiffusion(langevin_config)

    # Sample using Langevin
    key, _ = jax.random.split(key)
    x_t = jax.random.normal(key, shape=(config.n_test_samples, 1))

    # Run Langevin dynamics
    key, _ = jax.random.split(key)
    x_t = langevin_diffuser.annealed_langevin(x_t, score_fn, 100, key, sigma_schedule)

    # Use Tweedie's formula to remove noise in the last step
    x_t = x_t + sigma_schedule[-1]*score_fn(x_t, T-1)

    # To reverse the transformation
    x_t = scaler.inverse_transform(x_t)
    
    # Plot samples along with density
    fig, _ = visualise_samples_density([x_t], [gmm_density_fn], ["Model Samples"])
    plt.show()
    # Log the histogram to wandb
    wandb.log({"Langevin samples": wandb.Image(plt_to_image(fig))})

    """ Visualisations to compare the ground-truth scores with the models """
    # With this information, the marginal distributions can be computed
    gmm_means = jnp.expand_dims(jnp.array(config.gmm_means), axis=1) # (2,1)
    gmm_stds = jnp.expand_dims(jnp.array(config.gmm_stds), axis=1) # (2,1)
    # Expand dimensions for broadcasting
    sigma_schedule_expanded = jnp.expand_dims(sigma_schedule, axis=0) # (1,T)
    # GMM distribution after each noising step (see Target Score Matching paper)
    gmm_means_per_step = gmm_means * jnp.ones_like(sigma_schedule_expanded)
    gmm_stds_per_step = jnp.sqrt(gmm_stds**2 + sigma_schedule_expanded**2)

    # Definition of ground-truth scores and for different levels of noise
    data_score = wrap_fn(score_gmm, config.gmm_prior, config.gmm_means, config.gmm_stds)
    score_x_most_noisy = wrap_fn(score_gmm, config.gmm_prior, gmm_means_per_step[:,0], gmm_stds_per_step[:,0])
    score_x_lest_noisy = wrap_fn(score_gmm, config.gmm_prior, gmm_means_per_step[:,-1], gmm_stds_per_step[:,-1])
    score_net_most_noisy = wrap_fn(score_fn, 0)
    score_net_less_noisy = wrap_fn(score_fn, T-1)
    fig, _ = visualise_densities_1d(
        [data_score, score_x_most_noisy, score_x_lest_noisy, score_net_most_noisy, score_net_less_noisy], 
        labels=[
            "Data", 
            r"Score, $\sigma$={:.2f}".format(sigma_schedule[0]), 
            r"Score, $\sigma$={:.2f}".format(sigma_schedule[-1]),
            r"Net, $\sigma$={:.2f}".format(sigma_schedule[0]), 
            r"Net, $\sigma$={:.2f}".format(sigma_schedule[-1]),
        ])
    # Log the histogram to wandb
    wandb.log({"Scores": wandb.Image(plt_to_image(fig))})
    
    wandb.finish()

if __name__ == '__main__':
    app.run(main)