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
from datasets import generate_gmm, gmm_density
from diffusion import DDPM, Config
from langevin import LangevinConfig, LangevinDiffusion
from models import MLP
from schedulers import linear_schedule
from visualisation_utils import visualise_samples_density, plt_to_image
from utils import wrap_score_model_params

# Define flags
FLAGS = flags.FLAGS

# Diffusion params
flags.DEFINE_integer("T", 1000, "Number of diffusion steps")
flags.DEFINE_integer("ddim_step", 5, "DDIM step size")
flags.DEFINE_float("step_size", 0.1, "Size of each diffusion step")
# Target distribution params
flags.DEFINE_list("gmm_prior", [0.5, 0.5], 'GMM prior')
flags.DEFINE_list("gmm_means", [-1., 1.], 'GMM component means')
flags.DEFINE_list("gmm_stds", [0.2, 0.2], 'GMM component standard deviations')
# Training params
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("n_steps", 5000, "Number of training steps")
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
        name="diffusion-gmm",
        config={
            # Diffusion params
            "T": FLAGS.T,
            "step_size": FLAGS.step_size,
            "ddim_step": FLAGS.ddim_step,
            # Target distribution params
            "gmm_prior": FLAGS.gmm_prior,
            "gmm_means": FLAGS.gmm_means,
            "gmm_stds": FLAGS.gmm_stds,
            # Training params
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
    ddpm_config = Config(T=T)
    ddpm = DDPM(ddpm_config)
     
    @partial(jax.jit, static_argnums=(3,))
    def diffusion_loss(params, x, t, score, key):
        eps = jax.random.normal(key, shape=x.shape)
        x_t = ddpm.add_noise(x, t, key)
        # t is between 0 and T-1, both included. Therefore, t/(T-1) is in the interval [0,1]
        eps_pred = score.apply(params, jnp.concatenate([x_t, t/(T-1)], -1))
        # Unweighted objective
        return jnp.mean((eps - eps_pred) ** 2)

    @partial(jax.jit, static_argnums=(3,))
    def weighted_diffusion_loss(params, x, t, score, key):
        eps = jax.random.normal(key, shape=x.shape)
        x_t = ddpm.add_noise(x, t, key)
        # t is between 0 and T-1, both included. Therefore, t/(T-1) is in the interval [0,1]
        eps_pred = score.apply(params, jnp.concatenate([x_t, t/(T-1)], -1))
        return jnp.sum(((1.-ddpm.gamma_squared[t])/ddpm.gamma_squared[t])**2/(1.-ddpm.prod_gamma_squared_prev[t])*jnp.mean((eps - eps_pred) ** 2, axis=1))
  
    # Create lambda for GMM density
    gmm_density_fn = lambda x : gmm_density(x, config.gmm_prior, config.gmm_means, config.gmm_stds)
    # Generate samples from the GMM
    x, _ = generate_gmm(config.gmm_prior, config.gmm_means, config.gmm_stds, config.n_train_samples)
    # Standarise samples
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)

    # Set random seeds
    key = jax.random.PRNGKey(config.seed)

    # Initialise score network
    t_n = jnp.linspace(start=0, stop=T, num=T)
    score = MLP(hidden_dim=128, out_dim=1, n_layers=5)
    params = score.init(key, jnp.concatenate([x[:T], jnp.expand_dims(t_n/T, axis=-1)], axis=1))
        
    opt = optax.adam(learning_rate=config.lr)
    opt_state = opt.init(params)

    best_loss = float('inf')
    checkpointer = Checkpointer('./checkpoint.pkl')

    # Allow to use a different optimisation objective
    loss_fn = weighted_diffusion_loss if config.use_weighted_loss else diffusion_loss

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
    # params = checkpointer.load()

    """
    key, _ = jax.random.split(key)
    x_t = jax.random.normal(key, shape=(config.n_test_samples, 1))
    samples_per_step = [x_t]
    for t in tqdm(range(T-1, -1, -1)):
        t_array = jnp.repeat(jnp.atleast_1d(t), config.n_test_samples, axis=0)
        eps_pred = score.apply(params, jnp.concatenate([x_t, jnp.expand_dims(t_array/(T-1), axis=-1)], axis=-1))
        key, subkey = jax.random.split(key)
        x_t = ddpm.sample_previous_eps_pred(x_t, eps_pred, t, key)
        # Add to the samples-per-step
        samples_per_step.append(x_t)

    # To reverse the transformation
    x_t = scaler.inverse_transform(x_t)

    # Plot samples along with density
    fig, _ = visualise_samples_density([x_t], [gmm_density_fn], ["Target"])
    # Log the histogram to wandb
    wandb.log({"DDPM samples": wandb.Image(plt_to_image(fig))})

    # Sample using DDIM
    key, _ = jax.random.split(key)
    x_t = jax.random.normal(key, shape=(config.n_test_samples, 1))
    samples_per_step = [x_t]

    sample_traj = list(range(T, -1, -config.ddim_step))

    for i in tqdm(range(len(sample_traj)-1)):
        t = sample_traj[i]-1
        t_prev = sample_traj[i+1]-1
        t_array = jnp.repeat(jnp.atleast_1d(t), config.n_test_samples, axis=0)
        eps_pred = score.apply(params, jnp.concatenate([x_t, jnp.expand_dims(t_array/(T-1), axis=-1)], axis=-1))
        key, subkey = jax.random.split(key)
        x_t = ddpm.sample_previous_eps_pred_ddim(x_t, eps_pred, t, t_prev, key, eta=0.0)
        # Add to the samples-per-step
        samples_per_step.append(x_t)

    # To reverse the transformation
    x_t = scaler.inverse_transform(x_t)

    # Plot samples along with density
    fig, _ = visualise_samples_density([x_t], [gmm_density_fn], ["Target"])
    # Log the histogram to wandb
    wandb.log({"DDIM samples": wandb.Image(plt_to_image(fig))})
    """

    """Langevin sampling"""

    # Use langevin diffuser for sampling
    score_fn = wrap_score_model_params(params, score, config.T, ddpm.prod_gamma_squared, reversed=True)
    langevin_config = LangevinConfig(T = config.T, step_size= config.step_size)
    langevin_diffuser = LangevinDiffusion(langevin_config)

    # Sample using Langevin
    key, _ = jax.random.split(key)
    x_t = jax.random.normal(key, shape=(config.n_test_samples, 1))
    # Run Langevin dynamics
    key, _ = jax.random.split(key)

    x_t = langevin_diffuser.annealed_langevin(x_t, score_fn, 10 , key)

    x_t = (x_t - ddpm.sigma[0]*score_fn(x_t, 0)) / ddpm.gamma[0]

    # To reverse the transformation
    x_t = scaler.inverse_transform(x_t)

    # Plot samples along with density
    fig, _ = visualise_samples_density([x_t], [gmm_density_fn], ["Target"])
    plt.show()
    # Log the histogram to wandb
    wandb.log({"Langevin samples": wandb.Image(plt_to_image(fig))})

    wandb.finish()

if __name__ == '__main__':
    app.run(main)