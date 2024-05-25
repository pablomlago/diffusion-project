from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm
from tqdm import trange
import optax
import wandb
from absl import app, flags

from PIL import Image

from datasets import generate_gmm, log_gmm_density, gmm_density, score_gmm
from models import MLP
from schedulers import linear_schedule
from langevin import LangevinConfig, LangevinDiffusion
from checkpointer import Checkpointer

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
# Test hyperparameters
flags.DEFINE_integer("num_samples", 10_000, "Number of samples to generate")
# Model hyperparameters
flags.DEFINE_integer("hidden_dim", 128, "Dimension of hidden layers in the model")
flags.DEFINE_integer("n_layers", 3, "Number of layers in the model")

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
            # Test hyperparameters
            "num_samples": FLAGS.num_samples,
            # Model hyperparameters
            "hidden_dim": FLAGS.hidden_dim,
            "n_layers": FLAGS.n_layers,
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

    # Compute scores for target distribution
    _mean_x_0 = config.mean_target
    _std_x_0 = config.std_target
    # Score function based on noise
    _sigma_squared = linear_schedule(T)
    _gamma_squared = 1. - _sigma_squared
    _gamma = jnp.sqrt(_gamma_squared)
    _prod_gamma = jnp.cumprod(_gamma)
    _prod_gamma_squared = jnp.cumprod(_gamma_squared)

    # Score function for initial normal distribution
    def score_fn(x_t: jnp.array, t: int):
        return -(x_t - _prod_gamma[t]*_mean_x_0)/((_prod_gamma[t]*_std_x_0)**2 + (1. - _prod_gamma_squared[t]))

    # Score function for initial normal distribution
    def score_fn_reversed(x_t: jnp.array, t: int):
        return -(x_t - _prod_gamma[T-t-1]*_mean_x_0)/((_prod_gamma[T-t-1]*_std_x_0)**2 + (1. - _prod_gamma_squared[T-t-1]))

    # Initialise drift correction model
    drift_correction = MLP(hidden_dim=config.hidden_dim, out_dim=1, n_layers=config.n_layers)
    # Sample elements for initialisation
    key, subkey = jax.random.split(key)
    x_T = jax.random.normal(key, shape=(T, 1))
    t_T = jnp.expand_dims(jnp.linspace(start=0, stop=T, num=T)/T, axis=-1)
    # Init with an element for all steps
    params = drift_correction.init(key, jnp.concatenate([x_T, t_T], axis=-1))

    # Function to set all parameters to zeros
    def set_params_to_zeros(params):
        return jax.tree_util.tree_map(lambda x: x, params)
    
    #params = set_params_to_zeros(params)

    # Optimiser initialisation
    opt = optax.adam(learning_rate=3e-4)
    opt_state = opt.init(params)

    # Training parameters
    n_steps = config.n_steps
    n_batch = config.n_batch

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
                score_fn_reversed,
                lambda x : norm.logpdf(x, loc=_mean_x_0, scale=_std_x_0),
                lambda x : norm.logpdf(x),
                (1. - _prod_gamma_squared),
                key,
            )
            # Update model params
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            # wandb logging
            wandb.log({"Train Loss": loss})
            # Update progress bar
            steps.set_postfix(val=loss)

    # Qualitative model evaluation
    num_samples = config.num_samples
    # Use for sampling
    key, subkey = jax.random.split(key)
    x_T = jax.random.normal(key, shape=(num_samples, 1))
    x_0, _ = langevin_diffuser.cmcd_train(
        params,
        x_T,
        drift_correction,
        score_fn_reversed,
        lambda x : norm.logpdf(x, loc=_mean_x_0, scale=_std_x_0),
        lambda x : norm.logpdf(x),
        (1. - _prod_gamma_squared),
        key,
    )

    # Log histogram to qualitatively study sample quality
    fig, ax = plt.subplots()
    # Compute densities
    x_values = np.linspace(-4, 4, 1000)
    density_x_0 = norm.pdf(x_values, loc=_mean_x_0, scale=_std_x_0)
    density_x_T = norm.pdf(x_values, loc=_mean_x_0*_prod_gamma[T-1], scale=jnp.sqrt(_prod_gamma[T-1]*_std_x_0)**2 + (1. - _prod_gamma_squared[T-1]))
    density_standard = norm.pdf(x_values)

    plt.hist(np.array(x_0), density=True, bins=50, label="Final Samples", color='blue')
    plt.hist(np.array(x_T), density=True, bins=50, label="Initial Samples", color='red')
    plt.plot(x_values, np.array(density_x_0), label='Target Density', color='blue')
    plt.plot(x_values, np.array(density_x_T), label='Initial Density', color='red')
    plt.plot(x_values, np.array(density_standard), label='Sample Density', color='green')
    plt.legend()
    # Save the plot to a buffer
    from io import BytesIO
    image_data = BytesIO() #Create empty in-memory file
    plt.savefig(image_data, format='png') #Save pyplot figure to in-memory file
    image_data.seek(0) #Move stream position back to beginning of file 

    # Log the histogram to wandb
    wandb.log({"Trained Model Samples": wandb.Image(Image.open(image_data))})

    wandb.finish()

if __name__ == '__main__':
    app.run(main)