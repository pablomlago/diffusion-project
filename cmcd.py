from functools import partial
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
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
from haikunator import Haikunator

from PIL import Image

from datasets import load_model
from models import MLP
from schedulers import NoiseScheduler, cosine_schedule
from langevin import LangevinConfig, LangevinDiffusion
from checkpointer import Checkpointer
from visualisation_utils import plt_to_image, visualise_samples_density
from utils import wrap_fn

# Define flags
FLAGS = flags.FLAGS

# Diffusion params
flags.DEFINE_integer("T", 50, "Number of diffusion steps")
flags.DEFINE_float("step_size", 0.01, "Size of each diffusion step")
# Which model to use
flags.DEFINE_enum("dataset", "gmm", ["gmm", "normal"], "Dataset choice")
# Training params
flags.DEFINE_string("checkpoint_dir", ".", "Directory to save checkpoint")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("n_steps", 1, "Number of training steps")
flags.DEFINE_integer("n_batch", 128, "Batch size for training")
flags.DEFINE_float("lr", 3e-4, "Adam optimiser LR")
flags.DEFINE_boolean("use_kl_loss", False, "Use alternative estimator for KL divergence")
# Test hyperparameters
flags.DEFINE_integer("n_samples_train", 10_000, "Number of samples to use in training")
# Model hyperparameters
flags.DEFINE_integer("hidden_dim", 128, "Dimension of hidden layers in the model")
flags.DEFINE_integer("n_layers", 3, "Number of layers in the model")
# Define number of validation steps
flags.DEFINE_integer("n_steps_eval", 10, "Frequency for qualitative evaluation")
flags.DEFINE_integer("n_samples_val", 4096, "Number of samples for validation")

def main(argv):
    config={
        # Diffusion params
        "T": FLAGS.T,
        "step_size": FLAGS.step_size,
        # Model of the data
        "dataset": FLAGS.dataset,
        # Training params
        "checkpoint_dir": FLAGS.checkpoint_dir,
        "seed": FLAGS.seed,
        "n_steps": FLAGS.n_steps,
        "n_batch": FLAGS.n_batch,
        "lr": FLAGS.lr,
        "use_kl_loss": FLAGS.use_kl_loss,
        # Test hyperparameters
        "n_samples_train": FLAGS.n_samples_train,
        # Model hyperparameters
        "hidden_dim": FLAGS.hidden_dim,
        "n_layers": FLAGS.n_layers,
        # Num steps eval
        "n_steps_eval": FLAGS.n_steps_eval,
        "n_samples_val": FLAGS.n_samples_val,
    }
    train_single(config)

def train_sweep(config=None):
    wandb.init(
        config=config,
    )
    train(wandb.config)

def train_single(config):
    # Generate ID for the run
    run_id = Haikunator().haikunate(delimiter='-', token_length=0)
    # Initialize wandb and save hyperparameters
    wandb.init(
        project="master-project",
        name=f"cmcd-gmm-{run_id}",
        config=config,
        mode="disabled",
    )
    train(wandb.config, run_id)


def train(config=None, run_id=None):
    # Generate ID for the run
    run_id = Haikunator().haikunate(delimiter='-', token_length=0) if run_id is None else run_id
    # Generate RandomKey
    key = jax.random.PRNGKey(config.seed)

    # Configuration for the Langevin Diffuser
    T = config.T
    langevin_config = LangevinConfig(
        T=T, step_size=config.step_size,
    )
    langevin_diffuser = LangevinDiffusion(langevin_config)
    noise_scheduler = NoiseScheduler(scheduler=cosine_schedule)

    # Target distribution parameters
    data_model = load_model(noise_scheduler, model_name=config.dataset)

    # Initialise drift correction model
    drift_correction = MLP(hidden_dim=config.hidden_dim, out_dim=1, n_layers=config.n_layers)

    # Sample elements for initialisation
    key, subkey = jax.random.split(key)
    # Generate datasets for training and validation
    x_train = jax.random.normal(key, shape=(config.n_samples_train, 1))
    x_val = jax.random.normal(subkey, shape=(config.n_samples_val, 1))

    # Split key for initialisation
    key, subkey = jax.random.split(key)
    # Initialise score network
    t_T = jnp.arange(T)
    # Init with an element for all steps
    params = drift_correction.init(key, jnp.concatenate([x_train[t_T], jnp.expand_dims(t_T, axis=-1)/(T-1)], axis=-1))

    # Zero output layer to start from a better initial point
    output_layer_name = list(params['params'].keys())[-1]
    params["params"][output_layer_name]["kernel"] = jnp.zeros_like(params["params"][output_layer_name]["kernel"])
    params["params"][output_layer_name]["bias"] = jnp.zeros_like(params["params"][output_layer_name]["bias"])
    
    # Optimiser initialisation
    opt = optax.adam(learning_rate=config.lr)
    opt_state = opt.init(params)

    # Training parameters
    n_steps = config.n_steps
    n_batch = config.n_batch

    checkpoint_dir_name = f"{config.checkpoint_dir}/cmcd-{run_id}.pkl"

    checkpointer = Checkpointer(checkpoint_dir_name)
    # Best loss
    best_loss = float('inf')

    loss_fn = langevin_diffuser.cmcd_kl_loss if config.use_kl_loss else langevin_diffuser.cmcd_train_loss

    # Run training loop
    with trange(n_steps) as steps:
        for step in steps:
            key, subkey = jax.random.split(key)
            # Draw a random batch from the train data
            batch_idxs = jax.random.choice(key, x_train.shape[0], shape=(n_batch,))
            x_T = x_train[batch_idxs]
            # Compute loss
            train_loss_fn = jax.jit(loss_fn, static_argnums=(2,3,4,5))
            loss, grads = jax.value_and_grad(train_loss_fn)(
                params,
                x_T,
                drift_correction,
                data_model.noisy_score_reversed,
                data_model.log_density,
                lambda x : norm.logpdf(x),
                subkey,
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
                # Use validation data
                x_0, log_w = langevin_diffuser.cmcd_diffusion(
                    params,
                    x_val,
                    drift_correction,
                    data_model.noisy_score_reversed,
                    data_model.log_density,
                    lambda x : norm.logpdf(x),
                    subkey,
                )
                # Samples log-likelihood
                log_likelihood = np.mean(data_model.log_density(x_0))
                # Computation of effective sample size
                adjusted_sample_size = config.n_samples_val/(1. + jnp.var(jnp.exp(log_w - logsumexp(log_w, axis=0, keepdims=True) + jnp.log(config.n_samples_val))))
                # Log validation data
                wandb.log({"val_loss": jnp.mean(-log_w) / T, "Log w": wandb.Histogram(np.array(-log_w)), "Mean log-likelihood": log_likelihood, "Adjusted sample size": adjusted_sample_size}, commit=False)
            # wandb logging
            wandb.log({"Train Loss": loss})
            # Update progress bar
            steps.set_postfix(val=loss)

    params = checkpointer.load()
    # Save checkpoint
    # wandb.save(checkpoint_dir_name)

    # Use for sampling
    key, subkey = jax.random.split(key)
    # Generate test set for qualitative evaluation
    x_T = jax.random.normal(key, shape=(config.n_samples_train, 1))
    x_0, log_w = langevin_diffuser.cmcd_diffusion(
        params,
        x_T,
        drift_correction,
        data_model.noisy_score_reversed,
        data_model.log_density,
        lambda x : norm.logpdf(x),
        subkey,
    )

    # Compute densities
    fig, ax = visualise_samples_density([x_T, x_0], [
        wrap_fn(data_model.noisy_density, 1.0), data_model.density, norm.pdf,
    ], ["sample", "target", "standard"])
    wandb.log({"Trained Model Samples": wandb.Image(plt_to_image(fig))})
    # Create final histogram
    wandb.log({"Test Log w": wandb.Histogram(np.array(-log_w))})
    # Finish
    wandb.finish()

if __name__ == '__main__':
    app.run(main)