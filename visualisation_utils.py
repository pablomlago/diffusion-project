from io import BytesIO
from PIL import Image

import jax.numpy as jnp
from typing import Callable, List, Optional
import itertools
import matplotlib.pyplot as plt
import numpy as np

# Some colors to use in plots
colours = [
    "red", "blue", "green", "yellow", "cyan", "magenta", "white",
    "gray", "orange", "purple", "brown", "pink", "lime", "navy", "teal",
    "coral", "olive", "maroon", "mint", "apricot", "beige", "lavender", 
    "turquoise", "azure"
]

def visualise_samples_density(xs: List[np.array], density_fns: List[Callable], labels: List[str], n_bins : int = 50, xlim: float = 4.):
    # Instantiate figure
    fig, ax = plt.subplots()

    # Values in which to evaluate the density
    x_values = jnp.linspace(-jnp.abs(xlim), jnp.abs(xlim), 1000)

    # Show samples
    for i, x in enumerate(xs):
        x = jnp.squeeze(x)
        ax.hist(np.array(x), density=True, bins=n_bins, color=colours[i], alpha=0.5)
    
    # Show densities
    for i, density_fn in enumerate(density_fns):
        density_x_values = jnp.squeeze(density_fn(jnp.expand_dims(x_values, axis=-1)))
        ax.plot(np.array(x_values), np.array(density_x_values), label=labels[i], color=colours[i])

    ax.legend()

    return fig, ax

def visualise_densities_1d(density_fns: List[Callable], labels: List[str], xlim: float = 4.):
    # Instantiate figure
    fig, ax = plt.subplots()

    # Values in which to evaluate the density
    x_values = jnp.linspace(-jnp.abs(xlim), jnp.abs(xlim), 1000)

    # Show densities
    for i, density_fn in enumerate(density_fns):
        density_x_values = jnp.squeeze(density_fn(jnp.expand_dims(x_values, axis=-1)))
        ax.plot(np.array(x_values), np.array(density_x_values), label=labels[i], color=colours[i])

    ax.legend()

    return fig, ax

# Taken from https://github.com/lollcat/fab-jax/blob/632e0a7d3dbd8da6b2ef043ab41e2346f29dfece/fabjax/utils/plot.py#L30
def plot_marginal_pair(
    samples, ax=None, marginal_dims=(0, 1), bounds=(-5, 5), alpha: float = 0.5
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(
        samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha
    )

# Taken from https://github.com/lollcat/fab-jax/blob/632e0a7d3dbd8da6b2ef043ab41e2346f29dfece/fabjax/utils/plot.py#L11
def plot_contours_2D(
    log_prob_func, ax: Optional[plt.Axes] = None, bound: int = 3, levels: int = 20
):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contour(x1, x2, z, levels=levels)


def plot_gmm(samples, log_p_fn, loc_scaling):
    plot_bound = loc_scaling * 1.5
    fig, axs = plt.subplots(1, figsize=(5, 5))
    plot_contours_2D(log_p_fn, axs, bound=plot_bound, levels=50)
    plot_marginal_pair(samples, axs, bounds=(-plot_bound, plot_bound))
    axs.set_title("samples")
    return fig, axs

def plt_to_image(fig):
    # Save the plot to a buffer
    
    image_data = BytesIO() #Create empty in-memory file
    fig.savefig(image_data, format='png') #Save pyplot figure to in-memory file
    image_data.seek(0) #Move stream position back to beginning of file 

    return Image.open(image_data)
