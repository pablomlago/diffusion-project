from io import BytesIO
from PIL import Image

import jax.numpy as jnp
from typing import Callable, List
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

def plt_to_image(fig):
    # Save the plot to a buffer
    
    image_data = BytesIO() #Create empty in-memory file
    fig.savefig(image_data, format='png') #Save pyplot figure to in-memory file
    image_data.seek(0) #Move stream position back to beginning of file 

    return Image.open(image_data)
