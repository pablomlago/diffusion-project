import jax.numpy as jnp
import jax
import numpy as np
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

# Generate RandomKey
key = jax.random.PRNGKey(0)

x_0 = 2. + 0.1*jax.random.normal(key, shape=(10_000, 1))
# Compute densities
x_values = np.linspace(-4, 4, 1000)
density_x_0 = norm.pdf(x_values, loc=2., scale=0.1)

plt.hist(np.array(x_0), density=True, bins=50, label="Final Samples", color='blue')
plt.plot(x_values, np.array(density_x_0), label='Sample Density', color='green')
plt.legend()
plt.show()