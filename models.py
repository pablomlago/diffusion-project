import jax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

class MLP(nn.Module):
    """ A simple MLP in Flax. This is the noise-prediction or score function.
    """
    hidden_dim: int = 32
    out_dim: int = 1
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.softplus(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x

class MLP_Pos(nn.Module):
    """ A simple MLP in Flax. This is the noise-prediction or score function.
    """
    hidden_dim: int = 32
    out_dim: int = 1
    n_layers: int = 2

    @nn.compact
    def __call__(self, x, t):
        x = nn.Dense(features=self.hidden_dim)(x)
        pos = pos_encoding(t, self.hidden_dim)
        x = x + pos
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x
    
def pos_encoding(t, hidden_dim):
    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    pe = jnp.zeros((t.shape[0], hidden_dim))
    
    # Generate frequencies based on channel indices, which are only for even indices
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, hidden_dim, 2) / hidden_dim))
    
    # Compute the sine and cosine values
    pos_enc_sin = jnp.sin(t * inv_freq[jnp.newaxis, :])
    pos_enc_cos = jnp.cos(t * inv_freq[jnp.newaxis, :])

    # Assign the sine values to the even indices and the cosine values to the odd indices
    pe = pe.at[:, 0::2].set(pos_enc_sin)
    pe = pe.at[:, 1::2].set(pos_enc_cos)

    return pe

