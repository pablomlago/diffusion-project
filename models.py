import jax
import flax.linen as nn
import numpy as np

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
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x
    
def pos_encoding(t, hidden_dim):
    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    pe = np.zeros((t.shape[0], hidden_dim))
    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    # Generate frequencies based on channel indices, which are only for even indices
    inv_freq = 1.0 / (10000 ** (np.arange(0, hidden_dim, 2) / hidden_dim))
    
    # Compute the sine and cosine values
    pos_enc_sin = np.sin(t * inv_freq)
    pos_enc_cos = np.cos(t * inv_freq)

    pe[:, 0::2] = pos_enc_sin
    pe[:, 1::2] = pos_enc_cos

    return jax.device_put(pe)

