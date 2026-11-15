"""Pre-norm transformer block with masked attention and a GELU feed-forward layer.

Both sublayers normalize their inputs and add a residual connection. The caller
supplies the key mask and applies the final output normalization: after pooling
in the configuration encoder, or before the surrogate's bar-distribution head.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from icicle_jax.nn.attention import Attention


class EncoderLayer(nnx.Module):
    """One pre-norm block: masked attention + FFN, each with a residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        *,
        qk_norm: bool = False,
        backend: str = "xla",
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        kw = {"dtype": dtype, "param_dtype": param_dtype, "rngs": rngs}
        self.norm1 = nnx.RMSNorm(embed_dim, **kw)
        self.attn = Attention(embed_dim, num_heads, qk_norm=qk_norm, backend=backend, **kw)
        self.norm2 = nnx.RMSNorm(embed_dim, **kw)
        self.w_in = nnx.Linear(embed_dim, hidden_dim, **kw)
        self.w_out = nnx.Linear(hidden_dim, embed_dim, **kw)

    def __call__(self, x: jax.Array, *, key_mask: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x), key_mask=key_mask)
        h = jax.nn.gelu(self.w_in(self.norm2(x)), approximate=False)
        return x + self.w_out(h)
