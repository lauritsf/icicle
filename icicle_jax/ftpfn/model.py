"""Inference for the published FT-PFN checkpoint (Rakotoarison et al., 2024).

The model has token encoders, six post-norm transformer layers and a head with
1000 uniform value buckets on [0, 1]. Tokens are [curve_id, t, hp_1..hp_k],
with y supplied separately. Curve IDs start at 1; 0 denotes a query for an
unobserved configuration. Queries attend only to context.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention_weights
from jax.typing import DTypeLike

from icicle_jax.nn.attention import dot_product_attention

# [0, 1] inputs (t and y) are standardized by the mean/std of U(0, 1).
_MEAN, _STD = 0.5, math.sqrt(1 / 12)


def plain_attention_fn(query, key, value, *, mask=None, bias=None, **_):
    """Adapt the shared 4D attention core to Flax's callback signature."""
    return dot_product_attention(query, key, value, mask, bias=bias)


def precise_attention_fn(query, key, value, *, mask=None, dtype=None, precision=None, **_):
    """High-precision attention for checkpoint parity checks."""
    weights = dot_product_attention_weights(query, key, mask=mask, dtype=dtype, precision=precision)
    return jnp.einsum("...hqk,...khd->...qhd", weights, value, precision=precision)


class EncoderLayer(nnx.Module):
    """One post-norm layer: context-masked attention, then Linear-GELU-Linear."""

    def __init__(
        self,
        emsize: int,
        num_heads: int,
        hidden_dim: int,
        *,
        dtype: DTypeLike | None,
        param_dtype: DTypeLike,
        attention_fn: Callable[..., jax.Array] | None,
        rngs: nnx.Rngs,
    ):
        kw = {"dtype": dtype, "param_dtype": param_dtype, "rngs": rngs}
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=emsize,
            decode=False,
            attention_fn=attention_fn or plain_attention_fn,
            **kw,
        )
        self.linear1 = nnx.Linear(emsize, hidden_dim, **kw)
        self.linear2 = nnx.Linear(hidden_dim, emsize, **kw)
        self.norm1 = nnx.LayerNorm(emsize, epsilon=1e-5, **kw)
        self.norm2 = nnx.LayerNorm(emsize, epsilon=1e-5, **kw)

    def __call__(self, src: jax.Array, single_eval_pos: int | jax.Array) -> jax.Array:
        mask = (jnp.arange(src.shape[1]) < single_eval_pos)[None, None, None, :]
        # Return zero attention for an empty context.
        attn_out = jnp.where(mask.any(-1), self.attn(src, src, mask=mask), 0.0)
        return self._finish(src, attn_out)

    def _finish(self, src, attn_out):
        src = self.norm1(src + attn_out)
        return self.norm2(src + self.linear2(jax.nn.gelu(self.linear1(src), approximate=False)))

    def forward_padded_prefix(
        self, src: jax.Array, single_eval_pos: int | jax.Array, context_width: int
    ) -> jax.Array:
        """Run the same masked layer with a static padded context key/value prefix.

        Positions from ``single_eval_pos`` through ``context_width`` are invisible padding.
        Positions after ``context_width`` are queries. Neither kind can serve as a key under
        the reference mask, so omitting them from the key/value tensors preserves the layer.
        """
        key_value = src[:, :context_width]
        mask = (jnp.arange(context_width) < single_eval_pos)[None, None, None, :]
        attn_out = jnp.where(mask.any(-1), self.attn(src, key_value, mask=mask), 0.0)
        return self._finish(src, attn_out)


class FTPFN(nnx.Module):
    """The FT-PFN forward pass: encode tokens, run the masked stack, decode logits."""

    def __init__(
        self,
        *,
        config_dim: int = 10,
        emsize: int = 512,
        num_heads: int = 4,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_bars: int = 1000,
        max_curves: int = 1000,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        attention_fn: Callable[..., jax.Array] | None = None,
        rngs: nnx.Rngs,
    ):
        self.config_dim = config_dim
        kw = {"dtype": dtype, "param_dtype": param_dtype, "rngs": rngs}
        self.epoch_enc = nnx.Linear(1, emsize, use_bias=False, **kw)
        self.idcurve_enc = nnx.Embed(max_curves + 1, emsize, **kw)
        self.config_enc = nnx.Linear(config_dim, emsize, **kw)
        self.y_enc = nnx.Linear(1, emsize, **kw)
        self.layers = nnx.List(
            [
                EncoderLayer(
                    emsize,
                    num_heads,
                    hidden_dim,
                    attention_fn=attention_fn,
                    **kw,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_linear1 = nnx.Linear(emsize, hidden_dim, **kw)
        self.decoder_linear2 = nnx.Linear(hidden_dim, num_bars, **kw)

    def _encode(self, x: jax.Array, y: jax.Array, single_eval_pos: int | jax.Array) -> jax.Array:
        """Build the shared token representation before the attention stack."""
        hps = x[..., 2:]
        hps = hps * (self.config_dim / hps.shape[-1])
        pad = self.config_dim - hps.shape[-1]
        if pad:
            hps = jnp.concatenate([hps, jnp.zeros((*hps.shape[:-1], pad), hps.dtype)], axis=-1)
        x_src = (
            self.epoch_enc((x[..., 1:2] - _MEAN) / _STD)
            + self.idcurve_enc(x[..., 0].astype(jnp.int32))
            + self.config_enc(hps)
        )
        # y embeddings are added only at context positions with a finite
        # observation; a NaN context y counts as unobserved (x embedding only).
        y = y[..., None]
        y_src = self.y_enc((jnp.nan_to_num(y) - _MEAN) / _STD)
        keep_y = (jnp.arange(x.shape[1]) < single_eval_pos)[None, :, None] & jnp.isfinite(y)
        return x_src + jnp.where(keep_y, y_src, 0.0)

    def _decode(self, src):
        return self.decoder_linear2(jax.nn.gelu(self.decoder_linear1(src), approximate=False))

    def __call__(self, x: jax.Array, y: jax.Array, single_eval_pos: int | jax.Array) -> jax.Array:
        """``x`` (batch, seq, 2 + n_hps), ``y`` (batch, seq) -> logits (batch, seq, num_bars).

        The first ``single_eval_pos`` positions are context; the caller slices
        ``[:, single_eval_pos:]`` for the query predictions. HPs may come at any
        width up to ``config_dim``: they are rescaled by ``config_dim / n_hps``
        and zero-padded (the checkpoint's variable-feature-count convention).
        """
        src = self._encode(x, y, single_eval_pos)
        for layer in self.layers:
            src = layer(src, single_eval_pos)
        return self._decode(src)

    def forward_padded_prefix(
        self,
        x: jax.Array,
        y: jax.Array,
        single_eval_pos: int | jax.Array,
        context_width: int,
    ) -> jax.Array:
        """Forward with a padded context prefix and context-only attention keys."""
        if not 0 < context_width <= x.shape[1]:
            raise ValueError(f"context_width must be in [1, {x.shape[1]}], got {context_width}")
        src = self._encode(x, y, single_eval_pos)
        for layer in self.layers:
            src = layer.forward_padded_prefix(src, single_eval_pos, context_width)
        return self._decode(src)
