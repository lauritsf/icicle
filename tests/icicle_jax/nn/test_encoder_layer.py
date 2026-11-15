"""Tests for the pre-norm encoder block."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from icicle_jax.nn.attention import incontext_key_mask
from icicle_jax.nn.encoder_layer import EncoderLayer

jax.config.update("jax_enable_x64", True)


def _layer(**kw):
    return EncoderLayer(
        16, num_heads=4, hidden_dim=32, param_dtype=jnp.float64, rngs=nnx.Rngs(0), **kw
    )


def test_block_no_query_leakage():
    rng = np.random.default_rng(0)
    seq, sep = 12, 7
    layer = _layer()
    km = incontext_key_mask(sep, seq)
    x = jnp.asarray(rng.normal(size=(2, seq, 16)))
    out = layer(x, key_mask=km)

    j = sep + 2  # a query position
    x2 = x.at[:, j].set(jnp.asarray(rng.normal(size=(2, 16))))
    out2 = layer(x2, key_mask=km)

    keep = jnp.arange(seq) != j
    np.testing.assert_allclose(np.asarray(out2[:, keep]), np.asarray(out[:, keep]), atol=1e-10)
    assert not np.allclose(np.asarray(out2[:, j]), np.asarray(out[:, j]))


def test_block_context_permutation_invariance():
    rng = np.random.default_rng(1)
    seq, sep = 12, 7
    layer = _layer()
    km = incontext_key_mask(sep, seq)
    x = jnp.asarray(rng.normal(size=(2, seq, 16)))
    out = layer(x, key_mask=km)

    perm = rng.permutation(sep)
    xp = x.at[:, :sep].set(x[:, :sep][:, perm])
    outp = layer(xp, key_mask=km)

    # atol 1e-6: the fp32 softmax core gives ~1e-7 reduction-order wiggle between the permuted
    # and unpermuted programs (see dot_product_attention); a leakage bug is orders above it.
    np.testing.assert_allclose(
        np.asarray(outp[:, sep:]), np.asarray(out[:, sep:]), rtol=1e-5, atol=1e-6
    )


def test_block_qk_norm_runs():
    rng = np.random.default_rng(2)
    layer = _layer(qk_norm=True)
    km = incontext_key_mask(7, 12)
    x = jnp.asarray(rng.normal(size=(2, 12, 16)))
    out = layer(x, key_mask=km)
    assert out.shape == (2, 12, 16)
    assert np.all(np.isfinite(np.asarray(out)))
