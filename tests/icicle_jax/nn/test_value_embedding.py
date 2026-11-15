"""Tests for the sincos value embedding."""

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.nn.value_embedding import sincos

jax.config.update("jax_enable_x64", True)


def test_sincos_shape():
    val = jnp.asarray(np.random.default_rng(0).uniform(0, 1, size=(2, 5, 3)))
    assert sincos(val, 16).shape == (2, 5, 3, 16)


def test_sincos_absolute_and_bounded():
    pe = sincos(jnp.asarray([0.0, 0.2, 0.7]), 16)
    assert float(jnp.max(jnp.abs(pe))) <= 1.0 + 1e-12
    # v=0 -> [sin(0)..., cos(0)...] = [0...0, 1...1]
    np.testing.assert_allclose(np.asarray(pe[0, :8]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pe[0, 8:]), 1.0, atol=1e-12)
    assert not np.allclose(np.asarray(pe[1]), np.asarray(pe[2]))


def test_sincos_constant_norm():
    """Each sin/cos pair contributes one to squared norm, regardless of the input value."""
    vals = jnp.asarray(np.random.default_rng(3).uniform(0, 1, size=(50,)))
    for dim in (16, 64):
        norms = jnp.linalg.norm(sincos(vals, dim), axis=-1)
        np.testing.assert_allclose(np.asarray(norms), np.sqrt(dim / 2), rtol=1e-12)
