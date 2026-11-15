"""Tests for the config->curve-params BNN (behavioural -- the RNG is ours, so no bit-parity)."""

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.prior.bnn import bnn_forward, sample_bnn

jax.config.update("jax_enable_x64", True)


def _bnn(seed=0, num_hidden=5, num_layers=3, max_hidden=8, max_layers=4):
    return sample_bnn(
        jax.random.key(seed),
        num_inputs=3,
        num_outputs=23,
        num_hidden=num_hidden,
        num_layers=num_layers,
        max_hidden=max_hidden,
        max_layers=max_layers,
    )


_X = jnp.reshape(jnp.arange(12.0) / 12.0, (4, 3))


def test_shapes():
    bnn = _bnn(max_hidden=8, max_layers=4)
    assert bnn.w_in.shape == (8, 3)
    assert bnn.w_hidden.shape == (2, 8, 8)  # max_layers - 2
    assert bnn.w_out.shape == (23, 8)
    out = bnn_forward(bnn, _X, jax.random.key(1))
    assert out.shape == (4, 23)
    assert np.all(np.isfinite(np.asarray(out)))


def test_width_masking():
    bnn = _bnn(num_hidden=5, max_hidden=8)
    assert np.allclose(np.asarray(bnn.w_in[5:]), 0.0)
    assert np.allclose(np.asarray(bnn.w_out[:, 5:]), 0.0)
    assert not np.allclose(np.asarray(bnn.w_in[:5]), 0.0)


def test_sparseness_drops_hidden_fraction():
    bnn = sample_bnn(
        jax.random.key(3),
        num_inputs=3,
        num_outputs=23,
        num_hidden=64,
        num_layers=4,
        max_hidden=64,
        max_layers=4,
    )
    frac_zero = float(np.mean(np.asarray(bnn.w_hidden) == 0.0))
    assert 0.11 < frac_zero < 0.18  # ~0.145 over 2*64*64 weights


def test_depth_controls_active_layers():
    shallow = _bnn(num_layers=2, max_layers=4)
    deeper = shallow._replace(num_layers=jnp.asarray(4))
    out_shallow = bnn_forward(shallow, _X, jax.random.key(1))
    out_deeper = bnn_forward(deeper, _X, jax.random.key(1))
    assert not np.allclose(np.asarray(out_shallow), np.asarray(out_deeper))
    # with no hidden step running, perturbing the hidden weights leaves the output unchanged
    perturbed = shallow._replace(w_hidden=shallow.w_hidden + 1.0)
    np.testing.assert_allclose(
        np.asarray(bnn_forward(perturbed, _X, jax.random.key(1))),
        np.asarray(out_shallow),
        atol=1e-12,
    )
