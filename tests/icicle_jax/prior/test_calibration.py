"""Tests for the BNN-output calibration table (built fresh from our own BNN population)."""

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.prior.bnn import bnn_forward, build_output_sorted, sample_bnn, standardize_uniform
from icicle_jax.prior.curves import NUM_BNN_OUTPUTS, extract_curve_params

jax.config.update("jax_enable_x64", True)


def test_build_output_sorted():
    table = build_output_sorted(
        jax.random.key(0),
        config_dim=3,
        layers=(2, 4),
        hidden=(8, 16),
        num_networks=64,
        configs_per_network=32,
    )
    a = np.asarray(table)
    assert a.ndim == 1 and a.size == 64 * 32 * NUM_BNN_OUTPUTS
    assert np.all(np.diff(a) >= 0)
    assert a.min() < 0.0 < a.max()


def test_built_table_calibrates_extract():
    table = build_output_sorted(
        jax.random.key(1),
        config_dim=3,
        layers=(2, 4),
        hidden=(8, 16),
        num_networks=128,
        configs_per_network=32,
    )
    bnn = sample_bnn(
        jax.random.key(2),
        num_inputs=3,
        num_outputs=NUM_BNN_OUTPUTS,
        num_hidden=12,
        num_layers=3,
        max_hidden=16,
        max_layers=4,
    )
    configs = jax.random.uniform(jax.random.key(3), (300, 3))
    out = bnn_forward(bnn, standardize_uniform(configs), jax.random.key(4))
    p = extract_curve_params(out, jnp.array(0.1), jnp.array(0.9), table)
    yinf = np.asarray(p.y_inf)
    assert np.all((yinf >= 0.1 - 1e-9) & (yinf <= 0.9 + 1e-9))
    np.testing.assert_allclose(np.asarray(p.weights.sum(0)), 1.0, rtol=1e-6)
