"""Tests for BNN-latent -> curve-params extraction (distribution properties + compilability)."""

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.prior.bnn import build_output_sorted
from icicle_jax.prior.curves import (
    NUM_BNN_OUTPUTS,
    _empirical_quantile,
    extract_curve_params,
    learning_curve,
)

jax.config.update("jax_enable_x64", True)

# A small table built fresh (cheap) -- the calibration table is only a test fixture here.
_TABLE = build_output_sorted(
    jax.random.key(0),
    config_dim=3,
    layers=(2, 4),
    hidden=(8, 16),
    num_networks=64,
    configs_per_network=32,
)


def test_num_outputs_derived_from_layout():
    assert NUM_BNN_OUTPUTS == 23  # 1 + 4 + 4 + 1 + 4 + 4 + 4 + 1


def test_param_distributions():
    bnn_output = jnp.asarray(np.random.default_rng(0).normal(0, 0.3, size=(200, NUM_BNN_OUTPUTS)))
    p = extract_curve_params(bnn_output, jnp.array(0.1), jnp.array(0.9), _TABLE)
    yinf = np.asarray(p.y_inf)
    assert np.all((yinf >= 0.1 - 1e-9) & (yinf <= 0.9 + 1e-9))
    np.testing.assert_allclose(np.asarray(p.weights.sum(0)), 1.0, rtol=1e-6)
    assert np.all(np.asarray(p.alpha) > 0)
    assert np.all(np.asarray(p.precision) > 0)
    assert np.all(np.asarray(p.sat_fidelity) >= 0)
    assert np.all(np.asarray(p.sigma) > 0)


def test_feeds_learning_curve():
    bnn_output = jnp.asarray(np.random.default_rng(1).normal(0, 0.3, size=(1, NUM_BNN_OUTPUTS)))
    p = extract_curve_params(bnn_output, jnp.array(0.1), jnp.array(0.9), _TABLE)
    x = jnp.linspace(0.0, 100.0, 50)
    y = learning_curve(
        x,
        jnp.array(0.1),
        p.y_inf[0],
        p.precision[:, 0],
        p.sat_fidelity[:, 0],
        p.alpha[:, 0],
        p.post_sat_rate[:, 0],
        p.weights[:, 0],
    )
    assert np.all(np.isfinite(np.asarray(y)))
    assert float(y[0]) < float(y[-1])


def test_jit_compiles():
    bnn_output = jnp.asarray(np.random.default_rng(2).normal(0, 0.3, size=(4, NUM_BNN_OUTPUTS)))
    p = jax.jit(extract_curve_params)(bnn_output, jnp.array(0.1), jnp.array(0.9), _TABLE)
    assert p.y_inf.shape == (4,)
    assert np.all(np.isfinite(np.asarray(p.sigma)))


def test_empirical_quantile():
    table = jnp.sort(jnp.asarray(np.random.default_rng(0).normal(size=1000)))
    u = _empirical_quantile(jnp.asarray([-9.0, 0.0, 9.0]), table)
    assert np.all((np.asarray(u) > 0.0) & (np.asarray(u) < 1.0))  # strictly inside (0, 1)
    assert u[0] < u[1] < u[2]  # monotone in the input
    mid = _empirical_quantile(table[500][None], table)[0]
    np.testing.assert_allclose(float(mid), 0.5, atol=0.01)  # a value at the median -> ~0.5
