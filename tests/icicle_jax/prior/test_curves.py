"""Tests for the parametric learning-curve family (parity + shape)."""

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.prior.curves import learning_curve

jax.config.update("jax_enable_x64", True)

# a non-degenerate parameter set (log basis needs alpha != 1)
_PARAMS = {
    "y0": jnp.array(0.1),
    "y_inf": jnp.array(0.9),
    "precision": jnp.array([2.0, 3.0, 2.5, 4.0]),
    "sat_fidelity": jnp.array([10.0, 8.0, 12.0, 9.0]),
    "alpha": jnp.array([1.5, 2.0, 1.5, 1.2]),
    "post_sat_rate": jnp.array([0.5, 0.6, 0.4, 0.7]),
    "weights": jnp.array([0.25, 0.25, 0.25, 0.25]),
}


def test_parity_with_reference():
    x = jnp.array([0.0, 1.0, 5.0, 20.0, 50.0])
    golden = np.array(
        [0.100000000000, 0.183698183310, 0.438204185164, 0.719421222755, 0.789965780714]
    )
    np.testing.assert_allclose(np.asarray(learning_curve(x, **_PARAMS)), golden, rtol=1e-10)


def test_starts_at_y0_approaches_yinf():
    np.testing.assert_allclose(float(learning_curve(jnp.array(0.0), **_PARAMS)), 0.1, atol=1e-9)
    ybig = float(learning_curve(jnp.array(1e5), **_PARAMS))
    assert 0.85 < ybig <= 0.9 + 1e-9


def test_monotone_increasing():
    y = np.asarray(learning_curve(jnp.linspace(0.0, 100.0, 200), **_PARAMS))
    assert np.all(np.diff(y) >= -1e-9)
