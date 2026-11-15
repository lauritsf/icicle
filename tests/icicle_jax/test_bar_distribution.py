"""Tests for the bar (Riemann) distribution."""

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.bar_distribution import BarDistribution

jax.config.update("jax_enable_x64", True)


def test_pdf_integrates_to_one():
    rng = np.random.default_rng(0)
    borders = jnp.asarray(np.sort(rng.uniform(-3.0, 4.0, size=21)))
    bd = BarDistribution(borders=borders)
    logits = jnp.asarray(rng.normal(size=(5, bd.num_bars)))

    masses = bd.probs(logits)
    np.testing.assert_allclose(masses.sum(axis=-1), 1.0, rtol=1e-12)

    density = jnp.exp(bd.log_density(logits))
    integral = (density * bd.bucket_widths).sum(axis=-1)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-12)


def test_nll_hand_computed():
    """NLL on a small case worked out by hand.

    Two unequal buckets [0,1] and [1,3] (widths 1 and 2). Equal logits give each
    bucket mass 0.5, so a target in bucket 1 has density 0.5 / 2 = 0.25 and NLL
    -log(0.25).
    """
    bd = BarDistribution(borders=jnp.asarray([0.0, 1.0, 3.0]))
    logits = jnp.asarray([0.0, 0.0])
    nll = bd.nll(logits, jnp.asarray(2.0))
    np.testing.assert_allclose(np.asarray(nll), -np.log(0.25), rtol=1e-12)


def test_nll_ignores_nan_targets():
    """NaN targets contribute 0 NLL and a finite gradient.

    The freeze-thaw prior occasionally emits a non-finite curve value; those tokens
    are discarded. A naive mask after an already-NaN NLL would still backprop
    0 * NaN = NaN, so the target must be sanitised before the gather -- this checks
    both the value and the gradient.
    """
    bd = BarDistribution(borders=jnp.asarray(np.linspace(0.0, 1.0, 1001)))
    logits = jnp.zeros((3, 1000))
    y = jnp.asarray([0.3, np.nan, 0.7])
    nll = bd.nll(logits, y)
    assert bool(jnp.all(jnp.isfinite(nll)))
    np.testing.assert_array_equal(np.asarray(nll)[1], 0.0)
    # The NaN-target position must not invalidate the gradient.
    grad = jax.grad(lambda lg: bd.nll(lg, y).sum())(logits)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_mean():
    borders = jnp.asarray([0.0, 1.0, 3.0])  # midpoints 0.5, 2.0
    bd = BarDistribution(borders=borders)

    np.testing.assert_allclose(np.asarray(bd.mean(jnp.zeros(2))), (0.5 + 2.0) / 2, rtol=1e-12)

    one_hot = jnp.asarray([50.0, -50.0])
    np.testing.assert_allclose(np.asarray(bd.mean(one_hot)), 0.5, rtol=1e-12)

    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(7, bd.num_bars)))
    means = np.asarray(bd.mean(logits))
    assert means.shape == (7,)
    assert (means >= borders[0]).all() and (means <= borders[-1]).all()
