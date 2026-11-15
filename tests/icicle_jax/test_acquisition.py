"""The device scorer must be the numpy scorer -- it decides which config gets the next step."""

import jax
import jax.numpy as jnp
import numpy as np

from icicle.hpo import decider as ref
from icicle_jax import acquisition as dev

jax.config.update("jax_enable_x64", True)

BORDERS = np.linspace(0.0, 1.0, 65)


def random_probs(rng, b, q, bars=64):
    p = rng.uniform(0.01, 1.0, size=(b, q, bars))
    return p / p.sum(-1, keepdims=True)


def test_device_scorer_matches_the_reference():
    rng = np.random.default_rng(0)
    probs = random_probs(rng, 5, 7)
    params = rng.uniform(-0.2, 1.2, size=5)

    got = np.asarray(dev.exceedance_prob( jnp.asarray(probs), jnp.asarray(BORDERS), jnp.asarray(params)))
    want = np.stack([ref.exceedance_prob(probs[i], BORDERS, params[i]) for i in range(5)])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_degenerate_thresholds_and_masses():
    probs = np.zeros((3, 2, 64))
    probs[0, :, 0] = 1.0  # all mass in the first bucket
    probs[1, :, -1] = 1.0  # all mass in the last
    probs[2, :, 30] = 1.0
    params = np.array([BORDERS[0], BORDERS[-1], BORDERS[31]])  # exactly on borders
    got = np.asarray(dev.exceedance_prob( jnp.asarray(probs), jnp.asarray(BORDERS), jnp.asarray(params)))
    want = np.stack([ref.exceedance_prob(probs[i], BORDERS, params[i]) for i in range(3)])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
