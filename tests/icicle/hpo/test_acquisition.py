"""Tests for MFPI-random acquisition math (backend-free)."""

import numpy as np

from icicle.hpo import (
    exceedance_prob,
    sample_mfpi_random,
)


def test_exceedance_hand_computed():
    """With buckets [0, 1] and [1, 3], P(y > 2) is half the second bucket’s mass."""
    borders = np.array([0.0, 1.0, 3.0])
    probs = np.array([0.4, 0.6])
    np.testing.assert_allclose(exceedance_prob(probs, borders, 2.0), 0.3)
    np.testing.assert_allclose(exceedance_prob(probs, borders, 1.0), 0.6)  # full upper bucket
    np.testing.assert_allclose(exceedance_prob(probs, borders, -5.0), 1.0)  # below range
    np.testing.assert_allclose(exceedance_prob(probs, borders, 5.0), 0.0)  # above range


def test_exceedance_batched_and_monotone():
    rng = np.random.default_rng(0)
    borders = np.linspace(0.0, 1.0, 101)
    probs = rng.dirichlet(np.ones(100), size=7)
    for t_lo, t_hi in [(0.1, 0.3), (0.5, 0.9)]:
        p_lo, p_hi = (exceedance_prob(probs, borders, t) for t in (t_lo, t_hi))
        assert p_lo.shape == (7,)
        assert np.all(p_lo >= p_hi)  # exceedance is non-increasing in the threshold
    np.testing.assert_allclose(exceedance_prob(probs, borders, 0.0), 1.0)


def test_sample_mfpi_random_ranges():
    rng = np.random.default_rng(1)
    hs, ts = [], []
    for _ in range(500):
        h, t = sample_mfpi_random(rng, b_max=50, f_best=0.8)
        hs.append(h)
        ts.append(t)
    assert min(hs) >= 1 and max(hs) <= 50 and len(set(hs)) > 30
    # T = f_best + tau * (1 - f_best), tau in [1e-4, 1e-1]
    assert min(ts) >= 0.8 + 1e-4 * 0.2 and max(ts) <= 0.8 + 0.1 * 0.2
