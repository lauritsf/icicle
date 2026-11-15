"""Trajectory parity: the jitted freeze-thaw GP scan against its numpy reference.

The scan uses float32 and the reference uses float64. These tests use separated
configurations and monotone curves to keep acquisition scores distinguishable
under float32 rounding.
"""

import dataclasses

import numpy as np

from reproduction.bench import BenchmarkTask
from reproduction.baselines.gp_ft import run_gp_freeze_thaw
from reproduction.jax.gpft_scan import run_gpft_scan


def make_task(n_cfg, b_max, seed, name="gpft/t"):
    """Separated configs: hps spread over a coarse lattice, saturating curves with
    distinct ceilings and rates -- every candidate pair is far apart in EI terms.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n_cfg)
    hps = np.stack(
        [2.0 * i / n_cfg, ((i * 7) % n_cfg) / n_cfg, ((i * 3) % n_cfg) / n_cfg], axis=1
    ).astype(np.float32)
    ceil = np.linspace(0.3, 0.95, n_cfg)[rng.permutation(n_cfg)]
    rate = np.linspace(2.0, 9.0, n_cfg)[rng.permutation(n_cfg)]
    fid = (np.arange(b_max) + 1) / b_max
    curves = ceil[:, None] * (1.0 - np.exp(-rate[:, None] * fid[None, :]))
    return BenchmarkTask(name, hps, curves.astype(np.float32), fid.astype(np.float32))


def make_ragged(n_cfg, b_max, seed):
    task = make_task(n_cfg, b_max, seed, name="gpft/ragged")
    rng = np.random.default_rng(seed + 1)
    lengths = rng.integers(1, b_max + 1, size=n_cfg)
    lengths[rng.integers(n_cfg)] = b_max
    curves = task.curves.copy()
    for cfg, n in enumerate(lengths):
        curves[cfg, n:] = np.nan
    return dataclasses.replace(task, curves=curves)


def test_gpft_scan_matches_numpy_reference():
    chains = [
        (make_task(12, 15, 0), 0),
        (make_task(12, 15, 0), 3),  # second seed: different init draw
        (make_task(7, 25, 2), 0),  # mixed b_max: per-chain level normalization
        (make_ragged(9, 10, 4), 1),  # NaN suffixes
    ]
    # Exact order parity is pinned on an early window where log-EI gaps are far above the
    # fp32 ceiling; past it, near-exhausted candidates can swap adjacent picks (observed:
    # single 2<->3 swaps that re-converge) -- the full run must still do the same work.
    n_pin = 30
    results = run_gpft_scan(chains, n_iters=40)
    for (task, seed), res in zip(chains, results, strict=True):
        ref = run_gp_freeze_thaw(task, 40, seed=seed)
        label = f"gpft/{task.name}/s{seed}"
        np.testing.assert_array_equal(res.chosen[:n_pin], ref.chosen[:n_pin], err_msg=label)
        np.testing.assert_array_equal(res.budget[:n_pin], ref.budget[:n_pin], err_msg=label)
        np.testing.assert_allclose(res.observed[:n_pin], ref.observed[:n_pin], atol=1e-6)
        np.testing.assert_allclose(res.f_best[:n_pin], ref.f_best[:n_pin], atol=1e-6)
        np.testing.assert_allclose(res.regret[:n_pin], ref.regret[:n_pin], atol=1e-6)
        assert sorted(zip(res.chosen.tolist(), res.budget.tolist(), strict=True)) == sorted(
            zip(ref.chosen.tolist(), ref.budget.tolist(), strict=True)
        ), label
        np.testing.assert_allclose(res.f_best[-1], ref.f_best[-1], atol=1e-6, err_msg=label)


def test_gpft_scan_pool_exhaustion():
    chains = [(make_ragged(5, 6, 7), 0)]
    (res,) = run_gpft_scan(chains, n_iters=60)
    ref = run_gp_freeze_thaw(chains[0][0], 60, seed=0)
    np.testing.assert_array_equal(res.chosen, ref.chosen)
    assert len(res.chosen) < 60  # table exhausted well inside the budget


def test_tiny_pools_and_initialization_only():
    from reproduction.jax.gpblackbox_scan import run_gpblackbox_scan

    for runner in (run_gpft_scan, run_gpblackbox_scan):
        for size in (1, 2, 3):
            task = make_task(size, 1, 0)
            for iterations in (1, size, size + 2):
                result, = runner([(task, 0)], n_iters=iterations)
                assert len(result.chosen) == min(size, iterations)
                assert len(np.unique(result.chosen)) == len(result.chosen)
                np.testing.assert_allclose(result.observed, task.curves[result.chosen, 0])
