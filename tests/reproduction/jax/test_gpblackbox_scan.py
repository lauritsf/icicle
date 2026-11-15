"""Trajectory parity: the jitted black-box GP scan against its numpy reference.

The scan runs its GP in float32 against the reference's float64. Decisions match while
log-EI gaps exceed the fp32 ceiling, so the tasks here keep configs separated (spread hps,
distinct final values) -- the same convention as the gp_ft parity test (module docstring
of ``reproduction.jax.gpblackbox_scan``).
"""

import dataclasses

import numpy as np

from reproduction.bench import BenchmarkTask
from reproduction.baselines.gp_blackbox import run_gp_blackbox
from reproduction.jax.gpblackbox_scan import run_gpblackbox_scan


def make_task(n_cfg, b_max, seed, name="gpbb/t", fid2d=False):
    """Separated configs: hps spread over a coarse lattice, saturating curves with distinct
    ceilings and rates -- every candidate pair is far apart in EI terms.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n_cfg)
    hps = np.stack(
        [2.0 * i / n_cfg, ((i * 7) % n_cfg) / n_cfg, ((i * 3) % n_cfg) / n_cfg], axis=1
    ).astype(np.float32)
    ceil = np.linspace(0.3, 0.95, n_cfg)[rng.permutation(n_cfg)]
    rate = np.linspace(2.0, 9.0, n_cfg)[rng.permutation(n_cfg)]
    grid = (np.arange(b_max) + 1) / b_max
    curves = ceil[:, None] * (1.0 - np.exp(-rate[:, None] * grid[None, :]))
    if fid2d:
        inc = rng.uniform(0.01, 0.08, size=(n_cfg, b_max))
        fid = np.minimum(np.cumsum(inc, axis=1), 1.0).astype(np.float32)
    else:
        fid = grid.astype(np.float32)
    return BenchmarkTask(name, hps, curves.astype(np.float32), fid)


def make_ragged(n_cfg, b_max, seed):
    task = make_task(n_cfg, b_max, seed, name="gpbb/ragged", fid2d=True)
    rng = np.random.default_rng(seed + 1)
    lengths = rng.integers(1, b_max + 1, size=n_cfg)
    lengths[rng.integers(n_cfg)] = b_max
    curves, fid = task.curves.copy(), task.fidelities.copy()
    for cfg, n in enumerate(lengths):
        curves[cfg, n:] = np.nan
        fid[cfg, n:] = np.nan
    return dataclasses.replace(task, curves=curves, fidelities=fid)


def test_gpblackbox_scan_matches_numpy_reference():
    chains = [
        (make_task(14, 15, 0), 0),
        (make_task(14, 15, 0), 3),  # second seed: different init draw
        (make_task(12, 25, 2, fid2d=True), 0),  # walltime-view cost units
        (make_ragged(12, 10, 4), 1),  # NaN suffixes: final value at the ragged end
    ]
    # Exact order parity is pinned on an early window where log-EI gaps are far above the
    # fp32 ceiling; past it, the near-exhausted tail can swap adjacent picks (observed:
    # single swaps that re-converge) -- the full run must still evaluate the same configs.
    n_pin = 7
    results = run_gpblackbox_scan(chains, n_iters=10)
    for (task, seed), res in zip(chains, results, strict=True):
        ref = run_gp_blackbox(task, 10, seed=seed)
        label = f"gpbb/{task.name}/s{seed}"
        np.testing.assert_array_equal(res.chosen[:n_pin], ref.chosen[:n_pin], err_msg=label)
        np.testing.assert_array_equal(res.budget[:n_pin], ref.budget[:n_pin], err_msg=label)
        np.testing.assert_allclose(res.observed[:n_pin], ref.observed[:n_pin], atol=1e-6)
        np.testing.assert_allclose(res.f_best[:n_pin], ref.f_best[:n_pin], atol=1e-6)
        np.testing.assert_allclose(res.regret[:n_pin], ref.regret[:n_pin], atol=1e-6)
        np.testing.assert_allclose(res.cost[:n_pin], ref.cost[:n_pin], atol=1e-5)
        assert sorted(res.chosen.tolist()) == sorted(ref.chosen.tolist()), label
        np.testing.assert_allclose(res.f_best[-1], ref.f_best[-1], atol=1e-6, err_msg=label)
        np.testing.assert_allclose(res.cost[-1], ref.cost[-1], atol=1e-5, err_msg=label)


def test_gpblackbox_scan_pool_exhaustion():
    chains = [(make_ragged(5, 6, 7), 0)]
    (res,) = run_gpblackbox_scan(chains, n_iters=60)
    ref = run_gp_blackbox(chains[0][0], 60, seed=0)
    np.testing.assert_array_equal(res.chosen, ref.chosen)
    assert len(res.chosen) == 5  # every config exactly once, then the pool is exhausted
    np.testing.assert_array_equal(np.sort(res.chosen), np.arange(5))
