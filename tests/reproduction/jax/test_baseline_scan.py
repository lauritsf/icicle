"""Trajectory parity: the jitted baseline scans against their numpy references.

The scans and NumPy references use identical host-generated permutations.
Decisions (chosen/budget) match exactly; recorded values allow float32 rounding.
"""

import dataclasses

import numpy as np

from reproduction.bench import BenchmarkTask
from reproduction.baselines.scheduler import run_hyperband, run_successive_halving
from reproduction.jax.baseline_scan import run_asha_scan, run_hyperband_scan


def make_task(n_cfg, b_max, seed, name="bl/test"):
    rng = np.random.default_rng(seed)
    curves = rng.uniform(0.05, 0.95, size=(n_cfg, b_max)).astype(np.float32)
    return BenchmarkTask(
        name=name,
        hps=rng.uniform(size=(n_cfg, 3)).astype(np.float32),
        curves=curves,
        fidelities=((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )


def make_ragged_task(n_cfg, b_max, seed):
    task = make_task(n_cfg, b_max, seed, name="bl/ragged")
    rng = np.random.default_rng(seed + 1)
    lengths = rng.integers(1, b_max + 1, size=n_cfg)
    lengths[rng.integers(n_cfg)] = b_max  # at least one full curve
    curves = task.curves.copy()
    for cfg, n in enumerate(lengths):
        curves[cfg, n:] = np.nan
    return dataclasses.replace(task, curves=curves)


def assert_paired(ref, res, label):
    np.testing.assert_array_equal(res.chosen, ref.chosen, err_msg=f"{label}: chosen")
    np.testing.assert_array_equal(res.budget, ref.budget, err_msg=f"{label}: budget")
    np.testing.assert_allclose(res.observed, ref.observed, atol=1e-6, err_msg=label)
    np.testing.assert_allclose(res.f_best, ref.f_best, atol=1e-6, err_msg=label)
    np.testing.assert_allclose(res.regret, ref.regret, atol=1e-6, err_msg=label)


CHAINS = [
    (make_task(17, 27, 0), 0),
    (make_task(17, 27, 0), 1),  # same task, second seed: a different permutation
    (make_task(9, 51, 2), 0),  # mixed b_max in one batch: per-chain ladders
    (make_ragged_task(14, 20, 3), 0),  # NaN suffixes: b_valid guards every read
    (make_task(5, 9, 4), 7),  # tiny table: exhausts well inside n_iters
]


def test_asha_scan_matches_numpy_reference():
    for variant in ("li", "neps"):
        results = run_asha_scan(CHAINS, n_iters=200, variant=variant)
        for (task, seed), res in zip(CHAINS, results, strict=True):
            ref = run_successive_halving(task, 200, seed=seed, variant=variant)
            assert_paired(ref, res, f"asha-{variant}/{task.name}/s{seed}")


def test_hyperband_scan_matches_numpy_reference():
    results = run_hyperband_scan(CHAINS, n_iters=200)
    for (task, seed), res in zip(CHAINS, results, strict=True):
        ref = run_hyperband(task, 200, seed=seed)
        assert_paired(ref, res, f"hb/{task.name}/s{seed}")


def test_pool_exhaustion_parity():
    chains = [(make_task(4, 6, 5), 0), (make_ragged_task(6, 8, 6), 1)]
    for variant in ("li", "neps"):
        for (task, seed), res in zip(
            chains, run_asha_scan(chains, n_iters=400, variant=variant), strict=True
        ):
            ref = run_successive_halving(task, 400, seed=seed, variant=variant)
            assert_paired(ref, res, f"asha-{variant}-exhaust/{task.name}")
    for (task, seed), res in zip(chains, run_hyperband_scan(chains, n_iters=400), strict=True):
        ref = run_hyperband(task, 400, seed=seed)
        assert_paired(ref, res, f"hb-exhaust/{task.name}")
