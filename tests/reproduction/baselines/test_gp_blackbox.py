"""Black-box GP baseline reference: accounting, cost units, ragged safety, determinism."""

import dataclasses

import numpy as np

from reproduction.bench import BenchmarkTask
from reproduction.baselines.gp_blackbox import N_INIT, blackbox_view, run_gp_blackbox


def make_task(n_cfg=10, b_max=12, seed=0, fid2d=False):
    rng = np.random.default_rng(seed)
    curves = rng.uniform(0.1, 0.9, size=(n_cfg, b_max)).astype(np.float32)
    if fid2d:
        # cumulative walltime-like view: per-config increments, clipped at the budget
        inc = rng.uniform(0.01, 0.08, size=(n_cfg, b_max))
        fid = np.minimum(np.cumsum(inc, axis=1), 1.0).astype(np.float32)
    else:
        fid = ((np.arange(b_max) + 1) / b_max).astype(np.float32)
    return BenchmarkTask("gpbb/test", rng.uniform(size=(n_cfg, 3)).astype(np.float32), curves, fid)


def ragged(task, lengths):
    curves = task.curves.copy()
    fid = task.fidelities.copy()
    for cfg, n in enumerate(lengths):
        curves[cfg, n:] = np.nan
        if fid.ndim == 2:
            fid[cfg, n:] = np.nan
    return dataclasses.replace(task, curves=curves, fidelities=fid)


def test_gp_blackbox_accounting_and_determinism():
    task = make_task()
    res = run_gp_blackbox(task, 8, seed=0)
    assert len(res.chosen) == 8
    # black-box: every evaluation is a distinct config at its full budget and final value
    assert len(set(res.chosen.tolist())) == 8
    np.testing.assert_array_equal(res.budget, 12)
    np.testing.assert_allclose(res.observed, task.curves[res.chosen, -1], atol=1e-7)
    np.testing.assert_array_equal(res.f_best, np.maximum.accumulate(res.observed))
    assert len(set(res.chosen[:N_INIT].tolist())) == N_INIT
    res2 = run_gp_blackbox(task, 8, seed=0)
    np.testing.assert_array_equal(res.chosen, res2.chosen)
    assert (run_gp_blackbox(task, 8, seed=1).chosen[:N_INIT] != res.chosen[:N_INIT]).any()


def test_gp_blackbox_cost_shared_grid_is_one_per_full_curve():
    res = run_gp_blackbox(make_task(), 6, seed=0)
    np.testing.assert_allclose(res.cost, np.arange(1, 7), atol=1e-7)


def test_gp_blackbox_cost_walltime_view():
    task = make_task(fid2d=True)
    final_y, final_budget, eval_cost = blackbox_view(task)
    np.testing.assert_allclose(eval_cost, task.fidelities[:, -1], atol=1e-7)
    res = run_gp_blackbox(task, 6, seed=0)
    np.testing.assert_allclose(res.cost, np.cumsum(eval_cost[res.chosen]), atol=1e-6)


def test_gp_blackbox_ragged_charges_and_observes_the_curve_end():
    lengths = np.array([3, 12, 7, 1, 9, 12, 5, 2, 6, 4])
    task = ragged(make_task(fid2d=True), lengths)
    res = run_gp_blackbox(task, 100, seed=0)
    assert len(res.chosen) == 10  # exhausted pool stops the loop
    np.testing.assert_array_equal(np.sort(res.chosen), np.arange(10))
    np.testing.assert_array_equal(res.budget, lengths[res.chosen])
    assert not np.isnan(res.observed).any() and not np.isnan(res.cost).any()
    np.testing.assert_allclose(
        res.observed, task.curves[res.chosen, lengths[res.chosen] - 1], atol=1e-7
    )
