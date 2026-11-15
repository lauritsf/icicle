"""Freeze-thaw GP baseline reference: accounting, ragged safety, determinism."""

import dataclasses

import numpy as np

from reproduction.bench import BenchmarkTask
from reproduction.baselines.gp_ft import N_INIT, run_gp_freeze_thaw


def make_task(n_cfg=10, b_max=12, seed=0):
    rng = np.random.default_rng(seed)
    curves = rng.uniform(0.1, 0.9, size=(n_cfg, b_max)).astype(np.float32)
    return BenchmarkTask(
        "gpft/test",
        rng.uniform(size=(n_cfg, 3)).astype(np.float32),
        curves,
        ((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )


def test_gp_ft_accounting_and_determinism():
    task = make_task()
    res = run_gp_freeze_thaw(task, 40, seed=0)
    assert len(res.chosen) == 40
    counts = np.bincount(res.chosen, minlength=10)
    assert counts.max() <= 12  # never trains past the table
    # the init is 3 distinct configs at one step each
    assert len(set(res.chosen[:N_INIT].tolist())) == N_INIT
    np.testing.assert_array_equal(res.budget[:N_INIT], 1)
    res2 = run_gp_freeze_thaw(task, 40, seed=0)
    np.testing.assert_array_equal(res.chosen, res2.chosen)
    assert (run_gp_freeze_thaw(task, 40, seed=1).chosen[:N_INIT] != res.chosen[:N_INIT]).any()


def test_gp_ft_ragged_never_observes_a_nan():
    task = make_task()
    lengths = np.array([3, 12, 7, 1, 9, 12, 5, 2, 6, 4])
    curves = task.curves.copy()
    for cfg, n in enumerate(lengths):
        curves[cfg, n:] = np.nan
    task = dataclasses.replace(task, curves=curves)
    res = run_gp_freeze_thaw(task, 100, seed=0)
    assert len(res.chosen) <= int(lengths.sum())  # exhausted pool stops the loop
    assert not np.isnan(res.observed).any()
    for cfg in np.unique(res.chosen):
        assert res.budget[res.chosen == cfg].max() <= lengths[cfg]
