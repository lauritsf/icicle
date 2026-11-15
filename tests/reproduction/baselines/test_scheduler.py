"""Freeze-thaw scheduler tests with oracle/stub predictors (no surrogate needed)."""

import dataclasses

import numpy as np

from reproduction.bench import BenchmarkTask
from reproduction.baselines.scheduler import (
    run_freeze_thaw,
    run_hyperband,
    run_random_search,
    run_successive_halving,
    rung_ladder,
)

BORDERS = np.linspace(0.0, 1.0, 101)


def make_task(n_cfg=8, b_max=20, seed=0):
    """Monotone saturating curves; config quality increases with row index (row 7 best)."""
    rng = np.random.default_rng(seed)
    ceil = np.linspace(0.3, 0.95, n_cfg)
    rate = rng.uniform(3.0, 8.0, n_cfg)
    fid = (np.arange(b_max) + 1) / b_max
    curves = ceil[:, None] * (1.0 - np.exp(-rate[:, None] * fid[None, :]))
    return BenchmarkTask(
        name="hpo/test",
        hps=rng.uniform(size=(n_cfg, 2)).astype(np.float32),
        curves=curves.astype(np.float32),
        fidelities=fid.astype(np.float32),
    )


def oracle_predict(task):
    """Peeks at the true table: a near-delta bucket distribution on the true value."""

    def predict(ctx_cfg, ctx_lvl, ctx_y, qry_cfg, qry_lvl):
        probs = np.zeros((len(qry_cfg), len(BORDERS) - 1))
        idx = np.clip(
            np.searchsorted(BORDERS, task.curves[qry_cfg, qry_lvl], side="right") - 1,
            0,
            probs.shape[1] - 1,
        )
        probs[np.arange(len(qry_cfg)), idx] = 1.0
        return probs

    return predict


def uniform_predict(ctx_cfg, ctx_lvl, ctx_y, qry_cfg, qry_lvl):
    return np.full((len(qry_cfg), len(BORDERS) - 1), 1.0 / (len(BORDERS) - 1))


def test_oracle_finds_best_config_fast():
    task = make_task()
    res = run_freeze_thaw(task, oracle_predict(task), BORDERS, n_iters=60, seed=0)
    assert res.regret[-1] < 0.02
    best_cfg = int(np.argmax(task.curves[:, -1]))
    # the best config received the most training steps
    counts = np.bincount(res.chosen, minlength=task.curves.shape[0])
    assert counts.argmax() == best_cfg
    # regret is monotonically non-increasing by construction of f_best
    assert np.all(np.diff(res.regret) <= 1e-12)


def test_budget_accounting_and_reproducibility():
    task = make_task()
    res = run_freeze_thaw(task, uniform_predict, BORDERS, n_iters=40, seed=3)
    # one step per iteration, budgets never exceed b_max, prefixes consistent
    assert len(res.chosen) == 40
    for cfg in np.unique(res.chosen):
        budgets = res.budget[res.chosen == cfg]
        assert np.all(np.diff(budgets) == 1)  # each advance is exactly one step
        assert budgets[0] == 1 and budgets[-1] <= task.curves.shape[1]
    res2 = run_freeze_thaw(task, uniform_predict, BORDERS, n_iters=40, seed=3)
    np.testing.assert_array_equal(res.chosen, res2.chosen)


def test_random_search_baseline_covers_and_matches_currency():
    task = make_task()
    n = 3 * task.curves.shape[1]  # three full trainings' worth of steps
    res = run_random_search(task, n_iters=n, seed=1)
    assert len(res.chosen) == n
    assert len(np.unique(res.chosen)) == 3  # one config at a time, trained to completion
    assert np.all(np.diff(res.regret) <= 1e-12)

    assert res.cost is not None
    assert np.isfinite(res.cost).all() and np.all(np.diff(res.cost) >= 0)


def make_ragged_task(n_cfg=8, b_max=20, seed=0):
    """Meta-Album's shape: each config's curve ends early, the rest of the row is NaN."""
    task = make_task(n_cfg, b_max, seed)
    lengths = np.array([3, 20, 7, 1, 12, 20, 5, 9])[:n_cfg]
    curves = task.curves.copy()
    for cfg, n in enumerate(lengths):
        curves[cfg, n:] = np.nan
    return dataclasses.replace(task, curves=curves), lengths


def test_ragged_curves_never_observe_a_nan():
    task, lengths = make_ragged_task()
    y_star = np.nanmax(task.curves)
    assert np.isnan(task.curves.max())  # the plain max the loops used to take

    runs = {
        "freeze_thaw": run_freeze_thaw(task, oracle_predict(task), BORDERS, n_iters=90, seed=0),
        "random_search": run_random_search(task, n_iters=90, seed=0),
        "successive_halving": run_successive_halving(task, 90, seed=0),
        "asha_neps": run_successive_halving(task, 90, seed=0, variant="neps"),
        "hyperband": run_hyperband(task, 90, seed=0),
    }
    for name, res in runs.items():
        assert not np.isnan(res.observed).any(), name
        assert not np.isnan(res.regret).any(), name
        for cfg in np.unique(res.chosen):
            assert res.budget[res.chosen == cfg].max() <= lengths[cfg], f"{name}/{cfg}"
        np.testing.assert_allclose(res.regret, y_star - res.f_best, atol=1e-6)

    # Stop when the finite pool is exhausted. Random search continues cycling its
    # permutation and re-observing configurations.
    for name in ("freeze_thaw", "successive_halving", "asha_neps", "hyperband"):
        assert len(runs[name].chosen) <= int(lengths.sum()), name

    # under a budget it can't exhaust, the oracle still trains the reachable best to its end
    short = run_freeze_thaw(task, oracle_predict(task), BORDERS, n_iters=30, seed=0)
    reachable_best = int(np.nanargmax(np.nanmax(task.curves, axis=1)))
    assert short.budget[short.chosen == reachable_best].max() == lengths[reachable_best]
    assert short.regret[-1] < 1e-6  # that config holds the table optimum


def test_ragged_random_search_walks_whole_prefixes():
    task, lengths = make_ragged_task()
    n = int(lengths[:3].sum())  # first three configs of the seeded permutation... at least 3 ends
    res = run_random_search(task, n_iters=n, seed=1)
    # every config visited except possibly the last is trained to exactly its valid length
    visited = list(dict.fromkeys(res.chosen.tolist()))
    for cfg in visited[:-1]:
        budgets = res.budget[res.chosen == cfg]
        np.testing.assert_array_equal(budgets, np.arange(1, lengths[cfg] + 1))


def test_dense_task_is_bit_identical_to_the_pre_nan_path():
    task = make_task()
    b_max = task.curves.shape[1]
    ft = run_freeze_thaw(task, oracle_predict(task), BORDERS, n_iters=60, seed=0)
    rs = run_random_search(task, n_iters=3 * b_max, seed=1)
    # the old random-search indexing, written out
    order = np.random.default_rng([1, 424242]).permutation(len(task.curves))
    expect_cfg = np.array([order[(i // b_max) % len(order)] for i in range(3 * b_max)])
    np.testing.assert_array_equal(rs.chosen, expect_cfg)
    np.testing.assert_array_equal(rs.budget, np.tile(np.arange(1, b_max + 1), 3))
    assert ft.budget.max() <= b_max


def test_successive_halving_invariants():
    from reproduction.baselines.scheduler import run_successive_halving

    rng = np.random.default_rng(3)
    n_cfg, b_max = 40, 27
    curves = rng.uniform(0.0, 0.5, size=(n_cfg, b_max)).astype(np.float32)
    curves[7] = np.linspace(0.6, 0.99, b_max)  # dominant config
    task = BenchmarkTask(
        "t/sh",
        rng.uniform(size=(n_cfg, 2)).astype(np.float32),
        curves,
        ((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )
    res = run_successive_halving(task, 500, seed=0)
    assert len(res.chosen) == 500
    counts = np.bincount(res.chosen, minlength=n_cfg)
    assert counts.max() <= b_max  # never trains past the table
    # once config 7 reaches rung 1 it should out-promote everything and finish training
    assert counts[7] == b_max
    assert res.f_best[-1] >= 0.99 - 1e-6
    # determinism
    res2 = run_successive_halving(task, 500, seed=0)
    np.testing.assert_array_equal(res.chosen, res2.chosen)


def test_rung_ladder_matches_neps_construction():
    for b_max in (10, 27, 50, 51, 1414):
        ours = rung_ladder(b_max, eta=3)
        # NePS _get_rung_map: nrungs top-down divisions from b_max, int cast each rung
        nrungs = int(np.floor(np.log(b_max) / np.log(3))) + 1
        neps, budget = [], float(b_max)
        for _ in range(nrungs):
            neps.append(int(budget))
            budget /= 3
        assert ours == sorted(neps), b_max
    assert rung_ladder(51, 3) == [1, 5, 17, 51]
    assert rung_ladder(10, 3) == [1, 3, 10]


def make_flat_task(step0, rest, b_max=9):
    """Constant-per-config curves except a distinct first step: step0[c] then rest[c]."""
    n_cfg = len(step0)
    curves = np.tile(np.asarray(rest, np.float32)[:, None], (1, b_max))
    curves[:, 0] = step0
    rng = np.random.default_rng(7)
    return BenchmarkTask(
        "t/flat",
        rng.uniform(size=(n_cfg, 2)).astype(np.float32),
        curves,
        ((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )


def test_asha_promotes_on_at_rung_value_not_prefix_max():
    # 9 configs so the li quota grants exactly three rung-0 promotions (A, B, C)
    """A peaks before the rung; B is better at the rung. Promotion must choose B."""
    step0 = np.array([0.9, 0.5, 0.4] + [0.2] * 6, np.float32)
    rest = np.array([0.1, 0.5, 0.4] + [0.05] * 6, np.float32)
    task = make_flat_task(step0, rest)  # ladder {1, 3, 9}
    # 21 steps: 9 rung-0 finishers + 3 promotions x2 steps + the rung-1 winner's +6.
    # 24 leaves headroom without letting the quota-free tail retrain A past budget 3.
    res = run_successive_halving(task, 24, seed=0)
    budgets = {cfg: res.budget[res.chosen == cfg].max() for cfg in np.unique(res.chosen)}
    assert budgets[1] == 9  # B: best at-rung value at rung 1
    assert budgets[0] == 3  # A: prefix max 0.9 must not carry it up the ladder


def test_asha_neps_variant_promotes_more_aggressively():
    """NePS quota is over current members (promoted configs leave the rung), so nearly every
    finisher eventually promotes; Li's ever-seen quota keeps ~1-1/eta of configs at rung 0.
    Budget is kept well below pool exhaustion so the quota-free tail never kicks in.
    """
    rng = np.random.default_rng(11)
    step0 = rng.uniform(0.1, 0.9, 30).astype(np.float32)
    task = make_flat_task(step0, step0, b_max=9)
    li = run_successive_halving(task, 40, seed=0, variant="li")
    neps = run_successive_halving(task, 40, seed=0, variant="neps")

    def frac_promoted(res):
        seen = np.unique(res.chosen)
        past0 = sum(res.budget[res.chosen == c].max() > 1 for c in seen)
        return past0 / len(seen)

    assert frac_promoted(neps) > frac_promoted(li) + 0.2
    # both variants agree on the ladder: budgets only pause at rung boundaries {1, 3, 9}
    # (the config active when the step budget cuts off is legitimately caught mid-rung)
    for res in (li, neps):
        for cfg in np.unique(res.chosen):
            if cfg != res.chosen[-1]:
                assert res.budget[res.chosen == cfg].max() in (1, 3, 9)


def test_hyperband_invariants():
    rng = np.random.default_rng(5)
    n_cfg, b_max = 12, 9
    curves = rng.uniform(0.0, 0.5, size=(n_cfg, b_max)).astype(np.float32)
    curves[4] = np.linspace(0.6, 0.99, b_max)  # dominant config
    task = BenchmarkTask(
        "t/hb",
        rng.uniform(size=(n_cfg, 2)).astype(np.float32),
        curves,
        ((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )
    res = run_hyperband(task, 100, seed=0)
    assert len(res.chosen) == 100  # 12 configs x 9 steps = 108 > 100: budget fully spent
    counts = np.bincount(res.chosen, minlength=n_cfg)
    assert counts.max() <= b_max  # never trains past the table
    # whichever bracket first drew config 4, it tops every rung and reaches b_max
    assert counts[4] == b_max
    assert res.f_best[-1] >= 0.99 - 1e-6
    res2 = run_hyperband(task, 100, seed=0)
    np.testing.assert_array_equal(res.chosen, res2.chosen)
    # exhausting the table terminates cleanly short of the requested budget
    full = run_hyperband(task, 500, seed=0)
    assert len(full.chosen) == n_cfg * b_max
    assert full.budget[full.chosen == 4].max() == b_max


def test_random_search_stops_at_table_end():
    task = make_task(n_cfg=2, b_max=3)
    result = run_random_search(task, n_iters=20)
    assert len(result.chosen) == 6
    assert len(set(zip(result.chosen, result.budget))) == 6


def test_invalid_halving_factor():
    import pytest

    for eta in (0, 1):
        with pytest.raises(ValueError, match="eta"):
            rung_ladder(10, eta)
