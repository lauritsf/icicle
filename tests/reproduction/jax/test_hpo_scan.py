"""Scan-runner tests: decision parity with the host loop where randomness does not engage,
plus the scan-only features (cost/budget, tie uniformity).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from reproduction.bench.task import BenchmarkTask
from reproduction.baselines.scheduler import run_freeze_thaw
from icicle_jax.bar_distribution import BarDistribution
from reproduction.jax.hpo_scan import run_freeze_thaw_scan
from icicle_jax.nn.model import Surrogate

BORDERS = np.linspace(0.0, 1.0, 9)


def _task(rng, n_cfg=12, b_max=6, n_hps=3, name="t", nan_tail=False, fid2d=False):
    curves = rng.uniform(0.1, 0.9, size=(n_cfg, b_max)).astype(np.float32)
    if nan_tail:  # qtmeta-style ragged curves: NaN suffixes, every prefix >= 1
        for i in range(1, n_cfg, 3):
            curves[i, rng.integers(1, b_max) :] = np.nan
    if fid2d:  # per-config cumulative cost (walltime-view shape), monotone, capped at 1
        steps = rng.uniform(0.02, 0.3, size=(n_cfg, b_max))
        fid = np.minimum(np.cumsum(steps, axis=1), 1.0).astype(np.float32)
    else:
        fid = ((np.arange(b_max) + 1) / b_max).astype(np.float32)
    return BenchmarkTask(
        name=name,
        hps=rng.uniform(size=(n_cfg, n_hps)).astype(np.float32),
        curves=curves,
        fidelities=fid,
    )


def _det_score(row, it):
    """Deterministic, level-independent, unique per (row, it) for row < 37."""
    return ((row * 11 + it * 5) % 37).astype(np.float32) / 37.0


def _scan_seam():
    def score_fn(state, t, point_cfg, point_fid, y, sep, params, it):
        r_pad = t["hps"].shape[1]  # query block is always the trailing arange(R_pad) rows
        rows = jnp.arange(r_pad)
        s = ((rows * 11 + it * 5) % 37).astype(jnp.float32) / 37.0
        return jnp.broadcast_to(s, (point_cfg.shape[0], r_pad))

    return score_fn


def _host_stub():
    state = {"it": 0}

    def predict(ctx_cfg, ctx_lvl, ctx_y, qry_cfg, qry_lvl):
        state["it"] += 1
        scores = _det_score(qry_cfg, state["it"])
        probabilities = np.zeros((len(qry_cfg), len(BORDERS) - 1))
        probabilities[:, 0] = 1 - scores
        probabilities[:, -1] = scores
        return probabilities

    return predict


def test_trajectory_parity_with_host_loop():
    rng = np.random.default_rng(0)
    chains = [
        (_task(rng, n_cfg=12, b_max=6, name="dense"), 0),
        (_task(rng, n_cfg=10, b_max=5, name="ragged", nan_tail=True), 1),
        (_task(rng, n_cfg=8, b_max=4, name="walltime", fid2d=True), 2),
        (_task(rng, n_cfg=3, b_max=3, name="tiny"), 3),  # exhausts: 9 obs < n_iters
    ]
    n_iters = 20
    host = [run_freeze_thaw(task, _host_stub(), BORDERS, n_iters, seed) for task, seed in chains]
    scan = run_freeze_thaw_scan(
        None,
        None,
        chains,
        borders=BORDERS,
        config_dim=3,
        n_iters=n_iters,
        score_fn=_scan_seam(),
    )
    for h, s in zip(host, scan, strict=True):
        np.testing.assert_array_equal(h.chosen, s.chosen)
        np.testing.assert_array_equal(h.budget, s.budget)
        np.testing.assert_allclose(h.observed, s.observed, atol=1e-6)
        np.testing.assert_allclose(h.f_best, s.f_best, atol=1e-6)
        np.testing.assert_allclose(h.regret, s.regret, atol=1e-6)


def test_context_buckets_preserve_external_score_trajectory():
    rng = np.random.default_rng(101)
    chains = [
        (_task(rng, n_cfg=12, b_max=8, name="a", nan_tail=True), 3),
        (_task(rng, n_cfg=10, b_max=7, name="b"), 4),
    ]
    runs = {
        buckets: run_freeze_thaw_scan(
            None,
            None,
            chains,
            borders=BORDERS,
            config_dim=3,
            n_iters=24,
            score_fn=_scan_seam(),
            ctx_buckets=buckets,
        )
        for buckets in (1, 4)
    }
    for single, segmented in zip(runs[1], runs[4], strict=True):
        np.testing.assert_array_equal(single.chosen, segmented.chosen)
        np.testing.assert_array_equal(single.budget, segmented.budget)
        np.testing.assert_array_equal(single.observed, segmented.observed)
        np.testing.assert_array_equal(single.f_best, segmented.f_best)
        np.testing.assert_array_equal(single.regret, segmented.regret)


def test_cost_accumulates_and_budget_kills():
    rng = np.random.default_rng(1)
    task = _task(rng, n_cfg=6, b_max=5, name="wt", fid2d=True)
    fid = np.asarray(task.fidelities)
    inc = np.diff(fid, axis=1, prepend=0.0)

    free = run_freeze_thaw_scan(
        None,
        None,
        [(task, 0), (task, 0)],
        borders=BORDERS,
        config_dim=3,
        n_iters=15,
        score_fn=_scan_seam(),
        budget=None,
    )
    capped = run_freeze_thaw_scan(
        None,
        None,
        [(task, 0), (task, 0)],
        borders=BORDERS,
        config_dim=3,
        n_iters=15,
        score_fn=_scan_seam(),
        budget=0.5,
    )
    r = free[0]
    expect = np.cumsum([inc[c, b - 1] for c, b in zip(r.chosen, r.budget, strict=True)])
    np.testing.assert_allclose(r.cost, expect, atol=1e-6)
    c = capped[0]
    assert len(c.chosen) < len(r.chosen), "budget cap should truncate the trajectory"
    np.testing.assert_array_equal(c.chosen, r.chosen[: len(c.chosen)])
    # recording stops at the first step that ENDS past the budget
    assert all(c.cost[:-1] < 0.5)


def test_paired_initial_config_and_exact_table_cap_exclude_before_observe():
    rng = np.random.default_rng(9)
    base = _task(rng, n_cfg=4, b_max=4, name="exact")
    costs = np.full((4, 4), 3.0, np.float64)
    task = dataclasses_replace_costs(base, costs)
    results, proposals = run_freeze_thaw_scan(
        None,
        None,
        [(task, 200)],
        borders=BORDERS,
        config_dim=3,
        n_iters=8,
        score_fn=_scan_seam(),
        budget=1.0,
        initial_configs=[2],
        exact_cap=True,
        use_table_costs=True,
        cost_scale=10.0,
        return_proposals=True,
    )
    result, proposal = results[0], proposals[0]
    assert result.chosen[0] == 2
    assert len(result.chosen) == 3
    np.testing.assert_allclose(result.cost, [0.3, 0.6, 0.9], atol=1e-6)
    live = np.flatnonzero(proposal["proposed"])
    assert proposal["accepted"][live[-1]] == np.bool_(False)
    assert proposal["budget"][live[-1]] >= 1
    assert proposal["proposed_cumulative_cost"][live[-1]] > 1.0


def dataclasses_replace_costs(task, costs):
    import dataclasses

    return dataclasses.replace(task, costs=costs)


def test_exact_cap_allows_free_steps_then_records_rejection():
    task = _task(np.random.default_rng(2), n_cfg=1, b_max=4)
    task = dataclasses_replace_costs(task, np.array([[2., 2., 0., 1.]]))
    results, proposals = run_freeze_thaw_scan(
        None, None, [(task, 0)], borders=BORDERS, config_dim=3,
        n_iters=5, score_fn=_scan_seam(), budget=1., exact_cap=True,
        use_table_costs=True, cost_scale=4., return_proposals=True,
    )
    np.testing.assert_array_equal(results[0].budget, [1, 2, 3])
    np.testing.assert_array_equal(results[0].cost, [.5, 1., 1.])
    live = proposals[0]["proposed"]
    np.testing.assert_array_equal(proposals[0]["accepted"][live], [True, True, False])


def test_exact_cap_stops_scoring_only_after_the_last_chain_finishes():
    import jax

    base = _task(np.random.default_rng(9), n_cfg=4, b_max=4)
    tasks = [dataclasses_replace_costs(base, np.full((4, 4), cost)) for cost in (3., 1.5)]
    calls = []
    score = _scan_seam()

    def counted_score(*args):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return score(*args)

    results, proposals = run_freeze_thaw_scan(
        None, None, [(task, 0) for task in tasks], borders=BORDERS, config_dim=3,
        n_iters=12, score_fn=counted_score, budget=1., exact_cap=True,
        use_table_costs=True, cost_scale=10., return_proposals=True,
    )
    jax.effects_barrier()
    assert [len(result.chosen) for result in results] == [3, 6]
    assert len(calls) == 6  # Includes the last rejected proposal, but no later inference.
    for proposal in proposals:
        assert not proposal["accepted"][proposal["proposed"]][-1]


def test_exhausted_chain_stops_while_batch_continues():
    rng = np.random.default_rng(2)
    tasks = [_task(rng, n_cfg=1, b_max=1), _task(rng, n_cfg=3, b_max=2)]
    results = run_freeze_thaw_scan(
        None, None, [(task, 0) for task in tasks], borders=BORDERS,
        config_dim=3, n_iters=10, score_fn=_scan_seam(),
    )
    for task, result in zip(tasks, results, strict=True):
        assert len(result.chosen) == task.curves.size
        assert len(set(zip(result.chosen, result.budget, strict=True))) == task.curves.size
        np.testing.assert_array_equal(result.observed, task.curves[result.chosen, result.budget - 1])


def test_tie_break_uniform_over_candidates():
    rng = np.random.default_rng(2)
    task = _task(rng, n_cfg=8, b_max=50, name="ties")

    def flat(state, t, point_cfg, point_fid, y, sep, params, it):
        return jnp.ones((point_cfg.shape[0], t["hps"].shape[1]), jnp.float32)

    res = run_freeze_thaw_scan(
        None, None, [(task, 0)], borders=BORDERS, config_dim=3, n_iters=200, score_fn=flat
    )[0]
    counts = np.bincount(res.chosen, minlength=8)
    assert counts.min() >= 5, f"tie-break starves candidates: {counts}"


def test_cost_acquisition_prefers_the_cheap_config():
    curves = np.full((2, 6), 0.5, np.float32)
    fid = np.stack([np.cumsum(np.full(6, 0.01)), np.cumsum(np.full(6, 0.1))]).astype(np.float32)
    task = BenchmarkTask(
        name="cost", hps=np.zeros((2, 3), np.float32), curves=curves, fidelities=fid
    )

    def flat(state, t, point_cfg, point_fid, y, sep, params, it):
        return jnp.ones((point_cfg.shape[0], t["hps"].shape[1]), jnp.float32)

    res = run_freeze_thaw_scan(
        None,
        None,
        [(task, 0)],
        borders=BORDERS,
        config_dim=3,
        n_iters=6,
        score_fn=flat,
        acquisition_name="mfpi-cost-cooled",
        budget=1.0,
    )[0]
    assert (res.chosen[1:] == 0).all(), res.chosen
    with pytest.raises(ValueError, match="budget"):
        run_freeze_thaw_scan(
            None,
            None,
            [(task, 0)],
            borders=BORDERS,
            config_dim=3,
            n_iters=4,
            score_fn=flat,
            acquisition_name="mfpi-cost-cooled",
        )


def test_end_to_end_smoke_with_model():
    rng = np.random.default_rng(4)
    # remat=True: the production checkpoints train with it, and the scan must drop it
    # (nnx.remat inside lax.scan trips a trace-level error) -- regression for that path
    model = Surrogate(
        emsize=16,
        num_heads=4,
        hidden_dim=32,
        n_feat_layers=1,
        n_icl_layers=1,
        num_bars=8,
        icl_backend="xla",
        remat=True,
        rngs=nnx.Rngs(0),
    )
    criterion = BarDistribution(jnp.array([0.0, 0.02, 0.08, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0]))
    chains = [
        (_task(rng, n_cfg=6, b_max=4, name="s1"), 0),
        (_task(rng, n_cfg=6, b_max=4, name="s2"), 1),
    ]
    for acq in ("mfpi-random", "mfpi-cost-cooled"):
        results = run_freeze_thaw_scan(
            model,
            criterion,
            chains,
            borders=BORDERS,
            config_dim=4,
            n_iters=8,
            acquisition_name=acq,
            budget=4.0 if acq == "mfpi-cost-cooled" else None,
        )
        for r in results:
            assert np.isfinite(r.f_best).all() and np.isfinite(r.observed).all()
            assert (np.diff(r.regret) <= 1e-7).all(), "regret must be non-increasing"


def test_adaptive_value_map_presents_the_observed_range_onto_the_band():
    import jax.numpy as jnp

    from reproduction.jax.hpo_scan import _present, _Static

    static = _Static(
        n_iters=2,
        r_pad=4,
        l_pad=3,
        n_hps=2,
        config_dim=2,
        budget=None,
        value_map="adaptive",
    )
    # a numerai-like sliver: every observed value inside 0.030 of 0.493
    y = jnp.asarray([[0.493, 0.508, 0.523, 0.0, 0.0]])
    real = jnp.asarray([[True, True, True, False, False]])
    out, fb = _present(static, y, real, jnp.asarray([0.523]))

    from icicle.hpo.metric_map import MetricMap

    mapping = MetricMap()
    mapping.fit(np.asarray(y)[np.asarray(real)])
    np.testing.assert_allclose(np.asarray(out)[np.asarray(real)], mapping(np.asarray(y)[np.asarray(real)]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(fb), mapping([0.523]), atol=1e-6)
    assert float(fb[0]) < 1.0


def test_adaptive_value_map_is_inert_when_off_and_neutral_with_no_scale():
    import jax.numpy as jnp

    from reproduction.jax.hpo_scan import _present, _Static

    kw = dict(
        n_iters=2,
        r_pad=4,
        l_pad=3,
        n_hps=2,
        config_dim=2,
        budget=None,
    )
    y = jnp.asarray([[0.4, 0.9, 0.0]])
    real = jnp.asarray([[True, True, False]])
    fb = jnp.asarray([0.9])

    off = _present(_Static(**kw), y, real, fb)  # "off": bit-identical passthrough
    assert off[0] is y and off[1] is fb

    # With one observation, map the value to the band midpoint.
    one = jnp.asarray([[True, False, False]])
    out, fbp = _present(_Static(**kw, value_map="adaptive"), y, one, fb)
    assert float(out[0, 0]) == pytest.approx(0.5, abs=1e-6)
    assert float(fbp[0]) == pytest.approx(0.5, abs=1e-6)


_MAP_STATIC = dict(
    n_iters=2,
    r_pad=4,
    l_pad=3,
    n_hps=2,
    config_dim=2,
    budget=None,
)


def test_fixed_value_map_presents_values_and_threshold_onto_the_band():
    """The contract: constant anchors -- raw [0, 1] lands on the band by the affine map,
    and the MFPI threshold basis f_best moves by the same affine (values and threshold live in
    one unit system), keeping reserve above even a raw incumbent of 1.0.
    """
    from reproduction.jax.hpo_scan import _present, _Static

    lo, hi = 1.0 / 3.0, 2.0 / 3.0
    static = _Static(**_MAP_STATIC, value_map="fixed")
    y = jnp.asarray([[0.0, 0.493, 0.523, 1.0]])
    real = jnp.asarray([[True, True, True, True]])
    fb = jnp.asarray([1.0])  # raw incumbent at the table ceiling
    out, fbp = _present(static, y, real, fb)

    np.testing.assert_allclose(np.asarray(out), lo + (hi - lo) * np.asarray(y), atol=1e-7)
    # model-visible values sit in-band (f32 rounding at the edges)
    assert lo - 1e-7 <= float(out.min()) and float(out.max()) <= hi + 1e-7
    # Fixed anchors preserve the occupied fraction of the band.
    assert float(out[0, 2] - out[0, 1]) == pytest.approx((hi - lo) * 0.030, abs=1e-6)
    assert float(fbp[0]) == pytest.approx(lo + (hi - lo) * 1.0, abs=1e-7)
    assert float(fbp[0]) < 1.0  # reserve above the band: (1 - f_best') is never 0


def test_fixed_value_map_leaves_reported_regret_in_raw_units():
    rng = np.random.default_rng(8)
    chains = [
        (_task(rng, n_cfg=10, b_max=5, name="raw1", nan_tail=True), 0),
        (_task(rng, n_cfg=8, b_max=4, name="raw2"), 1),
    ]
    runs = {
        vm: run_freeze_thaw_scan(
            None,
            None,
            chains,
            borders=BORDERS,
            config_dim=3,
            n_iters=12,
            score_fn=_scan_seam(),
            value_map=vm,
        )
        for vm in ("off", "fixed")
    }
    for off, fixed in zip(runs["off"], runs["fixed"], strict=True):
        np.testing.assert_array_equal(off.chosen, fixed.chosen)
        np.testing.assert_array_equal(np.asarray(off.observed), np.asarray(fixed.observed))
        np.testing.assert_array_equal(np.asarray(off.f_best), np.asarray(fixed.f_best))
        np.testing.assert_array_equal(np.asarray(off.regret), np.asarray(fixed.regret))


def test_initial_evaluation_respects_budget():
    task = _task(np.random.default_rng(0), n_cfg=2, b_max=2)
    kwargs = dict(borders=BORDERS, config_dim=3, n_iters=4,
                  initial_configs=[0], score_fn=_scan_seam())
    with pytest.raises(ValueError, match="initial evaluation exceeds"):
        run_freeze_thaw_scan(None, None, [(task, 0)], budget=.25, exact_cap=True, **kwargs)
    for exact in (False, True):
        result, = run_freeze_thaw_scan(None, None, [(task, 0)], budget=.5, exact_cap=exact, **kwargs)
        assert len(result.chosen) == 1
        assert result.cost[-1] == .5


def test_runner_cache_distinguishes_borders(monkeypatch):
    from reproduction.jax import hpo_scan

    original = hpo_scan._build_scan
    built = []
    def build(graphdef, criterion, borders, static, score_fn):
        built.append(np.asarray(borders))
        return original(graphdef, criterion, borders, static, score_fn)
    monkeypatch.setattr(hpo_scan, '_build_scan', build)
    task = _task(np.random.default_rng(0), n_cfg=2, b_max=2)
    score = _scan_seam()
    for borders in (BORDERS, BORDERS * 2, BORDERS):
        run_freeze_thaw_scan(None, None, [(task, 0)], borders=borders,
                            config_dim=3, n_iters=4, score_fn=score)
    assert len(built) == 2
    np.testing.assert_array_equal(built[1], BORDERS * 2)
