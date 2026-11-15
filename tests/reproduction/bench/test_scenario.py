"""Tests for the eval scenario sampler (backend-free, numpy only)."""

from pathlib import Path

import numpy as np
import pytest

from reproduction.bench import BenchmarkTask, EvalProtocol, iter_scenarios


def make_task(n_cfg=30, b_max=15, n_hps=4, name="test/task", seed=0):
    rng = np.random.default_rng(seed)
    return BenchmarkTask(
        name=name,
        hps=rng.uniform(size=(n_cfg, n_hps)).astype(np.float32),
        curves=rng.uniform(size=(n_cfg, b_max)).astype(np.float32),
        fidelities=((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )


def make_ragged_task(lengths=(15, 1, 7, 15, 3, 12), b_max=15, n_hps=4, per_config_fid=False):
    """NaN-suffix curves per ``lengths``; optionally per-config (2-D) fidelities."""
    rng = np.random.default_rng(1)
    n_cfg = len(lengths)
    curves = np.full((n_cfg, b_max), np.nan, dtype=np.float32)
    for i, n in enumerate(lengths):
        curves[i, :n] = rng.uniform(size=n)
    if per_config_fid:
        fidelities = np.full((n_cfg, b_max), np.nan, dtype=np.float32)
        for i, n in enumerate(lengths):
            fidelities[i, :n] = np.sort(rng.uniform(size=n))
    else:
        fidelities = ((np.arange(b_max) + 1) / b_max).astype(np.float32)
    return BenchmarkTask(
        name="test/ragged",
        hps=rng.uniform(size=(n_cfg, n_hps)).astype(np.float32),
        curves=curves,
        fidelities=fidelities,
    )


PROTO = EvalProtocol(n_total=100, n_scenarios=20)
PARITY = EvalProtocol(mode="ifbo-parity", n_total=100, n_scenarios=20)


def points_of(s):
    return list(
        zip(s.config_index[s.point_config_idx].tolist(), s.level_index.tolist(), strict=True)
    )


@pytest.mark.parametrize("protocol", [PROTO, PARITY], ids=["icicle-v1", "ifbo-parity"])
def test_invariants(protocol):
    task = make_task()
    n_seen = 0
    for _idx, s in iter_scenarios(task, protocol, seed=0):
        n_seen += 1
        pts = points_of(s)
        assert len(pts) == len(set(pts)) == s.n_total_eff == len(s.level_index)
        assert 0 < s.n_context < s.n_total_eff

        ctx, qry = pts[: s.n_context], pts[s.n_context :]
        cutoff = {}
        for cid, level in ctx:
            cutoff[cid] = cutoff.get(cid, 0) + 1
            # prefix: the k-th context point of a curve is exactly level k-1
            assert level == cutoff[cid] - 1
        for cid, level in qry:
            assert level >= cutoff.get(cid, 0)
    assert n_seen == protocol.n_scenarios


def test_icicle_v1_grid_and_forced_final_level():
    task = make_task()
    b_max = task.curves.shape[1]
    seen_cells = set()
    for idx, s in iter_scenarios(task, PROTO, seed=0):
        frac = PROTO.context_fractions[idx % 5]
        assert s.context_fraction == frac
        assert s.n_context == int(np.clip(round(frac * s.n_total_eff), 1, s.n_total_eff - 1))
        seen_cells.add((s.context_fraction, round(np.log10(s.alpha))))
        queried = {}
        for cid, level in points_of(s)[s.n_context :]:
            queried.setdefault(cid, set()).add(level)
        for levels in queried.values():
            assert b_max - 1 in levels
    assert len(seen_cells) == 20  # 5 fractions x 4 alphas, each exactly once


def test_pairing_determinism():
    task = make_task()
    a = list(iter_scenarios(task, PROTO, seed=7))
    b = list(iter_scenarios(task, PROTO, seed=7))
    assert len(a) == len(b) > 0
    for (ia, sa), (ib, sb) in zip(a, b, strict=True):
        assert ia == ib
        np.testing.assert_array_equal(sa.point_config_idx, sb.point_config_idx)
        np.testing.assert_array_equal(sa.level_index, sb.level_index)
        np.testing.assert_array_equal(sa.config_index, sb.config_index)
        assert sa.n_context == sb.n_context and sa.alpha == sb.alpha
    # different seed -> different scenarios
    c = list(iter_scenarios(task, PROTO, seed=8))
    assert any(
        not np.array_equal(sa.level_index, sc.level_index)
        for (_, sa), (_, sc) in zip(a, c, strict=True)
    )


def test_small_pool():
    task = make_task(n_cfg=5, b_max=10)  # pool = 50 < n_total = 100
    scenarios = list(iter_scenarios(task, PROTO, seed=0))
    assert len(scenarios) == PROTO.n_scenarios
    for _idx, s in scenarios:
        assert s.n_total_eff == 50
        assert len(points_of(s)) == len(set(points_of(s))) == 50  # the whole pool, once
    assert list(iter_scenarios(task, PARITY, seed=0)) == []


def test_parity_fixed_context_size():
    task = make_task(n_cfg=50, b_max=20)  # pool = 1000
    proto = EvalProtocol(mode="ifbo-parity", n_total=200, n_scenarios=4, context_size=150)
    for _idx, s in iter_scenarios(task, proto, seed=0):
        # realized context can only shrink below the request via the max_curves cap,
        # which cannot bind here (50 curves << cap)
        assert s.n_context == 150


def test_parity_golden_vs_ifbo_jax():
    """Reconstruct reference tokens and compare with saved ifbo-jax scenarios.

    Skip the model-specific curve-ID column; context and query masks still
    verify the curve-cap behavior.
    """
    golden_path = Path(__file__).parent / "golden" / "parity_scenarios.npz"
    golden = np.load(golden_path)
    task = make_task()  # must stay in sync with the dump script's constants
    proto = EvalProtocol(mode="ifbo-parity", n_total=100, n_scenarios=5)

    scenarios = dict(iter_scenarios(task, proto, seed=0))
    assert len(scenarios) == 5
    for idx, s in scenarios.items():
        x, y = golden[f"x_{idx}"], golden[f"y_{idx}"]
        sep, qmask, y_true = golden[f"sep_{idx}"], golden[f"qmask_{idx}"], golden[f"y_true_{idx}"]

        assert s.n_context == int(sep)
        assert s.n_total_eff == 100
        cfg = s.config_index[s.point_config_idx]
        np.testing.assert_array_equal(task.fidelities[s.level_index], x[:, 1])
        np.testing.assert_array_equal(task.hps[cfg], x[:, 2:])
        np.testing.assert_array_equal(np.arange(100) >= s.n_context, qmask)
        values = task.curves[cfg, s.level_index]
        np.testing.assert_array_equal(np.where(qmask, 0.0, values), y)
        np.testing.assert_array_equal(np.where(qmask, values, 0.0), y_true)


def test_icicle_v1_golden():
    golden = np.load(Path(__file__).parent / "golden" / "icicle_v1_scenarios.npz")
    scenarios = dict(iter_scenarios(make_task(), PROTO, seed=0))
    assert len(scenarios) == 20
    for idx, s in scenarios.items():
        np.testing.assert_array_equal(s.config_index, golden[f"config_index_{idx}"])
        np.testing.assert_array_equal(s.point_config_idx, golden[f"point_config_idx_{idx}"])
        np.testing.assert_array_equal(s.level_index, golden[f"level_index_{idx}"])
        assert s.n_context == int(golden[f"n_context_{idx}"])
        assert s.n_total_eff == int(golden[f"n_total_eff_{idx}"])
        assert s.alpha == float(golden[f"alpha_{idx}"])


@pytest.mark.parametrize("per_config_fid", [False, True], ids=["shared-fid", "2d-fid"])
def test_ragged_invariants(per_config_fid):
    """Sampling never touches a NaN-padded level; context is a prefix within the observed
    support; every queried curve force-includes its own last observed level; the whole
    observed pool (sum of lengths < n_total) is used exactly once.
    """
    lengths = (15, 1, 7, 15, 3, 12)
    task = make_ragged_task(lengths, per_config_fid=per_config_fid)
    pool = sum(lengths)  # 53 < n_total=100 -> icicle-v1 shrinks to the observed pool
    n_seen = 0
    for _idx, s in iter_scenarios(task, PROTO, seed=0):
        n_seen += 1
        pts = points_of(s)
        assert s.n_total_eff == pool
        assert len(pts) == len(set(pts)) == pool
        for cid, level in pts:
            assert level < lengths[cid]
        ctx, qry = pts[: s.n_context], pts[s.n_context :]
        cutoff = {}
        for cid, level in ctx:
            cutoff[cid] = cutoff.get(cid, 0) + 1
            assert level == cutoff[cid] - 1
        queried = {}
        for cid, level in qry:
            assert level >= cutoff.get(cid, 0)
            queried.setdefault(cid, set()).add(level)
        for cid, levels in queried.items():
            assert lengths[cid] - 1 in levels
    assert n_seen == PROTO.n_scenarios


def test_parity_max_curves_cap():
    task = make_task(n_cfg=40, b_max=5)
    capped = EvalProtocol(mode="ifbo-parity", n_total=100, n_scenarios=6, max_curves=3)
    for _idx, s in iter_scenarios(task, capped, seed=0):
        ctx_curves = {cid for cid, _ in points_of(s)[: s.n_context]}
        assert len(ctx_curves) <= 3
