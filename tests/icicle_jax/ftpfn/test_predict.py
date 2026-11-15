"""The FT-PFN shim's tokenization semantics and padding scheme.

The id rules mirror the official ``ifbo`` package's ``tokenize`` + ``forward``
(read verbatim from the source): first-appearance 1-based ordinals, exact
vector equality, id 0 for queries never observed in context. Tested directly
on ``assign_curve_ids`` (hand-computed expectations) plus, on a tiny random
model, the padding invariance the parked-pad scheme must provide.
"""

import numpy as np
import pytest


def test_assign_curve_ids_official_semantics():
    from icicle_jax.ftpfn.predict import assign_curve_ids

    a, b, c = [0.9, 0.1], [0.2, 0.8], [0.5, 0.5]
    # first-appearance order (b before a would be lexicographic; a comes first here)
    ctx = np.asarray([a, a, b, a])
    qry = np.asarray([b, c, a])
    ctx_ids, qry_ids = assign_curve_ids(ctx, qry)
    np.testing.assert_array_equal(ctx_ids, [1, 1, 2, 1])  # a=1 (appears first), b=2
    # b observed -> 2; c never observed -> 0; a observed -> 1
    np.testing.assert_array_equal(qry_ids, [2, 0, 1])


def test_assign_curve_ids_empty_context_all_zero():
    from icicle_jax.ftpfn.predict import assign_curve_ids

    ctx_ids, qry_ids = assign_curve_ids(np.zeros((0, 3)), np.asarray([[0.1] * 3, [0.2] * 3]))
    assert len(ctx_ids) == 0
    np.testing.assert_array_equal(qry_ids, [0, 0])


@pytest.fixture(scope="module")
def tiny():
    import jax.numpy as jnp
    from flax import nnx

    from icicle_jax.bar_distribution import BarDistribution
    from icicle_jax.ftpfn.model import FTPFN

    model = FTPFN(
        config_dim=10,
        emsize=16,
        num_heads=2,
        hidden_dim=32,
        num_layers=2,
        num_bars=8,
        max_curves=5,
        rngs=nnx.Rngs(0),
    )
    criterion = BarDistribution(borders=jnp.linspace(0.0, 1.0, 9))
    return model, criterion


def test_predict_fn_shapes_probs_and_padding_invariance(tiny):
    model, criterion = tiny
    from icicle_jax.ftpfn import make_ftpfn_predict_fn

    predict = make_ftpfn_predict_fn(model, criterion, ctx_pad=8, qry_pad=4)
    ctx_hps = np.asarray([[0.3] * 4, [0.3] * 4, [0.7] * 4])
    ctx_fid = np.asarray([0.2, 0.4, 0.2])
    ctx_y = np.asarray([0.5, 0.6, 0.4])
    q1 = np.asarray([[0.3] * 4])

    p1 = predict(ctx_hps, ctx_fid, ctx_y, q1, np.asarray([0.6]))
    assert p1.shape == (1, 8)
    np.testing.assert_allclose(p1.sum(axis=1), 1.0, atol=1e-5)

    # More queries in the same call must not change the first query's answer:
    # queries never attend each other, and pad rows are parked past sep.
    q3 = np.asarray([[0.3] * 4, [0.9] * 4, [0.7] * 4])
    p3 = predict(ctx_hps, ctx_fid, ctx_y, q3, np.asarray([0.6, 0.5, 0.9]))
    np.testing.assert_allclose(p3[0], p1[0], atol=1e-6)




def test_predict_fn_curve_capacity_error(tiny):
    model, criterion = tiny
    from icicle_jax.ftpfn import make_ftpfn_predict_fn

    predict = make_ftpfn_predict_fn(model, criterion, ctx_pad=8, qry_pad=8)
    ctx_hps = np.linspace(0.1, 0.9, 6)[:, None] * np.ones((6, 4))  # 6 distinct > max_curves 5
    with pytest.raises(ValueError, match="distinct context curves"):
        predict(ctx_hps, np.full(6, 0.5), np.full(6, 0.5), ctx_hps[:1], np.asarray([1.0]))


def test_scan_curve_ids_match_the_official_assignment():
    import jax.numpy as jnp

    from icicle_jax.ftpfn.predict import assign_curve_ids, scan_curve_ids

    rng = np.random.default_rng(0)
    n_cfg, ctx_pad = 6, 10
    hps = rng.uniform(size=(n_cfg, 3)).astype(np.float32)
    ctx_rows = np.array([0, 0, 2, 3, 3, 3], np.int32)  # config-major: ascending, contiguous
    qry_rows = np.arange(n_cfg, dtype=np.int32)
    n_ctx = len(ctx_rows)
    want_ctx, want_qry = assign_curve_ids(hps[ctx_rows], hps[qry_rows])

    point_cfg = np.concatenate([np.pad(ctx_rows, (0, ctx_pad - n_ctx)), qry_rows])[None]
    keep = (np.arange(ctx_pad) < n_ctx)[None]
    got = np.asarray(scan_curve_ids(jnp.asarray(point_cfg), jnp.asarray(keep), n_cfg))[0]

    np.testing.assert_array_equal(got[:n_ctx], want_ctx)
    np.testing.assert_array_equal(got[ctx_pad:], want_qry)  # unobserved configs -> 0
    assert set(np.unique(want_qry)) == {0, 1, 2, 3}  # the case is not degenerate


def test_scan_curve_ids_ignore_context_padding():
    import jax.numpy as jnp

    from icicle_jax.ftpfn.predict import scan_curve_ids

    n_cfg, ctx_pad = 4, 6
    # real context is rows [1, 2]; the two padding slots carry row 0, which is not observed
    point_cfg = np.concatenate(
        [np.array([1, 2, 0, 0], np.int32), np.arange(n_cfg, dtype=np.int32)]
    )[None]
    keep = (np.arange(ctx_pad - 2) < 2)[None]
    got = np.asarray(scan_curve_ids(jnp.asarray(point_cfg), jnp.asarray(keep), n_cfg))[0]
    np.testing.assert_array_equal(got[4:], [0, 1, 2, 0])  # row 0 unobserved despite the padding


def test_scan_score_fn_matches_raw_predictions(tiny):
    """Scan and host scorers agree for identical context and queries.

    Compare predictions directly because the runners use different RNG streams
    for later acquisition draws and cannot be trajectory-paired in general.
    """
    model, criterion = tiny
    import jax.numpy as jnp

    from reproduction.bench.task import BenchmarkTask
    from icicle_jax.ftpfn import make_ftpfn_predict_fn, make_ftpfn_scan_score_fn
    from icicle.hpo.decider import exceedance_prob

    rng = np.random.default_rng(1)
    n_cfg, n_lvl, n_hps, ctx_pad = 4, 3, 4, 8
    borders = np.linspace(0.0, 1.0, 9)
    tasks = [
        BenchmarkTask(
            name=f"t{j}",
            hps=rng.uniform(size=(n_cfg, n_hps)).astype(np.float32),
            curves=rng.uniform(size=(n_cfg, n_lvl)).astype(np.float32),
            fidelities=np.linspace(0.3, 1.0, n_lvl).astype(np.float32),
        )
        for j in range(2)
    ]
    # config-major, and EQUAL length across chains: the scan advances in lockstep, so sep is
    # one number for the whole batch
    ctx_cfg = [np.array([0, 0, 1], np.int32), np.array([1, 2, 2], np.int32)]
    ctx_lvl = [np.array([0, 1, 0], np.int32), np.array([0, 0, 1], np.int32)]
    ctx_y = [t.curves[c, lv] for t, c, lv in zip(tasks, ctx_cfg, ctx_lvl, strict=True)]
    qry_cfg = [np.arange(n_cfg, dtype=np.int32) for _ in tasks]
    qry_lvl = [np.full(n_cfg, n_lvl - 1, np.int32) for _ in tasks]
    params = np.asarray([0.6, 0.7], np.float32)
    n_ctx = len(ctx_cfg[0])

    raw = make_ftpfn_predict_fn(model, criterion, ctx_pad=ctx_pad, qry_pad=n_cfg)
    want = np.stack([
        exceedance_prob(raw(
            task.hps[ctx_cfg[i]], task.fidelities[ctx_lvl[i]], ctx_y[i],
            task.hps[qry_cfg[i]], task.fidelities[qry_lvl[i]],
        ), borders, params[i])
        for i, task in enumerate(tasks)
    ])

    # the scan's packing: [context buffer of width ctx_pad | one query per config]
    b = len(tasks)
    point_cfg = np.zeros((b, ctx_pad + n_cfg), np.int32)
    point_fid = np.zeros((b, ctx_pad + n_cfg), np.float32)  # zeros: padding must be masked to 1
    y = np.zeros((b, ctx_pad + n_cfg), np.float32)
    for i, t in enumerate(tasks):
        point_cfg[i, :n_ctx] = ctx_cfg[i]
        point_fid[i, :n_ctx] = t.fidelities[ctx_lvl[i]]
        y[i, :n_ctx] = ctx_y[i]
        point_cfg[i, ctx_pad:] = qry_cfg[i]
        point_fid[i, ctx_pad:] = t.fidelities[qry_lvl[i]]
    tables = {"hps": jnp.asarray(np.stack([t.hps for t in tasks]))}

    scan = make_ftpfn_scan_score_fn(
        model, criterion, borders=borders, n_hps=n_hps, max_distinct_curves=4
    )
    got = np.asarray(
        scan(
            None,
            tables,
            jnp.asarray(point_cfg),
            jnp.asarray(point_fid),
            jnp.asarray(y),
            jnp.asarray(n_ctx),
            jnp.asarray(params),
            1,
        )
    )
    np.testing.assert_allclose(got, want[:, :n_cfg], atol=1e-6)

    for query_chunk_size, prefix_attention in ((None, True), (2, False), (2, True)):
        optimized = make_ftpfn_scan_score_fn(
            model,
            criterion,
            borders=borders,
            n_hps=n_hps,
            max_distinct_curves=4,
            query_chunk_size=query_chunk_size,
            prefix_attention=prefix_attention,
        )
        got_optimized = np.asarray(
            optimized(
                None,
                tables,
                jnp.asarray(point_cfg),
                jnp.asarray(point_fid),
                jnp.asarray(y),
                jnp.asarray(n_ctx),
                jnp.asarray(params),
                1,
            )
        )
        np.testing.assert_allclose(got_optimized, got, atol=1e-6)

        # A segmented scan recompiles the same scoring interface with a different padded-context
        # width. Real context and queries stay identical; only masked padding grows.
        wide_pad = ctx_pad + 5
        wide_cfg = np.zeros((b, wide_pad + n_cfg), np.int32)
        wide_fid = np.zeros((b, wide_pad + n_cfg), np.float32)
        wide_y = np.zeros((b, wide_pad + n_cfg), np.float32)
        wide_cfg[:, :n_ctx] = point_cfg[:, :n_ctx]
        wide_fid[:, :n_ctx] = point_fid[:, :n_ctx]
        wide_y[:, :n_ctx] = y[:, :n_ctx]
        wide_cfg[:, wide_pad:] = point_cfg[:, ctx_pad:]
        wide_fid[:, wide_pad:] = point_fid[:, ctx_pad:]
        got_wide = np.asarray(
            optimized(
                None,
                tables,
                jnp.asarray(wide_cfg),
                jnp.asarray(wide_fid),
                jnp.asarray(wide_y),
                jnp.asarray(n_ctx),
                jnp.asarray(params),
                1,
            )
        )
        np.testing.assert_allclose(got_wide, got_optimized, atol=1e-6)


def test_scan_score_fn_rejects_more_curves_than_the_id_embedding(tiny):
    model, criterion = tiny
    from icicle_jax.ftpfn import make_ftpfn_scan_score_fn

    with pytest.raises(ValueError, match="distinct curves"):
        make_ftpfn_scan_score_fn(
            model,
            criterion,
            borders=np.linspace(0.0, 1.0, 9),
            n_hps=4,
            max_distinct_curves=6,  # the tiny model's id embedding holds 5
        )
