"""Run batched freeze-thaw BO loops on device with lax.scan.

Tables, acquisition draws, context packing and decisions stay on device until
trajectories are returned. JAX random streams differ from the NumPy reference
runner; do not merge results from the two implementations.

Fidelity increments define costs unless an explicit step-cost table is supplied.
Per-configuration fidelity tables support cumulative walltime. Cost cooling uses
fidelity differences; the budget can use exact table costs.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from reproduction.bench.task import BenchmarkTask
from icicle.hpo.metric_map import DEFAULT_BAND
from reproduction.baselines.scheduler import FreezeThawResult, _valid_lengths
from icicle_jax import acquisition

_SCAN_CACHE: dict[tuple, tuple] = {}


@dataclasses.dataclass(frozen=True)
class _Static:
    """Settings for one compiled scan segment."""

    n_iters: int
    r_pad: int  # padded row count (all chains)
    l_pad: int  # padded level count
    n_hps: int
    config_dim: int
    budget: float | None
    feature_seed: int = 0
    exact_cap: bool = False
    use_table_costs: bool = False
    cost_scale: float = 1.0
    cost_cooled: bool = False
    value_map: str = "off"
    # Size context buffers for this segment [it_start, it_end). More segments
    # reduce padding but require more compiled programs. Different shapes can
    # change float32 rounding and therefore BO decisions.
    it_start: int = 1
    it_end: int | None = None  # None = n_iters


def _pack_tables(chains, static: _Static):
    """Stack each chain's benchmark tables into padded device-ready arrays (built once)."""
    b = len(chains)
    r_pad, l_pad = static.r_pad, static.l_pad
    hps = np.zeros((b, r_pad, static.config_dim), np.float32)
    curves = np.full((b, r_pad, l_pad), np.nan, np.float32)
    fid = np.ones((b, r_pad, l_pad), np.float32)
    b_valid = np.zeros((b, r_pad), np.int32)
    n_cfg = np.zeros(b, np.int32)
    b_max = np.zeros(b, np.int32)
    c_floor = np.ones(b, np.float32)
    for i, (task, _seed) in enumerate(chains):
        r, sl = task.curves.shape
        hps[i, :r, : static.n_hps] = task.hps
        curves[i, :r, :sl] = task.curves
        f = np.asarray(task.fidelities, np.float32)
        fid[i, :r, :sl] = f[None, :] if f.ndim == 1 else f
        lens = _valid_lengths(task)
        b_valid[i, :r] = lens
        n_cfg[i], b_max[i] = r, sl
        # Regularize horizon costs with the smallest positive fidelity increment.
        sc_real = np.diff(f, axis=-1, prepend=0.0).clip(min=0.0)
        pos = sc_real[sc_real > 0]
        if pos.size:
            c_floor[i] = pos.min()
    y_star = np.nanmax(np.nan_to_num(curves, nan=-np.inf), axis=(1, 2))
    if static.use_table_costs:
        step_cost = np.zeros((b, r_pad, l_pad), np.float32)
        for i, (task, _seed) in enumerate(chains):
            if task.costs is None:
                raise ValueError(f"{task.name}: exact table-cost mode needs BenchmarkTask.costs")
            r, sl = task.curves.shape
            step_cost[i, :r, :sl] = np.nan_to_num(task.costs / static.cost_scale).astype(np.float32)
    else:
        # Keep the scan carry float32 even when JAX x64 is enabled.
        step_cost = np.diff(fid, axis=2, prepend=0.0).clip(min=0.0).astype(np.float32)
    return {
        "hps": hps,
        "curves": np.nan_to_num(curves),
        "fid": fid,
        "step_cost": step_cost,
        "b_valid": b_valid,
        "n_cfg": n_cfg,
        "b_max": b_max,
        "c_floor": c_floor,
    }, y_star


def _present(static: _Static, srt_y, real, f_best):
    """Map context responses and the incumbent into the scorer's value band.

    fixed maps [0, 1] affinely. adaptive uses a lower quantile and the observed
    maximum as bounds, matching MetricMap. With no observed range, use the
    band midpoint. Regret and recorded responses stay in the original units.
    """
    if static.value_map == "off":
        return srt_y, f_best
    lo, hi = DEFAULT_BAND
    if static.value_map == "fixed":
        return lo + (hi - lo) * srt_y, lo + (hi - lo) * f_best
    srt = jnp.sort(jnp.where(real, srt_y, jnp.inf), axis=1)  # real values first, ascending
    n = real.sum(axis=1)
    take = lambda i: jnp.take_along_axis(  # noqa: E731
        srt, jnp.clip(i, 0, srt.shape[1] - 1)[:, None], axis=1
    )[:, 0]
    # numpy-linear quantile over the first n entries, so this matches MetricMap exactly
    pos = 0.05 * jnp.maximum(n - 1, 0).astype(jnp.float32)
    i0 = jnp.floor(pos).astype(jnp.int32)
    frac = pos - i0.astype(jnp.float32)
    worst = take(i0) * (1.0 - frac) + take(i0 + 1) * frac
    best = take(jnp.maximum(n - 1, 0))
    span = best - worst
    ok = span > 1e-9 * jnp.maximum(jnp.abs(best), 1.0)
    scale = jnp.where(ok, (hi - lo) / jnp.where(ok, span, 1.0), 0.0)
    mid = 0.5 * (lo + hi)  # no scale yet (one point, or every config identical)
    fit = lambda v, w, sc, o: jnp.clip(  # noqa: E731
        jnp.where(o, lo + (v - w) * sc, mid), 0.0, 1.0
    )
    y2 = jnp.where(real, fit(srt_y, worst[:, None], scale[:, None], ok[:, None]), 0.0)
    return y2, fit(f_best, worst, scale, ok)


def _build_scan(graphdef, criterion, borders, static: _Static, score_fn):
    """The jitted whole-run function.

    Model state is an argument (one executable serves every checkpoint of an architecture);
    everything shape-like is pinned by ``static``.
    """
    r_pad = static.r_pad
    it_end = static.it_end if static.it_end is not None else static.n_iters
    ctx_pad = it_end
    solo_key = jax.random.split(jax.random.fold_in(jax.random.key(static.feature_seed), 0), 1)[0]

    def model_scores(state, t, point_cfg, point_fid, y, sep, params, it):
        model = nnx.merge(graphdef, state)
        keys = jnp.broadcast_to(solo_key, (point_cfg.shape[0],))
        logits = model(
            t["hps"], static.n_hps, point_cfg, point_fid, y, sep, feature_key=keys
        ).astype(jnp.float32)
        probs = criterion.probs(logits[:, ctx_pad:])
        return acquisition.exceedance_prob(probs, borders, params)

    score = score_fn if score_fn is not None else model_scores

    def body(state, t, chain_keys, carry, it):
        n_obs, f_best, ctx_row, ctx_lvl, ctx_y, ctx_key, cum_cost, alive = carry
        bsz = n_obs.shape[0]
        k_h, k_tau, k_tie = jax.vmap(lambda k: jax.random.split(jax.random.fold_in(k, it), 3))(
            chain_keys
        ).swapaxes(0, 1)

        h = jax.vmap(lambda k, bm: jax.random.randint(k, (), 1, bm + 1))(k_h, t["b_max"])
        # Sort by configuration, then level. Empty slots sort after the observations.
        order = jnp.argsort(ctx_key, axis=1)
        srt_row = jnp.take_along_axis(ctx_row, order, axis=1)
        srt_lvl = jnp.take_along_axis(ctx_lvl, order, axis=1)
        srt_y = jnp.take_along_axis(ctx_y, order, axis=1)

        # Map observations and incumbent before drawing the acquisition threshold.
        # Acquisition stays in mapped model units, including the upper endpoint 1.
        # Recorded responses and regret retain their original benchmark units.
        sep = it
        srt_y, f_best_p = _present(static, srt_y, jnp.arange(srt_y.shape[1])[None, :] < sep, f_best)
        tau = 10.0 ** jax.vmap(
            lambda k: jax.random.uniform(k, (), minval=-4.0, maxval=-1.0)
        )(k_tau)
        params = f_best_p + tau * (1.0 - f_best_p)

        target = jnp.clip(jnp.minimum(n_obs + h[:, None], t["b_valid"]) - 1, 0)
        point_cfg = jnp.concatenate(
            [srt_row, jnp.broadcast_to(jnp.arange(r_pad, dtype=jnp.int32), (bsz, r_pad))], axis=1
        )
        lvl_all = jnp.concatenate([srt_lvl, target], axis=1)
        fid_rows = jnp.take_along_axis(t["fid"], point_cfg[..., None], axis=1)  # (B, P, L_pad)
        point_fid = jnp.take_along_axis(fid_rows, lvl_all[..., None], axis=2)[..., 0]
        y_pts = jnp.concatenate([srt_y, jnp.zeros((bsz, r_pad), jnp.float32)], axis=1)

        scores = score(state, t, point_cfg, point_fid, y_pts, sep, params, it)
        if static.cost_cooled:
            fid_tgt = jnp.take_along_axis(t["fid"], target[..., None], axis=2)[..., 0]
            fid_obs = jnp.take_along_axis(t["fid"], jnp.clip(n_obs - 1, 0)[..., None], axis=2)[..., 0]
            cost_h = jnp.clip(fid_tgt - jnp.where(n_obs > 0, fid_obs, 0.0), 0.0, None)
            alpha = jnp.clip((static.budget - cum_cost) / static.budget, 0.0, 1.0)[:, None]
            scores = scores / (cost_h + t["c_floor"][:, None]) ** alpha
        candidate = (n_obs < t["b_valid"]) & (jnp.arange(r_pad)[None, :] < t["n_cfg"][:, None])
        scores = jnp.where(candidate, scores, -jnp.inf)

        # Break ties uniformly.
        ties = scores >= (scores.max(axis=1, keepdims=True) - 1e-12)
        chosen = jnp.argmax(
            jnp.where(ties, jax.vmap(lambda k: jax.random.uniform(k, (r_pad,)))(k_tie), -1.0),
            axis=1,
        ).astype(jnp.int32)

        batch_idx = jnp.arange(bsz)
        level = jnp.clip(n_obs[batch_idx, chosen], 0, static.l_pad - 1)
        y_val = t["curves"][batch_idx, chosen, level]
        proposed_step_cost = t["step_cost"][batch_idx, chosen, level]
        proposed_cum = cum_cost + proposed_step_cost
        accepted = alive
        if static.exact_cap:
            accepted &= proposed_cum <= static.budget

        # Rejected proposals leave the context and incumbent unchanged.
        ctx_row2 = ctx_row.at[:, it].set(jnp.where(accepted, chosen, 0))
        ctx_lvl2 = ctx_lvl.at[:, it].set(jnp.where(accepted, level, 0))
        ctx_y2 = ctx_y.at[:, it].set(jnp.where(accepted, y_val, 0.0))
        key_new = chosen.astype(jnp.float32) * (static.l_pad + 1) + level
        ctx_key2 = ctx_key.at[:, it].set(jnp.where(accepted, key_new, jnp.inf))
        n_obs2 = n_obs.at[batch_idx, chosen].add(accepted.astype(n_obs.dtype))
        f_best2 = jnp.where(accepted, jnp.maximum(f_best, y_val), f_best)
        cum2 = jnp.where(
            accepted,
            proposed_cum,
            cum_cost,
        )
        alive2 = accepted & (n_obs2 < t["b_valid"]).any(axis=1)
        if static.budget is not None and not static.exact_cap:
            alive2 &= proposed_cum < static.budget

        ys = (
            jnp.where(alive, chosen, 0),
            jnp.where(alive, level + 1, 0),
            jnp.where(alive, y_val, 0.0),
            f_best2,
            cum2,
            alive,  # this iteration recorded?
            accepted,
            proposed_step_cost,
            proposed_cum,
        )
        return (n_obs2, f_best2, ctx_row2, ctx_lvl2, ctx_y2, ctx_key2, cum2, alive2), ys

    @jax.jit
    def run(state, tables, chain_keys, init_carry):
        step = lambda c, it: body(state, tables, chain_keys, c, it)  # noqa: E731
        if static.exact_cap:
            # Keep fixed output shapes, but skip inference after the last chain stops.
            def stopped(c):
                integer = jnp.zeros_like(c[0][:, 0])
                value = jnp.zeros_like(c[1])
                flag = jnp.zeros_like(c[-1])
                return c, (integer, integer, value, c[1], c[6], flag, flag, value, c[6])

            def step(c, it):
                return jax.lax.cond(
                    jnp.any(c[-1]), lambda carry: body(state, tables, chain_keys, carry, it),
                    stopped, c,
                )
        carry, ys = jax.lax.scan(step, init_carry, jnp.arange(static.it_start, it_end))
        return carry, ys

    return run


def run_freeze_thaw_scan(
    model,
    criterion,
    chains: list[tuple[BenchmarkTask, int]],
    *,
    borders: np.ndarray,
    config_dim: int,
    n_iters: int,
    acquisition_name: str = "mfpi-random",
    n_cfg_pad: int | None = None,
    budget: float | None = None,
    score_fn=None,
    value_map: str = "off",
    ctx_buckets: int = 1,
    feature_seed: int = 0,
    initial_configs: list[int] | None = None,
    exact_cap: bool = False,
    use_table_costs: bool = False,
    cost_scale: float = 1.0,
    return_proposals: bool = False,
) -> list[FreezeThawResult] | tuple[list[FreezeThawResult], list[dict[str, np.ndarray]]]:
    """Return trajectories for B independent BO chains.

    Chains share HP count. n_cfg_pad fixes capacity across batches.
    score_fn can replace model scoring, for example in scheduler tests.
    ctx_buckets divides the run into segments with smaller context capacities;
    changing it can change compiled arithmetic and subsequent BO decisions.
    """
    if acquisition_name not in ("mfpi-random", "mfpi-cost-cooled"):
        raise ValueError(f"unknown acquisition {acquisition_name!r}")
    if ctx_buckets < 1 or ctx_buckets >= n_iters:
        raise ValueError(f"ctx_buckets must be in [1, n_iters), got {ctx_buckets}")
    if acquisition_name == "mfpi-cost-cooled" and budget is None:
        raise ValueError("mfpi-cost-cooled needs a budget: alpha_t = (budget - cum_cost)/budget")
    if budget is not None and (not np.isfinite(budget) or budget <= 0):
        raise ValueError("budget must be finite and positive")
    if exact_cap and budget is None:
        raise ValueError("exact_cap needs a budget")
    if use_table_costs and not cost_scale > 0:
        raise ValueError(f"cost_scale must be positive, got {cost_scale}")
    if initial_configs is not None and len(initial_configs) != len(chains):
        raise ValueError("initial_configs must have one entry per chain")
    if value_map not in ("off", "fixed", "adaptive"):
        raise ValueError(f"value_map must be 'off', 'fixed' or 'adaptive', not {value_map!r}")
    n_hps = {t.hps.shape[1] for t, _ in chains}
    if len(n_hps) != 1:
        raise ValueError(f"batched chains must share n_hps, got {sorted(n_hps)}")
    n_hps = n_hps.pop()
    if n_hps > config_dim:
        raise ValueError(f"{n_hps} HPs > model config_dim {config_dim}")

    r_pad = max(n_cfg_pad or 0, max(t.curves.shape[0] for t, _ in chains))
    static = _Static(
        n_iters=n_iters,
        r_pad=r_pad,
        l_pad=max(t.curves.shape[1] for t, _ in chains),
        n_hps=n_hps,
        config_dim=config_dim,
        budget=budget,
        feature_seed=feature_seed,
        exact_cap=exact_cap,
        use_table_costs=use_table_costs,
        cost_scale=cost_scale,
        cost_cooled=acquisition_name == "mfpi-cost-cooled",
        value_map=value_map,
    )
    tables_np, y_star = _pack_tables(chains, static)

    # Split the remaining iterations evenly, growing context capacity at each boundary.
    ends = [1 + ((n_iters - 1) * (j + 1)) // ctx_buckets for j in range(ctx_buckets)]
    segments = []
    start = 1
    for end in ends:
        if end > start:
            segments.append((start, end))
            start = end

    # Use the scalar reference's initialization draw.
    b = len(chains)
    ctx_pad = segments[0][1]
    n_obs = np.zeros((b, static.r_pad), np.int32)
    ctx_row = np.zeros((b, ctx_pad), np.int32)
    ctx_lvl = np.zeros((b, ctx_pad), np.int32)
    ctx_y = np.zeros((b, ctx_pad), np.float32)
    ctx_key = np.full((b, ctx_pad), np.inf, np.float32)
    f_best = np.zeros(b, np.float32)
    cum_cost = np.zeros(b, np.float32)
    init_rows = []
    for i, (_task, seed) in enumerate(chains):
        first = (
            int(initial_configs[i])
            if initial_configs is not None
            else int(np.random.default_rng([seed, 0]).integers(tables_np["n_cfg"][i]))
        )
        if not 0 <= first < tables_np["n_cfg"][i]:
            raise ValueError(f"{_task.name}: initial configuration {first} outside candidate pool")
        n_obs[i, first] = 1
        ctx_row[i, 0] = first
        ctx_y[i, 0] = tables_np["curves"][i, first, 0]
        ctx_key[i, 0] = first * (static.l_pad + 1)
        f_best[i] = tables_np["curves"][i, first, 0]
        cum_cost[i] = tables_np["step_cost"][i, first, 0]
        if exact_cap and cum_cost[i] > budget:
            raise ValueError(f"{_task.name}: initial evaluation exceeds the budget")
        init_rows.append((first, 1, f_best[i], f_best[i], y_star[i] - f_best[i], cum_cost[i]))
    alive = (n_obs < tables_np["b_valid"]).any(axis=1)
    if budget is not None and not exact_cap:
        alive &= cum_cost < budget
    init_carry = (
        jnp.asarray(n_obs),
        jnp.asarray(f_best),
        jnp.asarray(ctx_row),
        jnp.asarray(ctx_lvl),
        jnp.asarray(ctx_y),
        jnp.asarray(ctx_key),
        jnp.asarray(cum_cost),
        jnp.asarray(alive),
    )

    if model is not None:
        # Inference needs no backward-pass rematerialization. Nested remat transforms
        # would also merge NNX state at the wrong trace level inside the scan.
        model.remat = False
        model.config_encoder.remat = False
    graphdef, state = nnx.split(model) if model is not None else (None, None)

    border_array = np.ascontiguousarray(borders)
    border_key = (border_array.dtype.str, border_array.shape, border_array.tobytes())

    def resolve_run(seg_static: _Static):
        # Reuse the scorer object across chunks so compatible shapes share a JIT.
        # Model state is a runtime argument. Rebuild when the graph definition
        # changes; keep scorer and criterion references to prevent reuse of their ids.
        if score_fn is not None:
            cache_key = (id(criterion), id(score_fn), seg_static, border_key)
        else:
            cache_key = (id(criterion), seg_static, border_key)
        hit = _SCAN_CACHE.get(cache_key)
        if hit is None or hit[1] != graphdef:
            hit = (
                (criterion, score_fn),  # held alive so the id()s cannot be reused
                graphdef,
                _build_scan(graphdef, criterion, jnp.asarray(borders), seg_static, score_fn),
            )
            _SCAN_CACHE[cache_key] = hit
        return hit[2]

    tables = {k: jnp.asarray(v) for k, v in tables_np.items()}
    chain_keys = jnp.stack([jax.random.key(seed) for _, seed in chains])
    carry, ys_parts = init_carry, []
    for seg_start, seg_end in segments:
        want = seg_end
        extra = want - carry[2].shape[1]
        if extra:  # widen the context buffers to this segment's padding
            grow = lambda a, v, e=extra: jnp.pad(a, ((0, 0), (0, e)), constant_values=v)  # noqa: E731
            carry = (
                carry[0],
                carry[1],
                grow(carry[2], 0),
                grow(carry[3], 0),
                grow(carry[4], 0.0),
                grow(carry[5], jnp.inf),  # empty slots sort past the context split
                carry[6],
                carry[7],
            )
        run = resolve_run(dataclasses.replace(static, it_start=seg_start, it_end=seg_end))
        carry, ys_seg = run(state, tables, chain_keys, carry)
        ys_parts.append(ys_seg)
    ys = [np.concatenate([np.asarray(p[f]) for p in ys_parts]) for f in range(9)]
    chosen, budget_r, observed, f_best_r, cost_r, valid, accepted, step_cost_r, proposed_cum = ys

    results = []
    for i in range(b):
        keep = accepted[:, i]
        rows = [init_rows[i]] + [
            (
                int(chosen[j, i]),
                int(budget_r[j, i]),
                float(observed[j, i]),
                float(f_best_r[j, i]),
                float(y_star[i] - f_best_r[j, i]),
                float(cost_r[j, i]),
            )
            for j in range(len(keep))
            if keep[j]
        ]
        cols = list(zip(*rows, strict=True))
        results.append(FreezeThawResult(*(np.asarray(c) for c in cols)))
    if not return_proposals:
        return results
    proposals = []
    for i in range(b):
        proposals.append(
            {
                "chosen": chosen[:, i],
                "budget": budget_r[:, i],
                "observed": observed[:, i],
                "accepted": accepted[:, i],
                "proposed": valid[:, i],
                "step_cost": step_cost_r[:, i],
                "proposed_cumulative_cost": proposed_cum[:, i],
            }
        )
    return results, proposals
