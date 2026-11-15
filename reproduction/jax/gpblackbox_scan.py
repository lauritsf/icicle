"""Run the full-curve GP baseline on device.

Uses reproduction.baselines.gp_blackbox for final values, costs and initialization.
The GP uses float32; the reference uses float64. Close acquisition
scores can change order. Context has n_iters slots, with an identity covariance
block for unused rows.
"""

import jax
import jax.numpy as jnp
import numpy as np

from reproduction.bench.task import BenchmarkTask
from reproduction.baselines.gp_blackbox import RNG_CONST, blackbox_view
from reproduction.baselines.gp_ft import N_INIT, NOISE
from reproduction.baselines.scheduler import FreezeThawResult
from reproduction.jax.gpft_scan import _matern52, _log_ei

_CACHE: dict[tuple, object] = {}


def _build(n_iters: int, g_pad: int, n_init: int):
    def chain_step(carry, tables, t):
        evaluated, f_best, cum_cost, alive, ctx_cfg = carry
        live = jnp.arange(n_iters) < t  # (P,) context slots revealed so far

        # per-fit standardization over the live prefix
        ctx_y = tables["final_y"][ctx_cfg]  # (P,)
        n_live = jnp.maximum(t, 1).astype(jnp.float32)
        mean = jnp.where(live, ctx_y, 0.0).sum() / n_live
        std = jnp.sqrt(jnp.where(live, (ctx_y - mean) ** 2, 0.0).sum() / n_live)
        std = jnp.maximum(std, 1e-8)
        yz = jnp.where(live, (ctx_y - mean) / std, 0.0)

        x = tables["hps"][ctx_cfg]  # (P, n_hps)
        cand_ok = ~evaluated & (jnp.arange(g_pad) < tables["n_grp"])
        k = _matern52(x, x) + NOISE * jnp.eye(n_iters)
        mask2 = live[:, None] & live[None, :]
        k = jnp.where(mask2, k, jnp.eye(n_iters))  # unused slots: unit diagonal, defined chol
        chol = jnp.linalg.cholesky(k)
        alpha = jax.scipy.linalg.cho_solve((chol, True), yz)
        ks = jnp.where(live[None, :], _matern52(tables["hps"], x), 0.0)  # (G, P)
        mu = ks @ alpha
        v = jax.scipy.linalg.solve_triangular(chol, ks.T, lower=True)  # (P, G)
        var = jnp.maximum(1.0 - (v**2).sum(0), 1e-12)

        # plain EI against the incumbent final value, log-EI ranked
        sd = jnp.sqrt(var)
        u = (mu - (f_best - mean) / std) / sd
        ei = jnp.where(cand_ok, _log_ei(u, sd), -jnp.inf)

        alive &= cand_ok.any()
        cfg = jnp.argmax(ei).astype(jnp.int32)  # first max: the reference's np.argmax
        cfg = jnp.clip(cfg, 0, g_pad - 1)
        y = tables["final_y"][cfg]
        evaluated = evaluated.at[cfg].set(evaluated[cfg] | alive)
        f_best = jnp.where(alive, jnp.maximum(f_best, y), f_best)
        cum_cost = cum_cost + jnp.where(alive, tables["eval_cost"][cfg], 0.0)
        slot = jnp.clip(t, 0, n_iters - 1)
        ctx_cfg = jnp.where(alive, ctx_cfg.at[slot].set(cfg), ctx_cfg)

        carry = (evaluated, f_best, cum_cost, alive, ctx_cfg)
        return carry, (cfg, tables["final_budget"][cfg], y, f_best, cum_cost, alive)

    step = jax.vmap(chain_step, in_axes=(0, 0, None))

    @jax.jit
    def run(tables, init):
        body = lambda c, t: step(c, tables, t)  # noqa: E731
        _, ys = jax.lax.scan(body, init, jnp.arange(n_init, n_iters))
        return ys

    return run


def _collect(ys, y_star, b) -> list[FreezeThawResult]:
    """``baseline_scan._collect`` plus the cost column this runner emits."""
    chosen, budget, observed, f_best, cost, alive = (np.asarray(a) for a in ys)
    results = []
    for i in range(b):
        rows = [
            (
                int(chosen[j, i]),
                int(budget[j, i]),
                float(observed[j, i]),
                float(f_best[j, i]),
                float(y_star[i] - f_best[j, i]),
                float(cost[j, i]),
            )
            for j in range(len(alive))
            if alive[j, i]
        ]
        cols = list(zip(*rows, strict=True)) if rows else [()] * 6
        results.append(FreezeThawResult(*(
            np.asarray(c, dtype=np.int64 if j < 2 else np.float64)
            for j, c in enumerate(cols)
        )))
    return results


def run_gpblackbox_scan(
    chains: list[tuple[BenchmarkTask, int]],
    *,
    n_iters: int,
    n_cfg_pad: int | None = None,
) -> list[FreezeThawResult]:
    """B black-box-GP chains, one compiled scan; see the module docstring."""
    if not chains or n_iters < 1:
        raise ValueError("provide at least one chain and a positive iteration count")
    n_init = min(N_INIT, n_iters, *(len(task.curves) for task, _ in chains))
    n_hps = {t.hps.shape[1] for t, _ in chains}
    if len(n_hps) != 1:
        raise ValueError(f"batched chains must share n_hps, got {sorted(n_hps)}")
    n_hps = n_hps.pop()
    b = len(chains)
    g_pad = max(n_cfg_pad or 0, max(t.curves.shape[0] for t, _ in chains))

    hps = np.zeros((b, g_pad, n_hps), np.float32)
    final_y = np.zeros((b, g_pad), np.float32)
    final_budget = np.zeros((b, g_pad), np.int32)
    eval_cost = np.zeros((b, g_pad), np.float32)
    n_grp = np.zeros(b, np.int32)
    y_star = np.zeros(b, np.float64)
    evaluated = np.zeros((b, g_pad), bool)
    ctx_cfg = np.zeros((b, n_iters), np.int32)
    f_best = np.zeros(b, np.float32)
    cum_cost = np.zeros(b, np.float32)
    init_rows = []
    for i, (task, seed) in enumerate(chains):
        fy, fb, ec = blackbox_view(task)
        g = len(fy)
        hps[i, :g] = task.hps
        final_y[i, :g], final_budget[i, :g], eval_cost[i, :g] = fy, fb, ec
        n_grp[i] = g
        y_star[i] = float(np.nanmax(task.curves))
        # the reference's init draw, on the host: 3 distinct configs, fully evaluated
        rows_i = []
        for j, cfg in enumerate(np.random.default_rng([seed, RNG_CONST]).permutation(g)[:n_init]):
            y = float(fy[cfg])
            evaluated[i, cfg] = True
            ctx_cfg[i, j] = cfg
            f_best[i] = max(f_best[i], y) if j else y
            cum_cost[i] += ec[cfg]
            rows_i.append(
                (
                    int(cfg),
                    int(fb[cfg]),
                    y,
                    float(f_best[i]),
                    float(y_star[i] - f_best[i]),
                    float(cum_cost[i]),
                )
            )
        init_rows.append(rows_i)

    key = ("gpbb", n_iters, g_pad, n_hps, n_init)
    run = _CACHE.get(key)
    if run is None:
        run = _CACHE[key] = _build(n_iters, g_pad, n_init)

    tables = {
        "hps": jnp.asarray(hps),
        "final_y": jnp.asarray(final_y),
        "final_budget": jnp.asarray(final_budget),
        "eval_cost": jnp.asarray(eval_cost),
        "n_grp": jnp.asarray(n_grp),
    }
    init = (
        jnp.asarray(evaluated),
        jnp.asarray(f_best),
        jnp.asarray(cum_cost),
        jnp.ones(b, bool),
        jnp.asarray(ctx_cfg),
    )
    ys = run(tables, init)
    scan_results = _collect(ys, y_star, b)
    results = []
    for i, res in enumerate(scan_results):
        cols = list(zip(*init_rows[i], strict=True))
        head = FreezeThawResult(*(np.asarray(c) for c in cols))
        results.append(
            FreezeThawResult(
                *(
                    np.concatenate([getattr(head, f), getattr(res, f)])
                    for f in ("chosen", "budget", "observed", "f_best", "regret", "cost")
                )
            )
        )
    return results
