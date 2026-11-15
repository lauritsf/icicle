"""Run the freeze-thaw GP baseline on device.

Uses the kernel and acquisition settings in reproduction.baselines.gp_ft.
Initialization permutations come from the same NumPy RNG. GP calculations
use float32; the reference uses float64. Close acquisition scores can change order.

Context is padded to n_iters. Unused rows have an identity covariance block
so each fixed-size Cholesky factorization remains defined.
"""

import jax
import jax.numpy as jnp
import numpy as np

from reproduction.bench.task import BenchmarkTask
from reproduction.baselines.gp_ft import LENGTHSCALE, N_INIT, NOISE
from reproduction.baselines.scheduler import FreezeThawResult, _valid_lengths
from reproduction.jax.baseline_scan import _collect

_CACHE: dict[tuple, object] = {}


def _matern52(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    r = jnp.sqrt(jnp.maximum(d2, 0.0)) / LENGTHSCALE
    s5r = jnp.sqrt(5.0) * r
    return (1.0 + s5r + 5.0 / 3.0 * r**2) * jnp.exp(-s5r)


def _log_ei(u: jnp.ndarray, sd: jnp.ndarray) -> jnp.ndarray:
    """JAX log-EI with the reference's asymptotic branch below -8."""
    pdf = jnp.exp(-0.5 * u**2) / jnp.sqrt(2.0 * jnp.pi)
    cdf = 0.5 * (1.0 + jax.scipy.special.erf(u / jnp.sqrt(2.0)))
    direct = jnp.log(jnp.maximum(pdf + u * cdf, 1e-300))
    un = jnp.minimum(u, -8.0)
    asym = -0.5 * un**2 - 0.5 * jnp.log(2.0 * jnp.pi) - 2.0 * jnp.log(-un) + jnp.log1p(-3.0 / un**2)
    return jnp.log(sd) + jnp.where(u > -8.0, direct, asym)


def _build(n_iters: int, g_pad: int, l_pad: int, n_init: int):
    def chain_step(carry, tables, t):
        n_obs, f_best, alive, ctx_cfg, ctx_lvl, ctx_y = carry
        live = jnp.arange(n_iters) < t  # (P,) context slots revealed so far

        # per-fit standardization over the live prefix
        n_live = jnp.maximum(t, 1).astype(jnp.float32)
        mean = jnp.where(live, ctx_y, 0.0).sum() / n_live
        std = jnp.sqrt(jnp.where(live, (ctx_y - mean) ** 2, 0.0).sum() / n_live)
        std = jnp.maximum(std, 1e-8)
        yz = jnp.where(live, (ctx_y - mean) / std, 0.0)

        b_max = tables["b_max"].astype(jnp.float32)
        x = jnp.concatenate(
            [tables["hps"][ctx_cfg], ((ctx_lvl + 1.0) / b_max)[:, None]], axis=1
        )  # (P, n_hps + 1)
        cand_ok = (n_obs < tables["b_valid"]) & (jnp.arange(g_pad) < tables["n_grp"])
        q = jnp.concatenate(
            [tables["hps"], ((n_obs + 1.0) / b_max)[:, None]], axis=1
        )  # (G, n_hps + 1)

        k = _matern52(x, x) + NOISE * jnp.eye(n_iters)
        mask2 = live[:, None] & live[None, :]
        k = jnp.where(mask2, k, jnp.eye(n_iters))  # unused slots: unit diagonal, defined chol
        chol = jnp.linalg.cholesky(k)
        alpha = jax.scipy.linalg.cho_solve((chol, True), yz)
        ks = jnp.where(live[None, :], _matern52(q, x), 0.0)  # (G, P)
        mu = ks @ alpha
        v = jax.scipy.linalg.solve_triangular(chol, ks.T, lower=True)  # (P, G)
        var = jnp.maximum(1.0 - (v**2).sum(0), 1e-12)

        # MF-EI: incumbent AT the candidate's target level, global best as the fallback
        level_best = jnp.full(l_pad, -jnp.inf).at[ctx_lvl].max(jnp.where(live, ctx_y, -jnp.inf))
        inc = level_best[jnp.clip(n_obs, 0, l_pad - 1)]
        inc = jnp.where(inc > -jnp.inf, inc, f_best)
        inc_z = (inc - mean) / std

        sd = jnp.sqrt(var)
        u = (mu - inc_z) / sd
        ei = jnp.where(cand_ok, _log_ei(u, sd), -jnp.inf)

        alive &= cand_ok.any()
        cfg = jnp.argmax(ei).astype(jnp.int32)  # first max: the reference's np.argmax
        cfg = jnp.clip(cfg, 0, g_pad - 1)
        lvl = jnp.clip(n_obs[cfg], 0, l_pad - 1)
        y = tables["curves"][cfg, lvl]
        n_obs = n_obs.at[cfg].add(jnp.where(alive, 1, 0))
        f_best = jnp.where(alive, jnp.maximum(f_best, y), f_best)
        slot = jnp.clip(t, 0, n_iters - 1)
        ctx_cfg = jnp.where(alive, ctx_cfg.at[slot].set(cfg), ctx_cfg)
        ctx_lvl = jnp.where(alive, ctx_lvl.at[slot].set(lvl), ctx_lvl)
        ctx_y = jnp.where(alive, ctx_y.at[slot].set(y), ctx_y)

        carry = (n_obs, f_best, alive, ctx_cfg, ctx_lvl, ctx_y)
        return carry, (cfg, n_obs[cfg], y, f_best, alive)

    step = jax.vmap(chain_step, in_axes=(0, 0, None))

    @jax.jit
    def run(tables, init):
        body = lambda c, t: step(c, tables, t)  # noqa: E731
        _, ys = jax.lax.scan(body, init, jnp.arange(n_init, n_iters))
        return ys

    return run


def run_gpft_scan(
    chains: list[tuple[BenchmarkTask, int]],
    *,
    n_iters: int,
    n_cfg_pad: int | None = None,
) -> list[FreezeThawResult]:
    """B freeze-thaw-GP chains, one compiled scan; see the module docstring."""
    if not chains or n_iters < 1:
        raise ValueError("provide at least one chain and a positive iteration count")
    n_init = min(N_INIT, n_iters, *(len(task.curves) for task, _ in chains))
    n_hps = {t.hps.shape[1] for t, _ in chains}
    if len(n_hps) != 1:
        raise ValueError(f"batched chains must share n_hps, got {sorted(n_hps)}")
    n_hps = n_hps.pop()
    b = len(chains)
    g_pad = max(n_cfg_pad or 0, max(t.curves.shape[0] for t, _ in chains))
    l_pad = max(t.curves.shape[1] for t, _ in chains)

    hps = np.zeros((b, g_pad, n_hps), np.float32)
    curves = np.zeros((b, g_pad, l_pad), np.float32)
    b_valid = np.zeros((b, g_pad), np.int32)
    n_grp = np.zeros(b, np.int32)
    b_max = np.zeros(b, np.int32)
    y_star = np.zeros(b, np.float64)
    n_obs = np.zeros((b, g_pad), np.int32)
    ctx_cfg = np.zeros((b, n_iters), np.int32)
    ctx_lvl = np.zeros((b, n_iters), np.int32)
    ctx_y = np.zeros((b, n_iters), np.float32)
    f_best = np.zeros(b, np.float32)
    init_rows = []
    for i, (task, seed) in enumerate(chains):
        g, sl = task.curves.shape
        hps[i, :g] = task.hps
        curves[i, :g, :sl] = np.nan_to_num(task.curves)
        b_valid[i, :g] = _valid_lengths(task)
        n_grp[i], b_max[i] = g, sl
        y_star[i] = float(np.nanmax(task.curves))
        # the reference's init draw, on the host: 3 distinct configs, one step each
        rows_i = []
        for j, cfg in enumerate(np.random.default_rng([seed, 616161]).permutation(g)[:n_init]):
            y = float(curves[i, cfg, 0])
            ctx_cfg[i, j], ctx_lvl[i, j], ctx_y[i, j] = cfg, 0, y
            n_obs[i, cfg] = 1
            f_best[i] = max(f_best[i], y) if j else y
            rows_i.append((int(cfg), 1, y, float(f_best[i]), float(y_star[i] - f_best[i])))
        init_rows.append(rows_i)

    key = ("gpft", n_iters, g_pad, l_pad, n_hps, n_init)
    run = _CACHE.get(key)
    if run is None:
        run = _CACHE[key] = _build(n_iters, g_pad, l_pad, n_init)

    tables = {
        "hps": jnp.asarray(hps),
        "curves": jnp.asarray(curves),
        "b_valid": jnp.asarray(b_valid),
        "n_grp": jnp.asarray(n_grp),
        "b_max": jnp.asarray(b_max),
    }
    init = (
        jnp.asarray(n_obs),
        jnp.asarray(f_best),
        jnp.ones(b, bool),
        jnp.asarray(ctx_cfg),
        jnp.asarray(ctx_lvl),
        jnp.asarray(ctx_y),
    )
    ys = run(tables, init)
    scan_results = _collect(ys, y_star, b)
    results = []
    for i, res in enumerate(scan_results):
        pre = init_rows[i]
        cols = list(zip(*pre, strict=True))
        head = FreezeThawResult(*(np.asarray(c) for c in cols))
        results.append(
            FreezeThawResult(
                *(
                    np.concatenate([getattr(head, f), getattr(res, f)])
                    for f in ("chosen", "budget", "observed", "f_best", "regret")
                )
            )
        )
    return results
