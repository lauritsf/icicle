"""Run ASHA and Hyperband as batched JAX scans.

Use the same initialization permutations as the NumPy reference in
reproduction.baselines.scheduler. Grouped curves must be averaged before calling.

Shapes: B chains, G configurations, L levels, K rungs, N cohort slots.
"""

import jax
import jax.numpy as jnp
import numpy as np

from reproduction.bench.task import BenchmarkTask
from reproduction.baselines.scheduler import FreezeThawResult, _valid_lengths, rung_ladder

_CACHE: dict[tuple, object] = {}


def _pack(chains, order_const: int, g_pad: int, k_pad: int, eta: int):
    """Per-chain tables + the host-side permutation draw shared with the numpy reference."""
    b = len(chains)
    l_pad = max(t.curves.shape[1] for t, _ in chains)
    curves = np.zeros((b, g_pad, l_pad), np.float32)
    b_valid = np.zeros((b, g_pad), np.int32)
    ladder = np.zeros((b, k_pad), np.int32)
    n_rungs = np.zeros(b, np.int32)
    n_grp = np.zeros(b, np.int32)
    s_max = np.zeros(b, np.int32)
    order = np.zeros((b, g_pad), np.int32)
    y_star = np.zeros(b, np.float64)
    for i, (task, seed) in enumerate(chains):
        g, sl = task.curves.shape
        curves[i, :g, :sl] = np.nan_to_num(task.curves)
        b_valid[i, :g] = _valid_lengths(task)
        lad = rung_ladder(sl, eta)
        ladder[i, : len(lad)] = lad
        ladder[i, len(lad) :] = lad[-1]  # clamp: promotion targets past the top are inert
        n_rungs[i], n_grp[i], s_max[i] = len(lad), g, len(lad) - 1
        order[i, :g] = np.random.default_rng([seed, order_const]).permutation(g)
        y_star[i] = float(np.nanmax(task.curves))
    return {
        "curves": curves,
        "b_valid": b_valid,
        "ladder": ladder,
        "n_rungs": n_rungs,
        "n_grp": n_grp,
        "s_max": s_max,
        "order": order,
    }, y_star


def _collect(ys, y_star, b) -> list[FreezeThawResult]:
    chosen, budget, observed, f_best, alive = (np.asarray(a) for a in ys)
    results = []
    for i in range(b):
        keep = alive[:, i]
        rows = [
            (
                int(chosen[j, i]),
                int(budget[j, i]),
                float(observed[j, i]),
                float(f_best[j, i]),
                float(y_star[i] - f_best[j, i]),
            )
            for j in range(len(keep))
            if keep[j]
        ]
        cols = list(zip(*rows, strict=True)) if rows else [()] * 5
        results.append(FreezeThawResult(*(
            np.asarray(c, dtype=np.int64 if j < 2 else np.float64)
            for j, c in enumerate(cols)
        )))
    return results


def _build_asha(n_iters: int, g_pad: int, k_pad: int, eta: int, variant: str):
    """One compiled ASHA scan for this (shape, variant) class."""

    def promote(n_obs, tables, book):
        """Deepest-rung promotable config for both quota modes, one sorted pass per rung.

        In (value desc, index asc) order the completed members occupy the prefix, so
        rank == sorted position and both eligibility windows read off the same argsort.
        """
        not_exhausted = n_obs < tables["b_valid"]  # (G,)
        cand, cand_rung = jnp.int32(-1), jnp.int32(0)
        free, free_rung = jnp.int32(-1), jnp.int32(0)
        pos = jnp.arange(g_pad)
        for k in range(k_pad - 2, -1, -1):  # deepest promotable rung first, static unroll
            if variant == "li":
                val_mat, promoted = book
                vals = val_mat[k]  # (G,) -inf = never completed here
                members = vals > -jnp.inf
                blocked = promoted[k]
            else:  # neps: a config is a member only of its highest completed rung.
                # Exhausted configs linger as members (inflating the quota, as NePS's
                # observation history would) but are never themselves promoted.
                cur_rung, rung_val = book
                members = cur_rung == k
                vals = jnp.where(members, rung_val, -jnp.inf)
                blocked = jnp.zeros_like(members)
            count = members.sum()
            srt = jnp.argsort(-vals)  # stable: value desc, ties to the lower config
            open_s = members[srt] & ~blocked[srt] & not_exhausted[srt] & (k < tables["n_rungs"] - 1)
            q_elig = open_s & (pos < count // eta)
            f_elig = open_s & (pos < count)
            qhit = q_elig.any() & (cand < 0)
            fhit = f_elig.any() & (free < 0)
            cand = jnp.where(qhit, srt[jnp.argmax(q_elig)].astype(jnp.int32), cand)
            cand_rung = jnp.where(qhit, jnp.int32(k + 1), cand_rung)
            free = jnp.where(fhit, srt[jnp.argmax(f_elig)].astype(jnp.int32), free)
            free_rung = jnp.where(fhit, jnp.int32(k + 1), free_rung)
        return cand, cand_rung, free, free_rung

    def chain_step(carry, tables):
        n_obs, active, target, next_pos, f_best, alive, book = carry

        # pick a new active config when none is training: quota promotion, else a fresh
        # config off the permutation, else (pool exhausted) quota-free promotion
        need = active < 0
        cand, cand_rung, free, free_rung = promote(n_obs, tables, book)
        has_fresh = next_pos < tables["n_grp"]
        fresh = tables["order"][jnp.clip(next_pos, 0, g_pad - 1)]
        picked = jnp.where(cand >= 0, cand, jnp.where(has_fresh, fresh, free))
        picked_rung = jnp.where(cand >= 0, cand_rung, jnp.where(has_fresh, 0, free_rung))
        active = jnp.where(need, picked, active)
        target = jnp.where(need, picked_rung, target)
        next_pos += jnp.where(need & (cand < 0) & has_fresh, 1, 0)
        # the promoted-once-per-rung ledger records every promotion, quota or quota-free
        if variant == "li":
            val_mat, promoted = book
            promo = need & (picked >= 0) & ~(has_fresh & (cand < 0))
            promoted = promoted.at[picked_rung - 1, jnp.clip(picked, 0, g_pad - 1)].set(
                promoted[picked_rung - 1, jnp.clip(picked, 0, g_pad - 1)] | promo
            )
            book = (val_mat, promoted)
        alive &= active >= 0

        # one training step for the active config
        cfg = jnp.clip(active, 0, g_pad - 1)
        y = tables["curves"][cfg, jnp.clip(n_obs[cfg], 0, tables["curves"].shape[1] - 1)]
        n_obs = n_obs.at[cfg].add(jnp.where(alive, 1, 0))
        f_best = jnp.where(alive, jnp.maximum(f_best, y), f_best)
        budget = n_obs[cfg]

        # rung completion: y IS the at-rung value (completion lands on the just-revealed step)
        stop = jnp.minimum(tables["ladder"][jnp.clip(target, 0, k_pad - 1)], tables["b_valid"][cfg])
        done = alive & (budget >= stop)
        if variant == "li":
            val_mat, promoted = book
            val_mat = val_mat.at[jnp.clip(target, 0, k_pad - 1), cfg].set(
                jnp.where(done, y, val_mat[jnp.clip(target, 0, k_pad - 1), cfg])
            )
            book = (val_mat, promoted)
        else:
            cur_rung, rung_val = book
            cur_rung = jnp.where(done, cur_rung.at[cfg].set(target), cur_rung)
            rung_val = jnp.where(done, rung_val.at[cfg].set(y), rung_val)
            book = (cur_rung, rung_val)
        active = jnp.where(done, -1, active)

        return (n_obs, active, target, next_pos, f_best, alive, book), (
            cfg,
            budget,
            y,
            f_best,
            alive,
        )

    step = jax.vmap(chain_step)

    @jax.jit
    def run(tables, init):
        body = lambda c, _t: step(c, tables)  # noqa: E731
        _, ys = jax.lax.scan(body, init, None, length=n_iters)
        return ys

    return run


def run_asha_scan(
    chains: list[tuple[BenchmarkTask, int]],
    *,
    n_iters: int,
    eta: int = 3,
    variant: str = "li",
    n_cfg_pad: int | None = None,
) -> list[FreezeThawResult]:
    """B ASHA chains, one compiled scan -- exactly paired with ``run_successive_halving``."""
    if variant not in ("li", "neps"):
        raise ValueError(f"unknown ASHA variant {variant!r}")
    b = len(chains)
    g_pad = max(n_cfg_pad or 0, max(t.curves.shape[0] for t, _ in chains))
    k_pad = max(len(rung_ladder(t.curves.shape[1], eta)) for t, _ in chains)
    tables_np, y_star = _pack(chains, 313131, g_pad, k_pad, eta)

    key = ("asha", n_iters, g_pad, k_pad, tables_np["curves"].shape[2], eta, variant)
    run = _CACHE.get(key)
    if run is None:
        run = _CACHE[key] = _build_asha(n_iters, g_pad, k_pad, eta, variant)

    book = (
        (jnp.full((b, k_pad, g_pad), -jnp.inf), jnp.zeros((b, k_pad, g_pad), bool))
        if variant == "li"
        else (jnp.full((b, g_pad), -1, jnp.int32), jnp.full((b, g_pad), -jnp.inf))
    )
    init = (
        jnp.zeros((b, g_pad), jnp.int32),
        jnp.full(b, -1, jnp.int32),
        jnp.zeros(b, jnp.int32),
        jnp.zeros(b, jnp.int32),
        jnp.zeros(b, jnp.float32),
        jnp.ones(b, bool),
        book,
    )
    ys = run({k: jnp.asarray(v) for k, v in tables_np.items()}, init)
    return _collect(ys, y_star, b)


def _build_hyperband(n_iters: int, g_pad: int, k_pad: int, n_pad: int, eta: int):
    """One compiled Hyperband scan.

    Control state advances inside a bounded while_loop because rung completion, promotion
    and cohort draws cost no training steps.
    """

    def draw(next_pos, n_obs, tables, n_want):
        """The reference's ``fresh`` draw.

        Scans at most n_grp cyclic permutation slots from next_pos and takes the first
        n_want configs still trainable.
        """
        offs = jnp.arange(g_pad)
        idx = (next_pos + offs) % jnp.maximum(tables["n_grp"], 1)
        cfgs = tables["order"][idx]  # (G,)
        elig = (offs < tables["n_grp"]) & (n_obs[cfgs] < tables["b_valid"][cfgs])
        picks = jnp.cumsum(elig)  # k-th eligible has picks == k
        taken = jnp.minimum(n_want, elig.sum())
        slot = jnp.arange(n_pad)  # (N,)
        src = jnp.argmax(elig[None, :] & (picks[None, :] == slot[:, None] + 1), axis=1)
        cohort = jnp.where(slot < taken, cfgs[src], -1)
        # scanned slots: through the n_want-th eligible if found, else the whole window
        scanned = jnp.where(taken == n_want, src[jnp.clip(n_want - 1, 0, n_pad - 1)] + 1, g_pad)
        scanned = jnp.minimum(scanned, tables["n_grp"])
        return cohort, taken, next_pos + scanned

    def control(state, tables):
        """Advance bracket/rung/cohort bookkeeping until a step is takeable or dead.

        Bounded: each pass moves (rung, bracket, empty-draw count) forward.
        """

        def rung_budget(s, rung_i):
            return tables["ladder"][jnp.clip(tables["s_max"] - s + rung_i, 0, k_pad - 1)]

        def cond(st):
            (n_obs, s, rung_i, cohort, size, pos, next_pos, empties, alive) = st
            slots = jnp.arange(n_pad)
            needs = (
                (slots >= pos)
                & (slots < size)
                & (n_obs[cohort] < jnp.minimum(rung_budget(s, rung_i), tables["b_valid"][cohort]))
            )
            return alive & ~((rung_i >= 0) & needs.any())

        def body(st):
            (n_obs, s, rung_i, cohort, size, pos, next_pos, empties, alive) = st
            needs_draw = rung_i < 0

            # -- draw a fresh cohort for bracket s
            n_want = -(-((tables["s_max"] + 1) * eta**s) // (s + 1))  # ceil, Li's sizes
            dcohort, dtaken, dpos = draw(next_pos, n_obs, tables, n_want)
            drew = needs_draw & (dtaken > 0)
            dead_cycle = empties + 1 > tables["s_max"]  # a full cycle of empty draws
            # -- or complete the current rung: rank by at-rung value, keep the top len//eta
            stops = jnp.minimum(rung_budget(s, rung_i), tables["b_valid"][cohort])
            in_cohort = jnp.arange(n_pad) < size
            vals = jnp.where(
                in_cohort,
                tables["curves"][cohort, jnp.clip(stops - 1, 0, tables["curves"].shape[1] - 1)],
                -jnp.inf,
            )
            keep = size // eta
            last_rung = rung_i >= s  # bracket s has s+1 rungs
            bracket_over = ~needs_draw & (last_rung | (keep == 0))
            srt = jnp.argsort(-vals)  # stable: ties to the lower cohort slot
            promoted_cohort = jnp.where(jnp.arange(n_pad) < keep, cohort[srt], -1)

            next_s = jnp.where(s - 1 < 0, tables["s_max"], s - 1)  # cycle, aggressive first
            adv_bracket = (needs_draw & ~drew) | bracket_over
            s = jnp.where(adv_bracket, next_s, s)
            rung_i = jnp.where(
                adv_bracket, -1, jnp.where(drew, 0, jnp.where(needs_draw, rung_i, rung_i + 1))
            )
            cohort = jnp.where(drew, dcohort, jnp.where(needs_draw, cohort, promoted_cohort))
            size = jnp.where(drew, dtaken, jnp.where(needs_draw, size, keep))
            pos = jnp.where(drew | ~needs_draw, 0, pos)
            next_pos = jnp.where(needs_draw, jnp.where(drew, dpos, next_pos), next_pos)
            empties = jnp.where(needs_draw, jnp.where(drew, 0, empties + 1), empties)
            alive &= ~(needs_draw & ~drew & dead_cycle)
            # explicit i32: python-int arithmetic weak-promotes to i64 when a same-process
            # test has enabled x64, and the while_loop carry demands exact dtype equality
            i32 = jnp.int32
            return (
                n_obs,
                s.astype(i32),
                rung_i.astype(i32),
                cohort,
                size.astype(i32),
                pos.astype(i32),
                next_pos.astype(i32),
                empties.astype(i32),
                alive,
            )

        return jax.lax.while_loop(cond, body, state)

    def full_step(carry, tables):
        inner, f_best = carry
        st = control(inner, tables)
        (n_obs, s, rung_i, cohort, size, pos, next_pos, empties, alive) = st

        # first cohort member still short of its rung budget is the active one
        budget_r = tables["ladder"][jnp.clip(tables["s_max"] - s + rung_i, 0, k_pad - 1)]
        slots = jnp.arange(n_pad)
        needs = (
            (slots >= pos)
            & (slots < size)
            & (n_obs[cohort] < jnp.minimum(budget_r, tables["b_valid"][cohort]))
        )
        pos = jnp.where(alive, jnp.argmax(needs), pos).astype(jnp.int32)
        cfg = jnp.clip(cohort[jnp.clip(pos, 0, n_pad - 1)], 0, g_pad - 1)

        y = tables["curves"][cfg, jnp.clip(n_obs[cfg], 0, tables["curves"].shape[1] - 1)]
        n_obs = n_obs.at[cfg].add(jnp.where(alive, 1, 0))
        f_best = jnp.where(alive, jnp.maximum(f_best, y), f_best)
        inner = (n_obs, s, rung_i, cohort, size, pos, next_pos, empties, alive)
        return (inner, f_best), (cfg, n_obs[cfg], y, f_best, alive)

    step = jax.vmap(full_step)

    @jax.jit
    def run(tables, init):
        body = lambda c, _t: step(c, tables)  # noqa: E731
        _, ys = jax.lax.scan(body, init, None, length=n_iters)
        return ys

    return run


def run_hyperband_scan(
    chains: list[tuple[BenchmarkTask, int]],
    *,
    n_iters: int,
    eta: int = 3,
    n_cfg_pad: int | None = None,
) -> list[FreezeThawResult]:
    """B Hyperband chains, one compiled scan -- exactly paired with ``run_hyperband``."""
    b = len(chains)
    g_pad = max(n_cfg_pad or 0, max(t.curves.shape[0] for t, _ in chains))
    k_pad = max(len(rung_ladder(t.curves.shape[1], eta)) for t, _ in chains)
    # widest cohort any bracket can request: s = s_max, ceil((s_max+1)*eta^s_max/(s_max+1))
    n_pad = int(min(g_pad, eta ** (k_pad - 1)))
    tables_np, y_star = _pack(chains, 515151, g_pad, k_pad, eta)

    key = ("hb", n_iters, g_pad, k_pad, n_pad, tables_np["curves"].shape[2], eta)
    run = _CACHE.get(key)
    if run is None:
        run = _CACHE[key] = _build_hyperband(n_iters, g_pad, k_pad, n_pad, eta)

    inner = (
        jnp.zeros((b, g_pad), jnp.int32),
        jnp.asarray(tables_np["s_max"]),  # start at the most aggressive bracket
        jnp.full(b, -1, jnp.int32),  # rung_i = -1: bracket needs its cohort drawn
        jnp.full((b, n_pad), -1, jnp.int32),
        jnp.zeros(b, jnp.int32),
        jnp.zeros(b, jnp.int32),
        jnp.zeros(b, jnp.int32),
        jnp.zeros(b, jnp.int32),
        jnp.ones(b, bool),
    )
    init = (inner, jnp.zeros(b, jnp.float32))
    ys = run({k: jnp.asarray(v) for k, v in tables_np.items()}, init)
    return _collect(ys, y_star, b)
