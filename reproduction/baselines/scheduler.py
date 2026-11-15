"""NumPy reference freeze-thaw scheduler and tabular baselines.

Each configuration is a candidate. Advancing it reveals the next stored curve
value. The freeze-thaw loop starts with one random configuration at one step,
then draws an MFPI horizon and threshold, scores candidates and advances one.

Predictors accept context/query configuration and level indices and return
(Q, num_bars) probabilities. Random draws depend on (seed, iteration), allowing
paired comparisons given the same predictions. The accelerated implementation
is reproduction.jax.hpo_scan; these loops provide an independent test reference.
"""

import dataclasses

import numpy as np

from reproduction.bench.task import BenchmarkTask, valid_lengths
from icicle.hpo.decider import sample_mfpi_random, exceedance_prob


@dataclasses.dataclass
class FreezeThawResult:
    """Per-iteration trajectory of one BO run (arrays of length n_iters)."""

    chosen: np.ndarray  # config row advanced at each iteration
    budget: np.ndarray  # that config's budget after the step (1-based number of steps)
    observed: np.ndarray  # the curve value revealed by the step
    f_best: np.ndarray  # incumbent (best observed value) after the step
    regret: np.ndarray  # table optimum (max over all configs/budgets) minus f_best
    # cumulative fidelity-increment cost after the step (the wallclock share of the budget on
    # cumulative-walltime views). Only the scan runner fills it; the host loops leave None.
    cost: np.ndarray | None = None


def _observations(task, n_obs):
    """All observed (config, level, y) triples, config-major, prefix order."""
    cfg = np.repeat(np.arange(len(n_obs)), n_obs)
    lvl = np.concatenate([np.arange(n) for n in n_obs]) if cfg.size else np.zeros(0, int)
    return cfg.astype(np.int32), lvl.astype(np.int32), task.curves[cfg, lvl]


def _valid_lengths(task: BenchmarkTask) -> np.ndarray:
    return valid_lengths(task.curves)


def run_freeze_thaw(
    task: BenchmarkTask,
    predict_fn,
    borders: np.ndarray,
    n_iters: int,
    seed: int = 0,
) -> FreezeThawResult:
    n_cfg, b_max = task.curves.shape
    b_valid = _valid_lengths(task)
    n_obs = np.zeros(n_cfg, dtype=np.int64)  # observed prefix length per config
    y_star = float(np.nanmax(task.curves))
    rows = []

    # ifBO init: one uniformly random config, trained one step.
    first = int(np.random.default_rng([seed, 0]).integers(n_cfg))
    n_obs[first] = 1
    f_best = float(task.curves[first, 0])
    rows.append((first, 1, task.curves[first, 0], f_best, y_star - f_best))

    for it in range(1, n_iters):
        rng = np.random.default_rng([seed, it])
        h, threshold = sample_mfpi_random(rng, b_max, f_best)

        candidates = np.flatnonzero(n_obs < b_valid)  # fully-trained configs can't advance
        if candidates.size == 0:
            break
        # One query per candidate, h steps past its current budget, capped at its valid end.
        target = np.minimum(n_obs[candidates] + h, b_valid[candidates]) - 1
        ctx_cfg, ctx_lvl, ctx_y = _observations(task, n_obs)
        probs = predict_fn(
            ctx_cfg,
            ctx_lvl,
            ctx_y,
            candidates.astype(np.int32),
            target.astype(np.int32),
        )
        scores = exceedance_prob(np.asarray(probs), borders, threshold)

        # Choose uniformly among tied scores to avoid favoring low configuration indices.
        top = np.flatnonzero(scores >= scores.max() - 1e-12)
        chosen = int(candidates[rng.choice(top)])
        y = float(task.curves[chosen, n_obs[chosen]])
        n_obs[chosen] += 1
        f_best = max(f_best, y)
        rows.append((chosen, int(n_obs[chosen]), y, f_best, y_star - f_best))

    cols = list(zip(*rows, strict=True))
    return FreezeThawResult(*(np.asarray(c) for c in cols))


def rung_ladder(b_max: int, eta: int) -> list[int]:
    """Ascending rung budgets ``{b_max // eta^k}``.

    Construct budgets by repeated integer division from b_max, matching NePS.
    """
    if eta < 2 or b_max < 1:
        raise ValueError("eta must be at least 2 and b_max must be positive")
    k_levels = 0
    while b_max // eta ** (k_levels + 1) >= 1:
        k_levels += 1
    return sorted({b_max // eta**k for k in range(k_levels + 1)})


def run_successive_halving(
    task: BenchmarkTask, n_iters: int, seed: int = 0, eta: int = 3, variant: str = "li"
) -> FreezeThawResult:
    """ASHA with one recorded level per step -- the model-free competitor baseline.

    Each iteration trains the active config one step toward its target rung; on completing
    a rung, record its last observed value for ranking, as in Li et al. and NePS.
    The next active config is the deepest-rung promotable one, else a fresh
    uniformly-random config at rung 0. Sequential simulation
    of the asynchronous rule: promotions never wait for a rung to fill.

    ``variant`` selects the promotion quota, two published readings of the same algorithm:

    - ``"li"`` -- Li et al. (2020): a rung's quota is ``len(ever completed there) // eta``
      and a config promotes at most once per rung, so a rung permanently keeps the bottom
      ``1 - 1/eta`` fraction of everything it has seen.
    - ``"neps"`` -- the NePS implementation ifBO benchmarked: a config is a member only of
      its HIGHEST completed rung, so promoted configs leave the membership and the quota is
      ``len(current members) // eta`` -- markedly more aggressive (in steady state a rung
      keeps only its running worst few, and nearly every new finisher triggers a
      promotion).

    Table-world conventions, both variants: a config that exhausts its valid prefix is
    finished, never promoted into the NaN suffix; when the finite pool is exhausted,
    promotion goes quota-free so budget keeps flowing to the best observed configs.
    """
    if variant not in ("li", "neps"):
        raise ValueError(f"unknown ASHA variant {variant!r}")
    n_cfg, b_max = task.curves.shape
    b_valid = _valid_lengths(task)
    budgets = rung_ladder(b_max, eta)

    rng = np.random.default_rng([seed, 313131])
    order = iter(rng.permutation(n_cfg).tolist())
    # li: every (value, cfg) ever completed per rung, plus a promoted set.
    # neps: each config lives only at its highest completed rung (cur_rung), with its value.
    rung_values: list[list[tuple[float, int]]] = [[] for _ in budgets]
    promoted: list[set[int]] = [set() for _ in budgets]
    cur_rung = np.full(n_cfg, -1, dtype=np.int64)
    rung_val = np.full(n_cfg, np.nan)
    n_obs = np.zeros(n_cfg, dtype=np.int64)
    active, target = None, 0
    rows, f_best, y_star = [], -np.inf, float(np.nanmax(task.curves))

    def promote_li(quota: bool):
        for k in range(len(budgets) - 2, -1, -1):  # deepest promotable rung first
            # value desc, ties to the lower config index (kind-stable for the jax port)
            done = sorted(rung_values[k], key=lambda t: (-t[0], t[1]))
            n_promotable = len(done) // eta if quota else len(done)
            for _value, cfg in done[:n_promotable]:
                if cfg not in promoted[k] and n_obs[cfg] < b_valid[cfg]:
                    promoted[k].add(cfg)
                    return cfg, k + 1
        return None, 0

    def promote_neps(quota: bool):
        for k in range(len(budgets) - 2, -1, -1):  # deepest promotable rung first
            members = np.flatnonzero(cur_rung == k)
            n_promotable = len(members) // eta if quota else len(members)
            by_value = members[np.argsort(-rung_val[members], kind="stable")]
            for cfg in by_value[:n_promotable]:
                # exhausted configs linger as members (inflating the quota, as NePS's
                # observation history would) but are never themselves promoted
                if n_obs[cfg] < b_valid[cfg]:
                    return int(cfg), k + 1
        return None, 0

    promote = promote_li if variant == "li" else promote_neps

    def next_active():
        cfg, k = promote(quota=True)
        if cfg is not None:
            return cfg, k
        try:
            return next(order), 0
        except StopIteration:
            # bounded pool exhausted: keep budget flowing to the best observed configs
            # by promoting quota-free, deepest rung first
            return promote(quota=False)

    for _ in range(n_iters):
        if active is None:
            active, target = next_active()
            if active is None:
                break
        y = float(task.curves[active, n_obs[active]])
        n_obs[active] += 1
        f_best = max(f_best, y)
        rows.append((active, int(n_obs[active]), y, f_best, y_star - f_best))
        if n_obs[active] >= min(budgets[target], b_valid[active]):
            # y IS the at-rung value: completion always lands on the just-revealed step
            if variant == "li":
                rung_values[target].append((y, active))
            else:
                cur_rung[active] = target
                rung_val[active] = y
            active = None

    cols = list(zip(*rows, strict=True))
    return FreezeThawResult(*(np.asarray(c) for c in cols))


def run_hyperband(
    task: BenchmarkTask, n_iters: int, seed: int = 0, eta: int = 3
) -> FreezeThawResult:
    """Hyperband (Li et al. 2018) with one recorded level per step -- synchronous SH brackets, cycled.

    Bracket ``s`` (s halvings) starts ``ceil((s_max+1) * eta^s / (s+1))`` fresh configs at
    the ladder's ``(s_max - s)``-th rung; brackets run most-aggressive first and the whole
    cycle repeats until the step budget is spent. Within a bracket each rung trains every
    cohort member up to the rung budget (continuation: a config resumes from its revealed
    prefix), ranks the cohort by its at-rung value, and keeps the top ``len // eta``.
    Rung budgets are ``rung_ladder`` -- NePS's integer ladder, with Li's ceil cohort sizes
    (NePS uses floor rather than ceiling for cohort sizes).

    Table-world conventions: fresh configs come off a seeded permutation that skips
    fully-trained configs and recycles once exhausted (a re-drawn config resumes its
    prefix); a config whose valid prefix ends below a rung is ranked at its last valid
    value and trains no further.
    """
    n_cfg, b_max = task.curves.shape
    b_valid = _valid_lengths(task)
    ladder = rung_ladder(b_max, eta)
    s_max = len(ladder) - 1
    rng = np.random.default_rng([seed, 515151])
    order = rng.permutation(n_cfg).tolist()
    pos = 0
    n_obs = np.zeros(n_cfg, dtype=np.int64)
    rows, f_best, y_star = [], -np.inf, float(np.nanmax(task.curves))

    def fresh(n: int) -> list[int]:
        nonlocal pos
        out: list[int] = []
        scanned = 0
        while len(out) < n and scanned < n_cfg:
            cfg = order[pos % n_cfg]
            pos, scanned = pos + 1, scanned + 1
            if n_obs[cfg] < b_valid[cfg]:
                out.append(cfg)
        return out

    def result() -> FreezeThawResult:
        cols = list(zip(*rows, strict=True))
        return FreezeThawResult(*(np.asarray(c) for c in cols))

    it = 0
    while it < n_iters:
        progressed = False
        for s in range(s_max, -1, -1):  # most aggressive bracket first (Li's ordering)
            cohort = fresh(int(np.ceil((s_max + 1) * eta**s / (s + 1))))
            if not cohort:
                continue  # every config fully trained; later brackets can't draw either
            progressed = True
            rungs = ladder[s_max - s :]
            for i, rung in enumerate(rungs):
                vals = []
                for cfg in cohort:
                    stop = min(rung, int(b_valid[cfg]))
                    while n_obs[cfg] < stop:
                        y = float(task.curves[cfg, n_obs[cfg]])
                        n_obs[cfg] += 1
                        f_best = max(f_best, y)
                        rows.append((cfg, int(n_obs[cfg]), y, f_best, y_star - f_best))
                        it += 1
                        if it >= n_iters:
                            return result()
                    # at-rung value; already-revealed prefixes just read it
                    vals.append(float(task.curves[cfg, stop - 1]))
                if i == len(rungs) - 1:
                    break
                keep = len(cohort) // eta
                if keep == 0:
                    break
                # value desc, ties to the lower cohort position (stable for the jax port)
                top = np.argsort(-np.asarray(vals), kind="stable")[:keep]
                cohort = [cohort[j] for j in top]
        if not progressed:
            break
    return result()


def run_random_search(task: BenchmarkTask, n_iters: int, seed: int = 0) -> FreezeThawResult:
    """Baseline: train uniformly random configs to completion, one step per iteration.

    The classic random-search-with-full-training baseline expressed with one recorded level per step, so
    its regret curve is directly comparable to the freeze-thaw loop's.
    """
    n_cfg, b_max = task.curves.shape
    b_valid = _valid_lengths(task)
    rng = np.random.default_rng([seed, 424242])
    order = rng.permutation(n_cfg)
    rows, f_best, y_star = [], -np.inf, float(np.nanmax(task.curves))
    cumulative_cost = 0.0
    cfg_i, lvl = 0, 0
    for _ in range(n_iters):
        if cfg_i == n_cfg:
            break
        cfg = int(order[cfg_i])
        y = float(task.curves[cfg, lvl])
        f_best = max(f_best, y)
        fid = task.fidelities[cfg] if task.fidelities.ndim == 2 else task.fidelities
        previous = float(fid[lvl - 1]) if lvl else 0.0
        step_cost = max(float(fid[lvl]) - previous, 0.0)
        cumulative_cost += step_cost
        rows.append((cfg, lvl + 1, y, f_best, y_star - f_best, cumulative_cost))
        lvl += 1
        if lvl >= b_valid[cfg]:
            cfg_i, lvl = cfg_i + 1, 0
    cols = list(zip(*rows, strict=True))
    return FreezeThawResult(*(np.asarray(c) for c in cols))
