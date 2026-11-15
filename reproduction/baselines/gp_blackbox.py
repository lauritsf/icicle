"""NumPy full-curve GP baseline inspired by Quick-Tune's GP ablation.

Each evaluation reveals only the final value and charges the full recorded curve
cost. The GP uses HPs, a Matern-5/2 kernel with lengthscale 1.0, noise 1e-3 on
standardized observations, and expected improvement. Three random evaluations
initialize the model; subsequent proposals maximize acquisition over the remaining
finite pool. Kernel scales are fixed.

Regret is measured against the best value at any fidelity, as for the other
runners. It can therefore remain positive when the optimum occurs before a curve's
end. Ragged curves use their last finite value and cumulative cost.
"""

import numpy as np

from reproduction.bench.task import BenchmarkTask
from reproduction.baselines.gp_ft import N_INIT, NOISE, log_ei, matern52
from reproduction.baselines.scheduler import FreezeThawResult, _valid_lengths

RNG_CONST = 717171  # init-permutation stream id, distinct from gp_ft's 616161


def blackbox_view(task: BenchmarkTask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-config ``(final_y, final_budget, eval_cost)`` of the black-box presentation.

    ``final_y`` is the last finite curve value, ``final_budget`` the 1-based level count it
    lands on, ``eval_cost`` the full-curve cumulative fidelity (see the module docstring).
    Used by both the NumPy reference and the JAX scan for consistent cost accounting.
    """
    b_valid = _valid_lengths(task)
    idx = np.arange(len(b_valid))
    final_y = np.asarray(task.curves, np.float64)[idx, b_valid - 1]
    fid = np.asarray(task.fidelities, np.float64)
    eval_cost = fid[idx, b_valid - 1] if fid.ndim == 2 else fid[b_valid - 1]
    return final_y, b_valid, eval_cost


def run_gp_blackbox(task: BenchmarkTask, n_iters: int, seed: int = 0) -> FreezeThawResult:
    n_cfg = task.curves.shape[0]
    hps = np.asarray(task.hps, np.float64)
    final_y, final_budget, eval_cost = blackbox_view(task)
    y_star = float(np.nanmax(task.curves))
    evaluated = np.zeros(n_cfg, dtype=bool)
    ctx: list[int] = []  # evaluation order -- the scan implementation's context order, kept identical
    rows, f_best, cum_cost = [], -np.inf, 0.0

    def evaluate(cfg: int):
        nonlocal f_best, cum_cost
        y = float(final_y[cfg])
        evaluated[cfg] = True
        ctx.append(cfg)
        f_best = max(f_best, y)
        cum_cost += float(eval_cost[cfg])
        rows.append((cfg, int(final_budget[cfg]), y, f_best, y_star - f_best, cum_cost))

    for cfg in np.random.default_rng([seed, RNG_CONST]).permutation(n_cfg)[:N_INIT]:
        if len(rows) < n_iters:
            evaluate(int(cfg))

    for _ in range(len(rows), n_iters):
        candidates = np.flatnonzero(~evaluated)
        if candidates.size == 0:
            break
        # standardize y per fit -- in-run, no cross-task statistic
        y = final_y[ctx]
        mean, std = y.mean(), max(float(y.std()), 1e-8)
        yz = (y - mean) / std

        k = matern52(hps[ctx], hps[ctx]) + NOISE * np.eye(len(ctx))
        chol = np.linalg.cholesky(k)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yz))
        ks = matern52(hps[candidates], hps[ctx])  # (C, P)
        mu = ks @ alpha
        v = np.linalg.solve(chol, ks.T)  # (P, C)
        var = np.maximum(1.0 - (v**2).sum(0), 1e-12)

        # plain EI against the incumbent final value (black-box: no per-budget incumbents)
        sd = np.sqrt(var)
        u = (mu - (f_best - mean) / std) / sd
        evaluate(int(candidates[int(np.argmax(log_ei(u, sd)))]))

    cols = list(zip(*rows, strict=True))
    return FreezeThawResult(*(np.asarray(c) for c in cols))
