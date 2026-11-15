"""NumPy freeze-thaw GP baseline using one-step expected improvement.

The GP uses joint (HP, fidelity) inputs, a Matern-5/2 kernel with lengthscale
1.0 and noise 1e-3 on standardized observations. Fidelity is (level + 1) / b_max.
EI uses the incumbent at the proposed level, or the global incumbent when that
level is unobserved. Initialize with three random configurations at one step each.

This version fixes the kernel scales and scores the full candidate pool. The
NePS reference fits the scales and samples 100 unseen candidates.
"""

import math

import numpy as np

from reproduction.bench.task import BenchmarkTask
from reproduction.baselines.scheduler import FreezeThawResult, _valid_lengths

LENGTHSCALE = 1.0
NOISE = 1e-3  # variance, on standardized y (the NePS default; their MLE caps at 0.01)
N_INIT = 3

_erf = np.vectorize(math.erf, otypes=[np.float64])  # keeps the core scipy-free


def _norm_pdf(u):
    return np.exp(-0.5 * u**2) / math.sqrt(2.0 * math.pi)


def _norm_cdf(u):
    return 0.5 * (1.0 + _erf(u / math.sqrt(2.0)))


def log_ei(u, sd):
    """Underflow-stable log EI: log(sd) + log h(u) with h(u) = pdf(u) + u * cdf(u).

    A confident GP drives h to sub-denormal for every candidate, and a plain EI argmax
    degenerates to first-index ties. Below u = -8 the direct form is replaced by the
    two-term asymptotic log h ~ log pdf(u) - 2 log(-u) + log1p(-3/u^2) (relative error
    <0.5% at the cut, decisions shared with the JAX implementation which uses the same branch).
    """
    u = np.asarray(u, np.float64)
    h = _norm_pdf(u) + u * _norm_cdf(u)
    direct = np.log(np.maximum(h, 1e-300))
    un = np.minimum(u, -8.0)  # keeps log(-u)/log1p defined where the branch is unused
    asym = -0.5 * un**2 - 0.5 * np.log(2.0 * np.pi) - 2.0 * np.log(-un) + np.log1p(-3.0 / un**2)
    return np.log(sd) + np.where(u > -8.0, direct, asym)


def matern52(a: np.ndarray, b: np.ndarray, lengthscale: float = LENGTHSCALE) -> np.ndarray:
    """k(a, b) for row-stacked inputs; unit output scale."""
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    r = np.sqrt(np.maximum(d2, 0.0)) / lengthscale
    s5r = np.sqrt(5.0) * r
    return (1.0 + s5r + 5.0 / 3.0 * r**2) * np.exp(-s5r)


def run_gp_freeze_thaw(task: BenchmarkTask, n_iters: int, seed: int = 0) -> FreezeThawResult:
    n_cfg, b_max = task.curves.shape
    b_valid = _valid_lengths(task)
    hps = np.asarray(task.hps, np.float64)
    n_obs = np.zeros(n_cfg, dtype=np.int64)
    rows, f_best, y_star = [], -np.inf, float(np.nanmax(task.curves))
    ctx_cfg: list[int] = []
    ctx_lvl: list[int] = []
    ctx_y: list[float] = []

    def step(cfg: int):
        nonlocal f_best
        y = float(task.curves[cfg, n_obs[cfg]])
        ctx_cfg.append(cfg)
        ctx_lvl.append(int(n_obs[cfg]))
        ctx_y.append(y)
        n_obs[cfg] += 1
        f_best = max(f_best, y)
        rows.append((cfg, int(n_obs[cfg]), y, f_best, y_star - f_best))

    for cfg in np.random.default_rng([seed, 616161]).permutation(n_cfg)[:N_INIT]:
        if len(rows) < n_iters:
            step(int(cfg))

    for _ in range(len(rows), n_iters):
        candidates = np.flatnonzero(n_obs < b_valid)
        if candidates.size == 0:
            break
        # standardize y per fit -- in-run, no cross-task statistic
        y = np.asarray(ctx_y)
        mean, std = y.mean(), max(float(y.std()), 1e-8)
        yz = (y - mean) / std

        x = np.concatenate(
            [hps[ctx_cfg], (np.asarray(ctx_lvl, np.float64)[:, None] + 1.0) / b_max], axis=1
        )
        q = np.concatenate([hps[candidates], (n_obs[candidates, None] + 1.0) / b_max], axis=1)
        k = matern52(x, x) + NOISE * np.eye(len(x))
        chol = np.linalg.cholesky(k)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yz))
        ks = matern52(q, x)  # (C, P)
        mu = ks @ alpha
        v = np.linalg.solve(chol, ks.T)  # (P, C)
        var = np.maximum(1.0 - (v**2).sum(0), 1e-12)

        # MF-EI: incumbent AT the candidate's target level, global best as the fallback
        level_best = np.full(b_max, -np.inf)
        np.maximum.at(level_best, ctx_lvl, y)
        inc = np.where(
            level_best[n_obs[candidates]] > -np.inf, level_best[n_obs[candidates]], f_best
        )
        inc_z = (inc - mean) / std

        sd = np.sqrt(var)
        u = (mu - inc_z) / sd  # maximization form of NePS's minimizing EI
        step(int(candidates[int(np.argmax(log_ei(u, sd)))]))

    cols = list(zip(*rows, strict=True))
    return FreezeThawResult(*(np.asarray(c) for c in cols))
