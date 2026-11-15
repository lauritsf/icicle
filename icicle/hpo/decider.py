"""Select trials from learning-curve predictions."""

import dataclasses

import numpy as np


def sample_mfpi_random(rng: np.random.Generator, b_max: int, f_best: float) -> tuple[int, float]:
    """One (horizon, threshold) draw: h ~ U{1..b_max}, T = f_best + tau * (1 - f_best)."""
    h = int(rng.integers(1, b_max + 1))
    tau = 10.0 ** rng.uniform(-4.0, -1.0)
    return h, f_best + tau * (1.0 - f_best)


def exceedance_prob(probs: np.ndarray, borders: np.ndarray, threshold: float) -> np.ndarray:
    """P(y > threshold) under a bar distribution: full buckets above + the partial bucket.

    ``probs`` is (..., num_bars) bucket masses, ``borders`` the shared (num_bars + 1,) edges.
    The bucket containing the threshold contributes its mass times the fraction of its width
    above the threshold (the bar density is uniform within a bucket). Thresholds outside the
    border range give exactly 1 or 0.
    """
    t = np.clip(threshold, borders[0], borders[-1])
    idx = int(np.clip(np.searchsorted(borders, t, side="right") - 1, 0, len(borders) - 2))
    lo, hi = borders[idx], borders[idx + 1]
    frac_above = (hi - t) / (hi - lo) if hi > lo else 0.0
    return probs[..., idx + 1 :].sum(-1) + probs[..., idx] * frac_above


@dataclasses.dataclass
class FreezeThawDecider:
    """Choose which trial to continue using MFPI-random."""

    borders: np.ndarray  # shared (num_bars + 1,) bucket edges
    seed: int = 0

    n_decisions: int = dataclasses.field(default=0, init=False)

    def decide(
        self,
        cand_hps: np.ndarray,
        cand_steps: np.ndarray,
        fid_grid: np.ndarray,
        predict_fn,
        *,
        context_hps: np.ndarray,
        context_fidelities: np.ndarray,
        context_values: np.ndarray,
    ) -> int:
        """Return the index of the candidate to advance.

        Context arrays contain observed HP vectors, fidelities and mapped values.
        cand_steps counts completed fidelity levels for each candidate. Queries
        look ahead from those levels, up to the end of fid_grid.
        """
        rng = np.random.default_rng([self.seed, self.n_decisions])
        self.n_decisions += 1
        cand_hps = np.atleast_2d(np.asarray(cand_hps, dtype=np.float32))
        cand_steps = np.asarray(cand_steps)
        b_max = len(fid_grid)

        context_values = np.asarray(context_values)
        if context_values.size == 0:
            return int(rng.integers(len(cand_hps)))

        h, threshold = sample_mfpi_random(rng, b_max, float(context_values.max()))
        context_hps = np.asarray(context_hps, dtype=np.float32)
        context_fidelities = np.asarray(context_fidelities, dtype=np.float32)
        context_values = context_values.astype(np.float32)
        target = np.minimum(cand_steps + h, b_max) - 1  # (Q,)
        probs = predict_fn(
            context_hps,
            context_fidelities,
            context_values,
            cand_hps,
            np.asarray(fid_grid, dtype=np.float32)[target],
        )
        scores = exceedance_prob(np.asarray(probs), self.borders, threshold)

        # Break ties randomly.
        top = np.flatnonzero(scores >= scores.max() - 1e-12)
        return int(rng.choice(top))
