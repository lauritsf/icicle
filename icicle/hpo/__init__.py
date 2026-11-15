import numpy as np

from icicle.hpo.decider import FreezeThawDecider, exceedance_prob, sample_mfpi_random

__all__ = ["FreezeThawDecider", "exceedance_prob", "normalize_hp", "sample_mfpi_random"]


def normalize_hp(values: np.ndarray, lower: float, upper: float, log: bool) -> np.ndarray:
    """Map hyperparameter values to [0, 1] using their search-space bounds."""
    v = np.asarray(values, dtype=np.float64)
    if log:
        v = np.clip(v, lower, upper)
        return (np.log(v) - np.log(lower)) / (np.log(upper) - np.log(lower))
    return np.clip((v - lower) / max(upper - lower, 1e-12), 0.0, 1.0)
