"""Map metrics onto [1/3, 2/3], leaving room for better and worse predictions."""

import numpy as np

DEFAULT_BAND = (1.0 / 3.0, 2.0 / 3.0)


class MetricMap:
    """Scale metrics using fixed bounds or bounds fitted from observations.

    worst and best stay in the metric's original units. For maximized metrics,
    fit() estimates missing bounds using the 5th percentile and maximum. For
    minimized metrics, it uses the 95th percentile and minimum.
    """

    def __init__(self, *, worst: float | None = None, best: float | None = None,
                 maximize: bool = True):
        self.maximize = maximize
        self._fixed_worst = None if worst is None else float(worst)
        self._fixed_best = None if best is None else float(best)
        for bound in (self._fixed_worst, self._fixed_best):
            if bound is not None and not np.isfinite(bound):
                raise ValueError("Metric bounds must be finite.")
        if worst is not None and best is not None:
            valid_order = best > worst if self.maximize else best < worst
            if not valid_order:
                raise ValueError("Best must be better than worst under the chosen metric direction.")
        self.fit([])

    def fit(self, values):
        """Fit missing bounds from the complete observation history; return self."""
        values = np.asarray(values, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        estimated_worst = estimated_best = np.nan
        if values.size:
            if self.maximize:
                estimated_worst = float(np.quantile(values, 0.05))
                estimated_best = float(values.max())
            else:
                estimated_worst = float(np.quantile(values, 0.95))
                estimated_best = float(values.min())
        self._worst = self._fixed_worst if self._fixed_worst is not None else estimated_worst
        self._best = self._fixed_best if self._fixed_best is not None else estimated_best
        return self

    @property
    def ready(self) -> bool:
        """Whether the fitted bounds define a usable scale."""
        worst, best = self._worst, self._best
        improvement = best - worst if self.maximize else worst - best
        return bool(np.isfinite(worst) and np.isfinite(best)
                    and improvement > 1e-9 * max(abs(worst), abs(best), 1.0))

    def __call__(self, raw) -> np.ndarray:
        """Map metrics to [0, 1], with non-finite observations at zero."""
        raw = np.asarray(raw, dtype=np.float64)
        band_low, band_high = DEFAULT_BAND
        if self.ready:
            fraction_toward_best = (raw - self._worst) / (self._best - self._worst)
            out = band_low + fraction_toward_best * (band_high - band_low)
        else:
            out = np.full(raw.shape, 0.5 * (band_low + band_high), dtype=np.float64)
        return np.where(np.isfinite(raw), np.clip(out, 0.0, 1.0), 0.0).astype(np.float32)

    def to_raw(self, presented) -> np.ndarray:
        """Convert predictions back to metric units; return NaN until bounds are fitted."""
        presented = np.asarray(presented, dtype=np.float64)
        if not self.ready:
            return np.full(presented.shape, np.nan, dtype=np.float32)
        band_low, band_high = DEFAULT_BAND
        fraction_toward_best = (presented - band_low) / (band_high - band_low)
        raw = self._worst + fraction_toward_best * (self._best - self._worst)
        return raw.astype(np.float32)
