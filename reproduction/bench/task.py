"""Common representation of benchmark configurations and learning curves.

HPs are normalized to [0, 1] using search-space bounds or grid ranks; categorical
values use bucket centers (k + 0.5) / cardinality. Responses lie in [0, 1], with
larger values better.

Each curve has at least one finite level and may end with a contiguous NaN suffix.
Fidelities are either a shared (n_steps,) grid or a per-configuration
(n_configs, n_steps) array with the same NaN mask as the curves.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

from icicle.hpo import normalize_hp


@dataclasses.dataclass(frozen=True)
class BenchmarkTask:
    """One benchmark task (see module docstring for the conventions)."""

    name: str  # e.g. "lcbench/adult"
    hps: np.ndarray  # (n_configs, n_hps) float32 in [0, 1]
    curves: np.ndarray  # (n_configs, n_steps) float32 in [0, 1], higher = better; NaN suffix ok
    fidelities: np.ndarray  # (n_steps,) shared, or (n_configs, n_steps) per-config (see module)
    cat_card: np.ndarray | None = None  # (n_hps,) int32 category counts, 0 = continuous slot
    # Optional per-row, per-level incremental benchmark cost in native units. Fidelity remains
    # the model-visible coordinate; runners that need exact accounting opt into this table.
    costs: np.ndarray | None = None  # (n_configs, n_steps), NaN-padded like curves


def valid_lengths(curves: np.ndarray) -> np.ndarray:
    """Validate finite curve prefixes with optional trailing NaNs."""
    if curves.ndim != 2 or 0 in curves.shape:
        raise ValueError("curves must be a nonempty matrix")
    finite = np.isfinite(curves)
    lengths = finite.sum(axis=1)
    prefix = np.arange(curves.shape[1])[None, :] < lengths[:, None]
    if (lengths == 0).any() or not np.array_equal(finite, prefix) or np.isinf(curves).any():
        raise ValueError("each curve must have a finite prefix followed only by NaNs")
    return lengths


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """The benchmark-data root (the directory containing ``lcbench-tabular/`` etc.).

    Use the explicit argument or ``$ICICLE_BENCH_DATA``.
    """
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("ICICLE_BENCH_DATA")
    if env:
        return Path(env)
    raise RuntimeError(
        "benchmark data dir not set: pass data_dir=... or set $ICICLE_BENCH_DATA to the root "
        "containing lcbench-tabular/, pd1-tabular/, taskset-tabular/"
    )


def pivot(table: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return curves, epochs and HP rows in the published configuration order.

    Sort configuration IDs as strings to match ifBO. Address index levels by
    position because their names vary across benchmarks.
    """
    ids = sorted(table.index.get_level_values(0).unique(), key=str)
    mat = table[metric].astype(np.float64).unstack(level=1).reindex(ids)
    epochs = mat.columns.to_numpy(dtype=np.float64)
    first_rows = table.groupby(level=0, sort=False).first().reindex(ids)
    return mat.to_numpy(), epochs, first_rows


def hp_matrix(first_rows: pd.DataFrame, space) -> np.ndarray:
    """Normalize the per-config HP columns to [0, 1] per the ``space`` bound specs."""
    cols = [
        normalize_hp(first_rows[name].to_numpy(dtype=np.float64), lower, upper, log)
        for name, lower, upper, log in space
    ]
    return np.stack(cols, axis=1).astype(np.float32)
