"""ARLBench landscape tables for 42 environment/algorithm pairs.

Each task has 256 configurations, 10 training seeds and 10 fidelity levels.
Normalize evaluation returns using the finite range across the full task table.
Following the benchmark convention, replace crash NaNs with the minimum finite
return before averaging seeds.

seedK selects one training seed; mean averages all seeds. Fidelity is training
progress normalized to the final recorded step.
"""

import numpy as np
import pandas as pd

from reproduction.bench._hp_spaces import ARLBENCH_BOOL, ARLBENCH_NUM
from reproduction.bench.task import BenchmarkTask, normalize_hp, resolve_data_dir

# Provide each stored seed and the mean across seeds as separate views.
ARLBENCH_VIEWS = tuple(f"seed{i}" for i in range(10)) + ("mean",)

# Conditional HPs are stored as NaN when their parent boolean disables them; fill with the
# semantically equivalent active value (qtmeta's null-policy pattern): prioritized
# sampling off == uniform sampling (alpha/beta/epsilon at their lower bounds); target
# network off == the online net is its own target, i.e. hard sync every step
# (update_interval -> 1 = lower, tau -> 1.0 = upper).
_FILL_INACTIVE = {
    "buffer_alpha": "lower",
    "buffer_beta": "lower",
    "buffer_epsilon": "lower",
    "target_update_interval": "lower",
    "tau": "upper",
}


def arlbench_stems(data_dir=None) -> list[str]:
    """The (env, algorithm) task stems, e.g. ``cc_cartpole_ppo``, sorted."""
    root = resolve_data_dir(data_dir) / "arlbench" / "landscapes"
    return sorted(p.stem for p in root.glob("*.csv"))


def load_arlbench(stem: str, view: str = "mean", data_dir=None) -> BenchmarkTask:
    """One (env, algorithm) task in one seed view; see the module docstring."""
    if view not in ARLBENCH_VIEWS:
        raise ValueError(f"unknown arlbench view {view!r}")
    algo = stem.rsplit("_", 1)[1]
    df = pd.read_csv(resolve_data_dir(data_dir) / "arlbench" / "landscapes" / f"{stem}.csv")

    piv = df.pivot(index=["config_id", "seed"], columns="training_steps", values="performance")
    steps = piv.columns.to_numpy(dtype=np.float64)
    n_seeds = df["seed"].nunique()
    raw = piv.to_numpy(dtype=np.float64)
    if raw.shape[0] != piv.index.levels[0].size * n_seeds:
        raise ValueError(f"{stem}: missing configuration/seed pairs")
    # Use the full-table range for every view. Replace crash NaNs before averaging seeds.
    finite = raw[np.isfinite(raw)]
    if not finite.size:
        raise ValueError(f"{stem}: no finite performance values")
    f_min, f_max = finite.min(), finite.max()
    raw = np.where(np.isfinite(raw), raw, f_min)
    by_seed = raw.reshape(-1, n_seeds, steps.size)  # pivot sorts (config_id, seed)

    if view == "mean":
        curves = by_seed.mean(axis=1)
    elif view.startswith("seed"):
        seed = int(view.removeprefix("seed"))
        seeds = piv.index.get_level_values("seed").unique().to_numpy()
        if seed not in seeds:
            raise ValueError(f"{stem}: seed {seed} is absent")
        curves = by_seed[:, int(np.flatnonzero(seeds == seed)[0]), :]
    else:
        raise ValueError(f"unknown arlbench view {view!r}; seedK or mean")

    curves = np.clip((curves - f_min) / max(f_max - f_min, 1e-12), 0.0, 1.0)

    # HPs from any one row per config (constant across seed/level); union of numeric + bool
    # columns sorted by name, booleans at 2-way bucket centers.
    first_rows = df[df["seed"] == df["seed"].min()].groupby("config_id").first()
    num = {spec[0]: spec for spec in ARLBENCH_NUM[algo]}
    bools = ARLBENCH_BOOL[algo]
    cols, card = [], []
    for name in sorted([*num, *bools]):
        v = first_rows[f"hp_config.{name}"].to_numpy()
        if name in bools:
            cols.append(np.where(v.astype(bool), 0.75, 0.25))
            card.append(2)
        else:
            _, lower, upper, log = num[name]
            if name in _FILL_INACTIVE:
                fill = {"lower": lower, "upper": upper}[_FILL_INACTIVE[name]]
                v = np.where(np.isnan(v.astype(np.float64)), fill, v)
            cols.append(normalize_hp(v, lower, upper, log))
            card.append(0)
    hps = np.stack(cols, axis=1).astype(np.float32)
    if not np.isfinite(hps).all():
        raise ValueError(f"{stem}: nonfinite HP outside the known conditional slots")

    return BenchmarkTask(
        name=f"arlbench/{view}/{stem}",
        hps=hps,
        curves=curves.astype(np.float32),
        fidelities=(steps / steps.max()).astype(np.float32),
        cat_card=np.asarray(card, dtype=np.int32),
    )
