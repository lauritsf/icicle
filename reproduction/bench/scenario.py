"""Sample context/query splits from stored learning curves.

Draw configuration/level points without replacement, using gamma-distributed
configuration weights. Context is a prefix of each selected curve; queries lie
above its cutoff and within its recorded range.

icicle-v1 cycles a grid of allocation concentrations and context fractions,
includes each queried curve's final level, and shrinks the sample for small pools.
ifbo-parity follows the reference draw order and curve-ID capacity: curves beyond
the capacity lose their context and become queries without observations.

Each checkpoint uses the same sampled table indices for a given task and seed.
"""

import dataclasses
import zlib
from collections.abc import Callable, Iterator

import numpy as np

from reproduction.bench.arlbench import ARLBENCH_VIEWS, arlbench_stems, load_arlbench
from reproduction.bench.lcbench import lcbench_task_ids, load_lcbench
from reproduction.bench.pd1 import (
    PD1_IFBO_TASKS,
    load_pd1,
)
from reproduction.bench.qtmeta import QTMETA_SPLITS, load_qtmeta, qtmeta_datasets
from reproduction.bench.qtmeta import qtmeta_task_name as _qt_name
from reproduction.bench.task import BenchmarkTask, valid_lengths
from reproduction.bench.taskset import (
    TASKSET_IFBO_TASKS,
    TASKSET_OPTIMIZERS,
    load_taskset,
)

BENCHMARKS = (
    "lcbench_ifbo",
    "pd1_ifbo",
    "taskset_ifbo",
    "qtmeta",
    "qtmeta_epoch",
) + tuple(f"arlbench_{view}" for view in ARLBENCH_VIEWS)


@dataclasses.dataclass(frozen=True)
class EvalProtocol:
    """Settings for sampling context/query splits."""

    mode: str = "icicle-v1"  # or "ifbo-parity"
    n_total: int = 1000  # N = |context| + |queries|; match the trained seq_len
    n_scenarios: int = 20  # per task
    context_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)  # icicle-v1 grid
    log_alpha_grid: tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0)  # icicle-v1 grid
    context_size: int | None = None  # ifbo-parity: a fixed Table-1 row; None = U(0, N-1)
    max_curves: int = 1000  # ifbo-parity only: FT-PFN's curve-id cap

    def __post_init__(self):
        if self.mode not in ("icicle-v1", "ifbo-parity"):
            raise ValueError(f"unknown scenario mode {self.mode!r}")


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One (context, query) split as index arrays into the task's tables.

    Points are ordered context-first (config-ascending, each config's levels in epoch
    order), then queries (config-ascending). ``point_config_idx`` indexes into
    ``config_index``, which indexes into ``task.hps`` -- the compact config table the
    factored model encodes once.
    """

    config_index: np.ndarray  # (C_used,) int32 rows of task.hps referenced
    point_config_idx: np.ndarray  # (P,) int32 into config_index
    level_index: np.ndarray  # (P,) int32 columns of task.curves / task.fidelities
    n_context: int  # the context/query boundary (single_eval_pos)
    n_total_eff: int  # realized N (== P; < protocol.n_total when the pool was smaller)
    alpha: float  # the gamma concentration used
    context_fraction: float | None  # requested fraction (icicle-v1), None in parity


def sample_scenario(
    task: BenchmarkTask,
    rng: np.random.Generator,
    protocol: EvalProtocol,
    *,
    context_fraction: float | None = None,
    log_alpha: float | None = None,
) -> Scenario | None:
    """One scenario draw; None on a degenerate draw (no queries, or parity + small pool).

    In ``ifbo-parity`` mode the rng call order is kept identical to the reference sampler
    (integers -> uniform -> gamma -> choice -> per-curve choice) so golden scenarios dumped
    from ifbo-jax match bit-for-bit. ``context_fraction`` / ``log_alpha`` are the icicle-v1
    grid cell and must be None in parity mode.
    """
    parity = protocol.mode == "ifbo-parity"
    n_cfg = task.curves.shape[0]
    # Ragged curves (NaN suffix): each config contributes only its observed levels to the
    # pool. Rectangular tasks reduce to the pre-ragged code path bit-exactly (all n_valid
    # equal), which the parity and icicle-v1 golden tests pin.
    n_valid = valid_lengths(task.curves)
    pool_size = int(n_valid.sum())
    if pool_size < protocol.n_total and parity:
        return None  # ifBO skips tasks whose pool cannot fill N tokens
    seq_len = min(protocol.n_total, pool_size)

    if parity:
        if protocol.context_size is not None:
            n_context = min(protocol.context_size, seq_len - 1)
        else:
            n_context = int(rng.integers(0, seq_len))
        alpha = 10.0 ** rng.uniform(-4.0, -1.0)
    else:
        n_context = int(np.clip(round(context_fraction * seq_len), 1, seq_len - 1))
        alpha = 10.0**log_alpha

    # Gamma weights over configs -> per-(config, level) sampling probability; one ordering
    # of seq_len (config, level) slots without replacement. Only the per-curve COUNTS matter
    # downstream: context is the epoch prefix, queries are re-drawn above the cutoff.
    # Each config's mass spreads over its own observed levels (equal-per-curve mass, ifBO's
    # semantics): short/diverged curves keep full sampling weight, denser per slot.
    weights = rng.gamma(alpha, alpha, n_cfg) + 1e-9
    p = np.repeat(weights / weights.sum() / n_valid, n_valid)
    order = rng.choice(pool_size, size=seq_len, replace=False, p=p / p.sum())
    order_cfg = np.searchsorted(np.cumsum(n_valid), order, side="right").astype(np.int64)

    n_tokens_per_curve = np.bincount(order_cfg, minlength=n_cfg)
    n_observed_per_curve = np.bincount(order_cfg[:n_context], minlength=n_cfg)

    ctx: list[tuple[int, int]] = []  # (config, level), config-ascending, prefix order
    qry: list[tuple[int, int]] = []
    n_with_context = 0
    for cid in range(n_cfg):
        ec = int(n_tokens_per_curve[cid])
        if ec == 0:
            continue
        cut = int(n_observed_per_curve[cid])
        if parity and cut > 0:
            n_with_context += 1
            if n_with_context > protocol.max_curves:
                cut = 0  # past FT-PFN's id cap: context demoted to from-scratch queries
        ctx.extend((cid, e) for e in range(cut))
        n_q = ec - cut
        if n_q > 0:
            avail = np.arange(cut, n_valid[cid])
            if parity:
                levels = rng.choice(avail, size=n_q, replace=False)
            else:
                # Force-include the curve's last observed level -- the point BO acts on.
                rest = rng.choice(avail[:-1], size=n_q - 1, replace=False)
                levels = np.append(rest, n_valid[cid] - 1)
            qry.extend((cid, int(e)) for e in levels)

    if not qry:
        return None

    points = np.asarray(ctx + qry, dtype=np.int32)
    config_index = np.unique(points[:, 0])
    return Scenario(
        config_index=config_index,
        point_config_idx=np.searchsorted(config_index, points[:, 0]).astype(np.int32),
        level_index=points[:, 1],
        n_context=len(ctx),
        n_total_eff=len(points),
        alpha=float(alpha),
        context_fraction=context_fraction,
    )


def iter_scenarios(
    task: BenchmarkTask, protocol: EvalProtocol, seed: int
) -> Iterator[tuple[int, Scenario]]:
    """The paired scenario stream for one task.

    In icicle-v1 the (fraction, alpha) grid is cycled across the scenario index --
    fractions fastest -- so the default 20 scenarios cover the 5x4 grid exactly once.
    Degenerate draws retry up to 5 times on their own deterministic substream (the retry
    count is part of the rng seed, so pairing survives retries); a scenario that stays
    degenerate is skipped, which keeps the (task, scenario_idx) join keys stable.
    """
    task_hash = zlib.crc32(task.name.encode())
    for idx in range(protocol.n_scenarios):
        kw = {}
        if protocol.mode == "icicle-v1":
            fracs, alphas = protocol.context_fractions, protocol.log_alpha_grid
            kw = {
                "context_fraction": fracs[idx % len(fracs)],
                "log_alpha": alphas[(idx // len(fracs)) % len(alphas)],
            }
        for retry in range(5):
            rng = np.random.default_rng([1, seed, task_hash, idx, retry])
            scenario = sample_scenario(task, rng, protocol, **kw)
            if scenario is not None:
                yield idx, scenario
                break


def benchmark_tasks(benchmark: str, data_dir=None) -> list[tuple[str, Callable[[], BenchmarkTask]]]:
    """Yield (name, lazy loader) pairs in deterministic order."""
    if benchmark == "lcbench_ifbo":
        return [
            (f"lcbench/{tid}", lambda tid=tid: load_lcbench(tid, data_dir))
            for tid in lcbench_task_ids(data_dir)
        ]
    if benchmark == "pd1_ifbo":
        return [
            (
                f"pd1/{ds}-{m}-{bs}" + (f"-{c}" if c is not None else ""),
                lambda ds=ds, m=m, bs=bs, c=c: load_pd1(ds, m, bs, c, data_dir),
            )
            for ds, m, bs, c in PD1_IFBO_TASKS
        ]
    if benchmark == "taskset_ifbo":
        return [
            (
                f"taskset/{tid}-{opt}",
                lambda tid=tid, opt=opt: load_taskset(tid, opt, data_dir),
            )
            for tid in TASKSET_IFBO_TASKS
            for opt in TASKSET_OPTIMIZERS
        ]
    if benchmark in ("qtmeta", "qtmeta_epoch"):
        fid = "epoch" if benchmark == "qtmeta_epoch" else "walltime"
        return [
            (_qt_name(sp, ds), lambda sp=sp, ds=ds: load_qtmeta(sp, ds, fid, data_dir))
            for sp in QTMETA_SPLITS
            for ds in qtmeta_datasets(sp, data_dir)
        ]
    if benchmark.startswith("arlbench_"):
        view = benchmark.removeprefix("arlbench_")
        if view not in ARLBENCH_VIEWS:
            raise ValueError(f"unknown arlbench view {view!r}; one of {ARLBENCH_VIEWS}")
        return [
            (f"arlbench/{view}/{stem}", lambda stem=stem: load_arlbench(stem, view, data_dir))
            for stem in arlbench_stems(data_dir)
        ]
    raise ValueError(f"unknown benchmark {benchmark!r}; one of {BENCHMARKS}")
