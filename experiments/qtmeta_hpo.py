"""Run Meta-Album HPO with the paper's 1/4/16-hour table-cost budgets.

Methods share a seeded initial configuration. Each accepted step reveals one
recorded epoch and charges training plus evaluation time. ICICLE uses adaptive
value mapping and cost-cooled MFPI. Save a trajectory after every completed batch.

    python -m experiments.qtmeta_hpo --method icicle --model RUN --out results/qtmeta.parquet
    python -m experiments.qtmeta_hpo --method asha --out results/asha.parquet
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jaxlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tyro  # noqa: E402

from experiments.hpo_regret import make_runner  # noqa: E402
from reproduction.bench.qtmeta import (  # noqa: E402
    _T_MAX, load_qtmeta, qtmeta_datasets, qtmeta_task_name, raw_scores,
)
from reproduction.baselines.qtmeta_common import (  # noqa: E402
    initial_index, normalized_regret, random_search_events, structural_ceiling, task_bounds,
)
from reproduction.baselines.qtmeta_asha import run_asha  # noqa: E402

PROTOCOL = {
    "initialization": {"domain_label": "icicle/qtmeta-common-v2/paired-start/v1"},
    "random_search": {"domain_label": "icicle/qtmeta-common-v2/candidate-order/v1"},
}


@dataclass
class Args:
    method: Literal["icicle", "ftpfn", "asha", "random_search"]
    out: Path
    model: Path | None = None
    step: int | None = None
    feature_seed: int = 0
    seeds: tuple[int, ...] = tuple(range(20))
    chain_batch: int = 32
    split: Literal["micro", "mini", "extended"] | None = None
    tasks: tuple[str, ...] = ()
    data_dir: str | None = None
    value_map: Literal["off", "fixed", "adaptive"] = "adaptive"


def exact_trajectory(task, seed, proposals, cap):
    """Replay accepted proposals in table seconds and check the device's decisions."""
    first = initial_index(PROTOCOL, task.name, seed, len(task.curves))
    steps = [(first, 1, True)] + [
        (int(proposals["chosen"][i]), int(proposals["budget"][i]), bool(proposals["accepted"][i]))
        for i in np.flatnonzero(proposals["proposed"])
    ]
    worst, best = task_bounds(task.curves)
    cumulative, incumbent = 0.0, worst
    rows = []
    for chosen, level, device_accepted in steps:
        cost = float(task.costs[chosen, level - 1])
        accepted = cumulative + cost <= cap
        if accepted != device_accepted:
            raise RuntimeError(f"{task.name}: device and float64 cost accounting disagree")
        observed = None
        if accepted:
            cumulative += cost
            observed = float(task.curves[chosen, level - 1])
            incumbent = max(incumbent, observed)
        rows.append(dict(
            chosen=chosen, budget=level, observed=observed, accepted=accepted,
            cost=cumulative, f_best=incumbent, regret=best - incumbent,
            normalized_regret=normalized_regret(incumbent, worst, best),
        ))
        if not accepted:
            break
    if rows[-1]["accepted"] and len(rows) < np.isfinite(task.curves).sum():
        raise RuntimeError(f"{task.name}: run ended before table exhaustion or budget rejection")
    return pd.DataFrame(rows)


def baseline_trajectory(task, seed, method, cap, ceiling, scores=None):
    worst, best = task_bounds(task.curves)
    if method == "random_search":
        events = random_search_events(
            protocol=PROTOCOL, task=task.name, bo_seed=seed,
            candidate_ids_ordered=[str(i) for i in range(len(task.curves))],
            curves=task.curves, costs=task.costs, cap_seconds=cap, max_iterations=ceiling,
        )
        rows = [dict(chosen=e["proposed_config_index"], budget=e["proposed_epoch"],
                     observed=e["proposed_objective"] if e["accepted"] else None,
                     accepted=e["accepted"], cost=e["cumulative_table_cost_seconds"],
                     f_best=e["incumbent"]) for e in events]
    else:
        first = initial_index(PROTOCOL, task.name, seed, len(task.curves))
        rows = run_asha(
            scores, task.costs, cap=cap, paired_index=first, task_id=task.name, seed=seed,
            permutation_seed=PROTOCOL["random_search"]["domain_label"],
        )
    frame = pd.DataFrame(rows)
    frame["regret"] = best - frame.f_best
    frame["normalized_regret"] = [normalized_regret(v, worst, best) for v in frame.f_best]
    return frame


def main():
    args = tyro.cli(Args)
    if not args.seeds or args.chain_batch < 1:
        raise SystemExit("Provide seeds and a positive chain batch size.")
    surrogate = args.method in ("icicle", "ftpfn")
    if surrogate and args.model is None:
        raise SystemExit("Surrogate methods require --model.")
    if args.model is not None and args.model.suffix == ".safetensors" and args.step is not None:
        raise SystemExit("--step applies to training checkpoints.")
    if not surrogate and (args.model is not None or args.step is not None):
        raise SystemExit("Baseline methods do not use a checkpoint.")

    splits = (args.split,) if args.split is not None else _T_MAX
    tasks = [load_qtmeta(split, dataset, data_dir=args.data_dir)
             for split in splits for dataset in qtmeta_datasets(split, args.data_dir)
             if not args.tasks or qtmeta_task_name(split, dataset) in args.tasks]
    if not tasks or (args.tasks and set(args.tasks) != {t.name for t in tasks}):
        raise SystemExit("Requested Meta-Album tasks were not found.")

    scores = {t.name: raw_scores(t, args.data_dir) for t in tasks} if args.method == "asha" else {}

    if surrogate:
        config_dim = max(t.hps.shape[1] for t in tasks)
        runner, model_info = make_runner(args, config_dim, acquisition_name="mfpi-cost-cooled")
    else:
        runner, model_info = None, {"checkpoint": None, "step": None}

    metadata = dict(
        runner="qtmeta-time-v8",
        feature_seed=args.feature_seed,
        method=args.method,
        benchmark="qtmeta",
        acquisition="mfpi-cost-cooled" if surrogate else None,
        value_map=args.value_map if surrogate else "off",
        chain_batch=args.chain_batch,
        backend="xla" if surrogate else None,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        platform=jax.default_backend(),
        runtime=os.environ.get("ICICLE_IMAGE_KEY"),
        device=jax.local_devices()[0].device_kind,
        dtype="float64" if args.method == "asha" else "float32",
        **model_info,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".partial.parquet")
    frames = []

    for split, cap in _T_MAX.items():
        group = [t for t in tasks if t.name.split("/")[1] == split]
        if not group:
            continue
        ceiling = max(structural_ceiling(t.curves, t.costs, cap) for t in group)
        chains = [(t, seed) for t in group for seed in args.seeds]
        capacity = max(len(t.curves) for t in group)

        for start in range(0, len(chains), args.chain_batch):
            batch = chains[start:start + args.chain_batch]
            if surrogate:
                starts = [initial_index(PROTOCOL, t.name, seed, len(t.curves)) for t, seed in batch]
                if any(t.costs[first, 0] > cap for (t, _), first in zip(batch, starts, strict=True)):
                    raise ValueError("Initial observation exceeds the time budget.")
                _, proposals = runner(
                    batch, n_iters=max(2, ceiling), n_cfg_pad=capacity,
                    budget=1.0, initial_configs=starts, exact_cap=True,
                    use_table_costs=True, cost_scale=cap, return_proposals=True,
                )
                current = [exact_trajectory(t, seed, p, cap)
                           for (t, seed), p in zip(batch, proposals, strict=True)]
            else:
                current = [baseline_trajectory(t, seed, args.method, cap, ceiling, scores.get(t.name))
                           for t, seed in batch]

            for (task, seed), frame in zip(batch, current, strict=True):
                frames.append(frame.assign(**metadata, task=task.name, seed=seed,
                                           iteration=np.arange(len(frame)), iters=ceiling,
                                           budget_seconds=cap))

            pd.concat(frames, ignore_index=True).to_parquet(partial)
            print(f"{args.method} {split}: {min(start + args.chain_batch, len(chains))}/{len(chains)}", flush=True)

    partial.replace(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
