"""Run one HPO method on one benchmark and save its regret trajectories.

    python -m experiments.hpo_regret --method icicle --model RUN_DIR \
        --benchmark pd1_ifbo --out results/icicle_pd1.parquet
    python -m experiments.hpo_regret --method asha \
        --benchmark pd1_ifbo --out results/asha_pd1.parquet
    python -m experiments.hpo_regret --method ftpfn --model FTPFN.safetensors \
        --benchmark pd1_ifbo --out results/ftpfn_pd1.parquet

"""

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass  # noqa: E402
from functools import cache, partial  # noqa: E402
from typing import Literal  # noqa: E402

import jax  # noqa: E402
import jaxlib  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tyro  # noqa: E402

from icicle_jax.loading import load_eval_model  # noqa: E402
from icicle_jax.bar_distribution import BarDistribution  # noqa: E402
from icicle_jax.ftpfn import load_ftpfn, make_ftpfn_scan_score_fn  # noqa: E402
from reproduction.baselines.scheduler import run_random_search  # noqa: E402
from reproduction.bench import BENCHMARKS, benchmark_tasks  # noqa: E402
from reproduction.jax.baseline_scan import run_asha_scan, run_hyperband_scan  # noqa: E402
from reproduction.jax.gpblackbox_scan import run_gpblackbox_scan  # noqa: E402
from reproduction.jax.gpft_scan import run_gpft_scan  # noqa: E402
from reproduction.jax.hpo_scan import run_freeze_thaw_scan  # noqa: E402


@dataclass
class Args:
    method: Literal["icicle", "ftpfn", "random_search", "asha", "asha_neps", "hyperband", "gp_ft", "gp_blackbox"]
    benchmark: str
    out: Path
    model: Path | None = None
    step: int | None = None
    feature_seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2)
    iters: int = 1000  # Training steps per task; gp_blackbox counts full training runs.
    chain_batch: int = 32
    value_map: Literal["off", "fixed", "adaptive"] = "off"
    data_dir: str | None = None


def run_random_batch(chains, *, n_iters, n_cfg_pad):
    return [run_random_search(task, n_iters, seed) for task, seed in chains]


BASELINE_RUNNERS = {
    "random_search": run_random_batch,
    "asha": partial(run_asha_scan, variant="li"),
    "asha_neps": partial(run_asha_scan, variant="neps"),
    "hyperband": run_hyperband_scan,
    "gp_ft": run_gpft_scan,
    "gp_blackbox": run_gpblackbox_scan,
}


def make_runner(args, config_dim, **scan_options):
    if args.method in BASELINE_RUNNERS:
        return BASELINE_RUNNERS[args.method], {"checkpoint": None, "step": None}

    if args.method == "ftpfn" and args.model.suffix == ".safetensors":
        model = load_ftpfn(args.model)
        criterion = BarDistribution(borders=jnp.linspace(0.0, 1.0, 1001))
        step = None
    else:
        model, criterion, config, step = load_eval_model(
            args.model, step=args.step,
            config_dim=config_dim if args.method == "icicle" else None, backend="xla",
        )
        if config["model"].get("architecture", "icicle") != args.method:
            raise ValueError(f"--method {args.method} does not match the checkpoint architecture")
        if args.method == "ftpfn":
            model = model.model

    borders = np.asarray(criterion.borders)
    options = dict(acquisition_name="mfpi-random", value_map=args.value_map,
                   feature_seed=args.feature_seed)
    options.update(scan_options)
    if args.method == "ftpfn":
        if config_dim > model.config_dim:
            raise ValueError(f"FT-PFN supports {model.config_dim} hyperparameters; benchmark has {config_dim}")

        @cache
        def scorer(n_hps, capacity, n_iters):
            return make_ftpfn_scan_score_fn(
                model, criterion, borders=borders,
                n_hps=n_hps, max_distinct_curves=min(capacity, n_iters),
            )

        def runner(chains, *, n_iters, n_cfg_pad, **kwargs):
            n_hps = chains[0][0].hps.shape[1]
            return run_freeze_thaw_scan(
                None, criterion, chains, borders=borders,
                score_fn=scorer(n_hps, n_cfg_pad, n_iters), config_dim=n_hps,
                n_iters=n_iters, n_cfg_pad=n_cfg_pad, **options, **kwargs,
            )
    else:
        runner = partial(
            run_freeze_thaw_scan, model, criterion, borders=borders,
            config_dim=config_dim, **options,
        )
    return runner, {"checkpoint": str(args.model.resolve()), "step": step}


def trajectory_frame(result, task, seed, metadata):
    return pd.DataFrame({
        **metadata,
        "task": task.name,
        "seed": seed,
        "iteration": np.arange(len(result.chosen)),
        "chosen": result.chosen,
        "budget": result.budget,
        "observed": result.observed,
        "f_best": result.f_best,
        "regret": result.regret,
        "cost": result.cost if result.cost is not None else np.nan,
    })


def main():
    args = tyro.cli(Args)
    if args.benchmark not in BENCHMARKS:
        raise SystemExit(f"Choose one benchmark from {BENCHMARKS}")
    if args.iters < 1 or args.chain_batch < 1 or not args.seeds:
        raise SystemExit("Use a positive iteration count and batch size, and at least one seed.")
    if args.method in ("gp_ft", "gp_blackbox") and args.iters < 4:
        raise SystemExit("GP methods require at least four iterations.")
    if args.method in ("icicle", "ftpfn"):
        if args.model is None or args.iters < 2:
            raise SystemExit("Surrogate methods require --model and at least two iterations.")
        if args.method == "ftpfn" and args.model.suffix == ".safetensors" and args.step is not None:
            raise SystemExit("--step applies to training checkpoints.")
    elif args.model is not None or args.step is not None or args.value_map != "off":
        raise SystemExit("--model, --step and --value-map apply to surrogate methods.")

    tasks = [load() for _, load in benchmark_tasks(args.benchmark, args.data_dir)]
    if not tasks:
        raise SystemExit("No benchmark tasks found.")
    config_dim = max(task.hps.shape[1] for task in tasks)
    runner, model_info = make_runner(args, config_dim)
    metadata = {
        "runner": "hpo-regret-v5",
        "feature_seed": args.feature_seed,
        "method": args.method,
        "benchmark": args.benchmark,
        "iters": args.iters,
        "chain_batch": args.chain_batch,
        "acquisition": "mfpi-random" if args.method in ("icicle", "ftpfn") else None,
        "value_map": args.value_map,
        "backend": "xla" if args.method in ("icicle", "ftpfn") else None,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "runtime": os.environ.get("ICICLE_IMAGE_KEY"),
        "device": jax.local_devices()[0].device_kind,
        "dtype": "float32",
        "platform": jax.default_backend(),
        **model_info,
    }

    tasks_by_hp_count = {}
    for task in tasks:
        hp_count = task.hps.shape[1]
        tasks_by_hp_count.setdefault(hp_count, []).append(task)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.out.with_suffix(".partial.parquet")

    frames = []
    completed = 0
    total = len(tasks) * len(args.seeds)

    for group in tasks_by_hp_count.values():
        config_capacity = max(task.curves.shape[0] for task in group)
        chains = [(task, seed) for task in group for seed in args.seeds]

        for start in range(0, len(chains), args.chain_batch):
            batch = chains[start:start + args.chain_batch]
            results = runner(batch, n_iters=args.iters, n_cfg_pad=config_capacity)

            for (task, seed), result in zip(batch, results, strict=True):
                frame = trajectory_frame(result, task, seed, metadata)
                frames.append(frame)

            pd.concat(frames, ignore_index=True).to_parquet(partial_path)

            completed += len(batch)
            print(f"{args.method}: {completed}/{total} chains", flush=True)

    partial_path.replace(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
