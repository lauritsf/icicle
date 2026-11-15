"""Compare checkpoint predictions on sampled learning curves.

    python -m experiments.eval RUN_A RUN_B --benchmark pd1_ifbo \
        --out results/predictions.parquet
"""

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp
import pandas as pd  # noqa: E402
import tyro  # noqa: E402

from icicle_jax.ftpfn import load_ftpfn
from icicle_jax.bar_distribution import BarDistribution
from icicle_jax.loading import load_eval_model  # noqa: E402
from reproduction.bench import BENCHMARKS, EvalProtocol, benchmark_tasks  # noqa: E402
from reproduction.jax.eval import evaluate  # noqa: E402


@dataclass
class Args:
    runs: tyro.conf.Positional[tuple[Path, ...]]
    benchmark: str
    out: Path
    scenarios: int = 20
    n_total: int = 1000
    seed: int = 0
    feature_keys: int = 8
    feature_seed: int = 0
    step: int | None = None
    data_dir: str | None = None


def main():
    args = tyro.cli(Args)
    if args.benchmark not in BENCHMARKS:
        raise SystemExit(f"Choose one benchmark from {BENCHMARKS}")
    if not args.runs or args.scenarios < 1 or args.n_total < 2 or args.feature_keys < 1:
        raise SystemExit("Provide checkpoints, positive scenario/feature counts and at least two points.")

    protocol = EvalProtocol(n_total=args.n_total, n_scenarios=args.scenarios)
    tasks = [(args.benchmark, load()) for _, load in benchmark_tasks(args.benchmark, args.data_dir)]
    if not tasks:
        raise SystemExit("No benchmark tasks found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.out.with_suffix(".partial.parquet")
    completed = []

    for run in args.runs:
        published = run.suffix == ".safetensors"
        if published:
            if args.step is not None:
                raise SystemExit("--step applies to training checkpoints.")
            model = load_ftpfn(run)
            criterion = BarDistribution(borders=jnp.linspace(0, 1, 1001))
            step = None
            config = {"model": {"architecture": "ftpfn", "config_dim": model.config_dim,
                                "emsize": model.epoch_enc.kernel.shape[1]}, "data": {}}
        else:
            model, criterion, config, step = load_eval_model(run, step=args.step)
        architecture = config["model"].get("architecture", "icicle")
        required_dim = max(task.hps.shape[1] for _, task in tasks)
        config_dim = required_dim if architecture == "icicle" else config["model"]["config_dim"]
        if required_dim > config_dim:
            raise ValueError(f"FT-PFN supports {config_dim} hyperparameters; benchmark has {required_dim}")
        print(f"{run}: step {step}", flush=True)

        metadata = {
            "checkpoint": str(run.resolve()),
            "step": step,
            "runner": "eval-v4",
            "architecture": architecture,
            "scenarios": args.scenarios,
            "feature_keys": 1 if published else args.feature_keys,
            "config_dim_eval": config_dim,
            "config_dim_trained": config["model"]["config_dim"],
            "extensions": config.get("prior", {}).get("extensions"),
            "seq_len": config["data"].get("seq_len"),
            "emsize": config["model"]["emsize"],
            "backend": "xla",
            "dtype": "float32",
            "jax_version": jax.__version__,
            "platform": jax.default_backend(),
        }

        def save_progress(current):
            if current.empty:
                return
            current = current.assign(**metadata)
            pd.concat([*completed, current], ignore_index=True).to_parquet(partial_path)

        result = evaluate(
            model, criterion, tasks, protocol,
            config_dim=config_dim,
            seed=args.seed,
            feature_keys=args.feature_keys,
            feature_seed=args.feature_seed,
            log=lambda msg: print(f"  {msg}", flush=True),
            on_task=save_progress,
            published_ftpfn=published,
        )
        if result.empty:
            raise SystemExit(f"No scenarios could be evaluated for {run}.")
        result = result.assign(**metadata)
        completed.append(result)

        per_task = result.groupby("task")[["ll", "mse"]].mean()
        print(per_task.to_string(float_format=lambda value: f"{value:.4f}"))
        print(f"Mean across tasks: {per_task.mean().to_dict()}", flush=True)

    partial_path.replace(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
