"""Prepare figure caches from explicit HPO outputs or selected final-paper inputs.

Aggregate validated results once before running cache-only renderers.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import tyro

from reproduction.bench import benchmark_tasks


def build_consts(benches, data_dir):
    consts = {}
    for bench in benches:
        out = {}
        for _name, load in benchmark_tasks(bench, data_dir):
            t = load()
            out[t.name] = [float(np.nanmax(t.curves)), float(np.nanmin(t.curves))]
        consts[bench] = out
        print(f"consts[{bench}]: {len(out)} tasks", flush=True)
    return consts


def aggregate_inputs(inputs, out, data_dir):
    """Build a separate cache from explicitly selected current-format outputs."""
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("format") != "hpo-explicit":
        raise ValueError("Choose a separate cache directory for current HPO outputs.")
    if (out / "cells").exists():
        raise ValueError("Choose a cache directory without historical cell fragments.")
    frames, sources = [], []
    identity = [
        "runner",
        "method",
        "benchmark",
        "checkpoint",
        "step",
        "acquisition",
        "value_map",
        "feature_seed",
        "backend",
        "jax_version",
        "jaxlib_version",
        "platform",
        "runtime",
        "device",
        "dtype",
        "chain_batch",
    ]
    for path in inputs:
        if ".partial" in path.name:
            raise ValueError(f"Incomplete result: {path}")
        frame = pd.read_parquet(path)
        if frame.empty or not set(identity).issubset(frame.columns):
            raise ValueError(f"Expected current HPO output: {path}")
        settings = frame[identity].drop_duplicates()
        if len(settings) != 1:
            raise ValueError(f"Multiple experiment settings in {path}; write separate result files")
        setting = settings.iloc[0]
        method, checkpoint, step = setting["method"], setting["checkpoint"], setting["step"]
        frame["arm"] = method if pd.isna(checkpoint) else f"{method}:{checkpoint}@{step}"
        numerical = settings.drop(columns=["method", "benchmark", "checkpoint", "step"]).to_json(
            orient="records"
        )
        tag = setting["runner"] + "-" + hashlib.sha256(numerical.encode()).hexdigest()[:12]
        suffix = {"adaptive": "-vma", "fixed": "-vmf", "off": "-raw"}[setting["value_map"]]
        frame["tag"] = tag + suffix
        frame["bench"] = frame["benchmark"]
        frame["train_seed"] = -1
        frame["is_final"] = frame.iteration.eq(
            frame.groupby(["task", "seed"]).iteration.transform("max")
        )
        frames.append(frame)
        sources.append(
            dict(
                path=str(path.resolve()),
                benchmark=setting["benchmark"],
                size=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["arm", "tag", "bench", "iters", "task", "seed", "iteration"]).any():
        raise ValueError("Overlapping trajectories in the selected outputs")
    benchmarks = sorted(combined.bench.unique())
    constants = build_consts(benchmarks, data_dir)
    temporary = out / "cells.partial.parquet"
    combined.to_parquet(temporary)
    meta_path.unlink(missing_ok=True)
    temporary.replace(out / "cells.parquet")
    (out / "consts.json").write_text(json.dumps(constants))
    meta_path.write_text(
        json.dumps(dict(format="hpo-explicit", benchmarks=benchmarks, sources=sources), indent=2)
    )
    print(f"Wrote {len(combined)} rows to {out}")


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261106
BO_SEEDS = tuple(range(20))
SUITES = {"lcbench_ifbo": 35, "pd1_ifbo": 28, "taskset_ifbo": 24, "arlbench_mean": 42}
SPLITS = {"micro": (30, 3600), "mini": (30, 14400), "extended": (26, 57600)}


def bootstrap_weights(n_tasks, n_runs=20, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Two-level bootstrap; each repeated task occurrence gets independent run draws.

    Flattened weights multiply task x run arrays. Reusing them across methods and
    time points preserves pairing and samples whole trajectories.
    """
    rng = np.random.default_rng(seed)
    tasks = rng.integers(n_tasks, size=(draws, n_tasks))
    counts = rng.multinomial(n_runs, np.full(n_runs, 1 / n_runs), size=(draws, n_tasks))
    weights = np.zeros((draws, n_tasks, n_runs), dtype=np.float64)
    rows = np.arange(draws)
    for occurrence in range(n_tasks):
        np.add.at(weights, (rows, tasks[:, occurrence]), counts[:, occurrence])
    return weights.reshape(draws, -1) / (n_tasks * n_runs)


def interval(values, weights):
    flat = np.asarray(values, dtype=np.float64).reshape(weights.shape[1], -1)
    draws = weights @ flat
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return dict(mean=flat.mean(0).tolist(), lower=lo.tolist(), upper=hi.tolist())


def summarize(array, metrics, weights):
    result = interval(array, weights)
    result["metrics"] = {name: interval(value, weights) for name, value in metrics.items()}
    result["task_runs"] = {name: value.tolist() for name, value in metrics.items()}
    return result


def checked_table(path, spec, benchmark, sources):
    """Read one numerical implementation; reject changed identities and partial outputs."""
    import pyarrow.parquet as pq

    path = Path(path)
    if ".partial" in path.name:
        raise ValueError(f"Incomplete output: {path}")
    table = pq.read_table(path)
    expected = dict(spec["settings"], benchmark=benchmark)
    batches = spec.get("chain_batch_by_split")
    if batches is not None:
        if benchmark != "qtmeta" or "chain_batch" in expected:
            raise ValueError("chain_batch_by_split requires qtmeta without a global chain_batch")
        if set(batches) != set(SPLITS) or any(
            type(v) is not int or v < 1 for v in batches.values()
        ):
            raise ValueError("Declare a positive chain batch for every Meta-Album split")
        splits = [task.split("/")[1] for task in table["task"].to_pylist()]
        actual = table["chain_batch"].to_pylist()
        if any(split not in batches or batch != batches[split]
               for split, batch in zip(splits, actual, strict=True)):
            raise ValueError(f"{path}: chain_batch disagrees with chain_batch_by_split")
    for name, value in expected.items():
        actual = table[name].unique().to_pylist()
        if name == "checkpoint":
            actual = [str(Path(x).resolve()) if x else None for x in actual]
            value = str(Path(value).resolve()) if value else None
        if actual != [value]:
            raise ValueError(f"{path}: {name}={actual}, expected {value!r}")
    identities = [
        "runner",
        "method",
        "checkpoint",
        "step",
        "acquisition",
        "value_map",
        "feature_seed",
        "backend",
        "jax_version",
        "jaxlib_version",
        "runtime",
        "platform",
        "device",
        "dtype",
        "chain_batch",
    ]
    if batches is not None:
        identities.remove("chain_batch")  # Checked per row against the explicit split recipe above.
    identity = {key: table[key].unique().to_pylist() for key in identities}
    if any(len(values) != 1 for values in identity.values()):
        raise ValueError(f"Mixed numerical implementations: {path}")
    previous = spec.setdefault("_identity", identity)
    if previous != identity:
        raise ValueError(f"Numerical implementation differs between shards: {path}")
    if spec["settings"]["method"] == "icicle":
        config = json.loads(Path(spec["model_config"]).read_text())
        if config["model"].get("curve_identity", "none") != "none":
            raise ValueError("Paper ICICLE must have no curve identity tag")
    sources[str(path.resolve())] = dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    return table


def hpo_arrays(spec, suite, names, ladders, sources):
    table = checked_table(spec["paths"][0], spec, suite, sources)
    if len(spec["paths"]) != 1:
        raise ValueError("Expected one complete HPO result per model and benchmark")
    task_index = {name: i for i, name in enumerate(names)}
    task = np.array([task_index[x] for x in table["task"].to_pylist()])
    seed, iteration = table["seed"].to_numpy(), table["iteration"].to_numpy()
    if not np.isin(seed, BO_SEEDS).all() or (iteration < 0).any() or (iteration >= 1000).any():
        raise ValueError("Unexpected HPO seed or iteration")
    key = task * 20000 + seed * 1000 + iteration
    order = np.argsort(key)
    if not np.array_equal(key[order], np.arange(len(names) * 20000)):
        raise ValueError(f"{suite}: expected every task x BO seed 0--19 x iteration 0--999 once")
    raw = table["regret"].to_numpy()[order].reshape(len(names), 20, 1000).astype(float)
    spans = np.array([max(float(ladders[t][0]) - float(ladders[t][-1]), 1e-12) for t in names])
    norm = np.clip(raw / spans[:, None, None], 0, 1)
    if not np.isfinite(norm).all() or (np.diff(norm, axis=2) > 1e-6).any():
        raise ValueError("Invalid incumbent trajectory")
    return norm, dict(endpoint=norm[:, :, -1], anytime=norm.mean(2))


def time_trajectory(cost, norm, cap, points=512):
    cost, norm = np.asarray(cost, float), np.asarray(norm, float)
    if (
        not len(cost)
        or not np.isfinite(cost).all()
        or not np.isfinite(norm).all()
        or cost[0] < 0
        or (np.diff(cost) < 0).any()
        or cost[-1] > cap
        or (np.diff(norm) > 1e-6).any()
        or (norm < 0).any()
        or (norm > 1).any()
    ):
        raise ValueError("Invalid time-budget trajectory")
    pos = np.searchsorted(cost, np.linspace(0, cap, points), side="right") - 1
    curve = np.where(pos < 0, 1.0, norm[np.maximum(pos, 0)])
    area = np.dot(np.diff(np.r_[0.0, cost, cap]), np.r_[1.0, norm]) / cap
    return curve, norm[-1], area


def validate_native_identity(document, spec):
    """Keep native runtime/runner and fold-specific checkpoint identities fixed."""
    identity = {key: document[key] for key in ("torch", "hip", "runner_sha256")}
    if spec.setdefault("native_identity", identity) != identity:
        raise ValueError("Mixed native Quick-Tune numerical implementations")
    run = document["run"]
    fold = f"{run['split']}/{run['fold']}"
    hashes = {
        key: value
        for key, value in document["checkpoint_hashes"].items()
        if key.endswith("_sha256")
    }
    if not {"cost_sha256", "performance_sha256", "training_manifest_sha256"} <= hashes.keys():
        raise ValueError("Missing native checkpoint identities")
    checkpoints = spec.setdefault("native_checkpoints", {})
    if checkpoints.setdefault(fold, hashes) != hashes:
        raise ValueError("Changed native checkpoint within held-out fold")


def qt_arrays(spec, names, bounds, sources):
    import gzip

    arrays = np.full((len(names), 20, 512), np.nan)
    metrics = {k: np.full((len(names), 20), np.nan) for k in ("endpoint", "anytime")}
    task_index = {name: i for i, name in enumerate(names)}

    def add(task, seed, cost, norm, best=None):
        i = task_index[task]
        if seed not in BO_SEEDS or not np.isnan(arrays[i, seed]).all():
            raise ValueError("Unexpected or duplicate BO seed")
        lo, hi = bounds[task]
        if best is not None and not np.allclose(
            norm, np.clip((hi - np.asarray(best)) / (hi - lo), 0, 1), atol=1e-6
        ):
            raise ValueError("Normalized regret disagrees with full table bounds")
        cap = SPLITS[task.split("/")[1]][1]
        curve, endpoint, area = time_trajectory(cost, norm, cap)
        arrays[i, seed] = curve
        metrics["endpoint"][i, seed], metrics["anytime"][i, seed] = endpoint, area

    if "native_root" in spec:
        for task in names:
            for seed in BO_SEEDS:
                path = Path(spec["native_root"]) / (task.replace("/", "__") + f"__b{seed}.json.gz")
                with gzip.open(path, "rt") as stream:
                    document = json.load(stream)
                validate_native_identity(document, spec)
                run = document["run"]
                if (run["task"], run["meta_seed"], run["bo_seed"]) != (
                    task,
                    spec["training_seed"],
                    seed,
                ) or run["truncated_smoke"]:
                    raise ValueError(f"Wrong native Quick-Tune run: {path}")
                ev = run["events"]
                if any(e.get("failure") for e in ev):
                    raise ValueError(f"Native failure: {path}")
                add(
                    task,
                    seed,
                    [e["cumulative_table_cost_seconds"] for e in ev],
                    [e["normalized_regret"] for e in ev],
                    [e["incumbent"] for e in ev],
                )
                sources[str(path.resolve())] = dict(
                    size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns
                )
    else:
        for path in spec["paths"]:
            table = checked_table(path, spec, "qtmeta", sources)
            task = np.array([task_index[x] for x in table["task"].to_pylist()])
            seed, iteration = table["seed"].to_numpy(), table["iteration"].to_numpy()
            if not np.isin(seed, BO_SEEDS).all():
                raise ValueError("Unexpected time-run seed")
            order = np.lexsort((iteration, seed, task))
            keys = (task * 20 + seed)[order]
            starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
            cost, norm, best = [
                table[k].to_numpy()[order] for k in ("cost", "normalized_regret", "f_best")
            ]
            for a, b in zip(starts[:-1], starts[1:], strict=True):
                if not np.array_equal(iteration[order][a:b], np.arange(b - a)):
                    raise ValueError("Missing/duplicate time-run iterations")
                task_name, bo_seed = names[keys[a] // 20], int(keys[a] % 20)
                cap = SPLITS[task_name.split("/")[1]][1]
                if not np.all(table["budget_seconds"].to_numpy()[order][a:b] == cap):
                    raise ValueError("Wrong time budget")
                add(task_name, bo_seed, cost[a:b], norm[a:b], best[a:b])
    if not np.isfinite(arrays).all():
        raise ValueError("Incomplete Quick-Tune task x BO seed coverage")
    return arrays, metrics


def paired_results(metrics, weights, first, second):
    result = {}
    for name in ("endpoint", "anytime"):
        delta = metrics[first][name] - metrics[second][name]
        result[name] = interval(delta, weights)
        means = delta.mean(1)
        result[name]["task_wins_ties_losses"] = [
            int((means < -1e-12).sum()),
            int((abs(means) <= 1e-12).sum()),
            int((means > 1e-12).sum()),
        ]
    return result


def aggregate_paper(manifest: Path, out: Path, data_dir=None):
    """Aggregate explicit final-paper inputs once; renderers consume only this cache."""
    from reproduction.bench.qtmeta import raw_scores

    config = json.loads(manifest.read_text())
    if config.get("schema") != "icicle-paper-inputs-v1":
        raise ValueError("Expected icicle-paper-inputs-v1 manifest")
    result = dict(
        schema="icicle-paper-statistics-v1",
        inputs=config,
        panels={},
        sources={},
        uncertainty=dict(
            method="two-level percentile bootstrap",
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
            level=0.95,
            pointwise=True,
            bo_seeds=list(BO_SEEDS),
        ),
    )
    for suite, count in SUITES.items():
        ladders = {}
        for name, load in benchmark_tasks(suite, data_dir):
            task = load()
            ladders[name] = np.sort(np.nanmax(task.curves, axis=1))[::-1]
        names = sorted(ladders)
        if len(names) != count:
            raise ValueError(f"{suite}: wrong task count")
        weights = bootstrap_weights(count)
        panel = dict(tasks=names, x=list(range(1000)), models={})
        metrics = {}
        for key, spec in config["hpo"][suite].items():
            array, metrics[key] = hpo_arrays(spec, suite, names, ladders, result["sources"])
            panel["models"][key] = summarize(array, metrics[key], weights)
            print(f"Aggregated {suite}/{key}", flush=True)
        reference = "asha" if suite == "arlbench_mean" else "ftpfn-published"
        panel["comparison"] = paired_results(metrics, weights, "icicle-s0", reference)
        result["panels"][suite] = panel
    bounds = {}
    for name, load in benchmark_tasks("qtmeta", data_dir):
        raw = raw_scores(load(), data_dir)
        bounds[name] = float(np.nanmin(raw)), float(np.nanmax(raw))
    names = sorted(bounds)
    if len(names) != 86:
        raise ValueError("Expected 86 Meta-Album tasks")
    arrays, metrics = {}, {}
    for key, spec in config["qtmeta"].items():
        arrays[key], metrics[key] = qt_arrays(spec, names, bounds, result["sources"])
        print(f"Loaded qtmeta/{key}", flush=True)
    for split, (count, cap) in SPLITS.items():
        ix = [i for i, task in enumerate(names) if task.split("/")[1] == split]
        if len(ix) != count:
            raise ValueError("Wrong split task count")
        weights = bootstrap_weights(count)
        local = {key: {k: v[ix] for k, v in value.items()} for key, value in metrics.items()}
        panel = dict(
            tasks=[names[i] for i in ix], x=np.linspace(0, cap / 3600, 512).tolist(), models={}
        )
        for key, array in arrays.items():
            panel["models"][key] = summarize(array[ix], local[key], weights)
        panel["comparison"] = paired_results(local, weights, "icicle-s0", "quicktune-s0")
        result["panels"]["qtmeta_" + split] = panel
        print(f"Aggregated qtmeta/{split}", flush=True)
    weights = bootstrap_weights(86)
    result["qtmeta_overall"] = {
        key: {name: interval(value, weights) for name, value in values.items()}
        for key, values in metrics.items()
    }
    result["qtmeta_comparison"] = paired_results(metrics, weights, "icicle-s0", "quicktune-s0")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "paper-statistics.json"
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(result, allow_nan=False))
    temporary.replace(path)
    print(f"Wrote {path}", flush=True)


def main(
    data_dir: str | None = None,
    inputs: tuple[Path, ...] = (),
    out: Path | None = None,
    paper: bool = False,
    manifest: Path | None = None,
):
    if out is None:
        raise ValueError("Specify --inputs and --out, or --paper --manifest and --out")
    if paper:
        if inputs or manifest is None:
            raise ValueError("--paper requires --manifest, without --inputs")
        aggregate_paper(manifest, out, data_dir)
    elif inputs and manifest is None:
        aggregate_inputs(inputs, out, data_dir)
    else:
        raise ValueError("Select --inputs and --out, or --paper --manifest and --out")


if __name__ == "__main__":
    tyro.cli(main)
