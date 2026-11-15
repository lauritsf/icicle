"""Prepare the TaskSet ifBO task set from original Google Research curve archives.

Produces only the columns consumed by ICICLE's adapter: optimizer hyperparameters, raw
validation loss, training step and epoch. Losses are averaged across the five stored seeds
before normalization; step zero remains in the table for the adapter's normalization anchor.
TensorFlow is unnecessary. Downloaded inputs and their hashes remain beside the tables.
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.bench.taskset import TASKSET_IFBO_TASKS, TASKSET_OPTIMIZERS

CONFIG_COMMIT = "1e1ad741605dc8d98e1f1938dd8147adccfc2f0c"
CONFIG_HASHES = {
    "adam4p_wide_grid_1k": "be554f93728021c833963d256d3d1dc348cf3b267d27533987c490a184e64ea7",
    "adam8p_wide_grid_1k": "304de0be545d66895bce30658df669b6cc727b0b861bbb2ceca512cb567aaedd",
}
CURVE_BASE = "https://storage.googleapis.com/gresearch/task_set_data"
CONFIG_BASE = (
    f"https://raw.githubusercontent.com/google-research/google-research/{CONFIG_COMMIT}"
    "/task_set/optimizers/configs"
)


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(url, path, expected=None):
    temporary = path.with_suffix(path.suffix + ".partial")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    digest = file_digest(temporary)
    if expected is not None and digest != expected:
        raise ValueError(f"upstream checksum mismatch: {url}")
    temporary.rename(path)
    return {"url": url, "sha256": digest, "bytes": path.stat().st_size}


def prepare_table(archive, configs):
    with np.load(archive, allow_pickle=False) as data:
        steps, losses = data["xs"], data["ys"]
    if losses.ndim != 4 or losses.shape[:2] != (1000, 5) or losses.shape[2] != len(steps):
        raise ValueError("expected 1000 configurations with five paired seed curves")
    if steps.ndim != 1 or steps[0] != 0 or steps[-1] != 10000 or np.any(np.diff(steps) <= 0):
        raise ValueError("expected the complete increasing 0..10000 training-step grid")
    curves = np.asarray(
        [np.mean([losses[i, seed, :, 1] for seed in range(5)], axis=0) for i in range(1000)]
    )
    hps = pd.DataFrame.from_dict(
        {int(key.rsplit("_seed", 1)[1]): value[0] for key, value in configs.items()}, orient="index"
    ).sort_index()
    if list(hps.index) != list(range(1000)):
        raise ValueError("optimizer identities must match the curve archive's row order")
    table = hps.loc[np.repeat(np.arange(1000), len(steps))].copy()
    table.index.name = "config_id"
    table["epoch"] = np.tile(np.arange(1, len(steps) + 1), 1000)
    table["step"] = np.tile(steps, 1000)
    table["valid1_loss"] = curves.reshape(-1)
    return table.reset_index().set_index(["config_id", "epoch"]).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKSET_IFBO_TASKS, default=TASKSET_IFBO_TASKS
    )
    parser.add_argument(
        "--optimizers", nargs="+", choices=TASKSET_OPTIMIZERS, default=TASKSET_OPTIMIZERS
    )
    args = parser.parse_args()
    output = args.data_dir / "taskset-tabular"
    if output.exists():
        parser.error(
            "choose a data root without taskset-tabular; existing inputs are not overwritten"
        )
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.optimizers)) != len(args.optimizers):
        parser.error("duplicate tasks or optimizers")
    sources = output / "sources"
    sources.mkdir(parents=True)
    manifest = {
        "benchmark": "taskset_ifbo",
        "metric": "valid1_loss: arithmetic mean of all five stored seeds",
        "normalization": "none; the ICICLE adapter applies the declared benchmark protocol",
        "config_commit": CONFIG_COMMIT,
        "inputs": {},
        "tables": {},
    }
    for optimizer in args.optimizers:
        config_name = optimizer.removesuffix("_1k") + ".json"
        config_path = sources / config_name
        manifest["inputs"][config_name] = download(
            f"{CONFIG_BASE}/{config_name}", config_path, CONFIG_HASHES[optimizer]
        )
        configs = json.loads(config_path.read_text())
        for task in args.tasks:
            stem = f"{task}-{optimizer}_10000_replica5"
            archive = sources / f"{stem}.npz"
            url = f"{CURVE_BASE}/{task}/{optimizer}_10000_replica5.npz"
            manifest["inputs"][archive.name] = download(url, archive)
            table = prepare_table(archive, configs)
            destination = output / f"{stem}.parquet"
            table.to_parquet(destination)
            manifest["tables"][destination.name] = {
                "rows": len(table),
                "sha256": file_digest(destination),
            }
            (output / "PREPARATION.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["complete"] = True
    (output / "PREPARATION.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
