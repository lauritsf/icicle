"""Prepare PD1 tabular views from Google's original archive without surrogate dependencies.

The table protocol follows mf-prior-bench's tabular processing: matched phase1 except
ImageNet batch1024 (unmatched phase0), modal curve length, contiguous configuration IDs,
and the published linspace subsampling ladder. A CSV round trip preserves the reference
float parsing boundary. ICICLE's adapter owns metric normalization and incomplete-row
filtering. The input archive and processed-file hashes accompany every preparation.
"""

import argparse
import gzip
import hashlib
import io
import itertools
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.bench.pd1 import PD1_IFBO_TASKS

ARCHIVE_URL = "https://storage.googleapis.com/gresearch/pint/pd1.tar.gz"
HP_COLUMNS = {
    "hps.lr_hparams.decay_steps_factor": "lr_decay_factor",
    "hps.lr_hparams.initial_value": "lr_initial",
    "hps.lr_hparams.power": "lr_power",
    "hps.opt_hparams.momentum": "opt_momentum",
}


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def record_frame(record):
    cost = record["train_cost"]
    if not isinstance(cost, list):
        return None
    cumulative = list(itertools.accumulate(np.nan if value is None else value for value in cost))
    frame = pd.DataFrame(
        {
            "epoch": record["epoch"],
            "valid_error_rate": record["valid/error_rate"],
            "train_cost": cumulative,
        }
    )
    frame = frame.loc[frame.train_cost.notna()].copy()
    for source, target in HP_COLUMNS.items():
        frame[target] = record[source]
    return frame


def enumerate_table(frames):
    table = pd.concat(frames, ignore_index=True)
    text = io.StringIO()
    table.to_csv(text, index=False)
    text.seek(0)
    table = pd.read_csv(text, float_precision="high")
    first = table[list(HP_COLUMNS.values())].drop_duplicates(keep="first").index
    table["id"] = pd.Series(np.arange(1, len(first) + 1), index=first).reindex(table.index).ffill()
    table["id"] = table.id.astype(int)
    lengths = table.groupby("id", sort=True).size()
    values, counts = np.unique(lengths.to_numpy(), return_counts=True)
    modal = values[counts.argmax()]
    table = table.loc[table.id.isin(lengths.index[lengths == modal])].copy()
    identities = table.id.drop_duplicates().to_numpy()
    table["id"] = table.id.map(dict(zip(identities, range(1, len(identities) + 1), strict=True)))
    table["original_steps"] = table.epoch
    table["epoch"] = table.groupby("id", sort=False).cumcount() + 1
    return table.set_index(["id", "epoch"]).sort_index()


def coarsen(table, stride):
    if stride == 1:
        return table
    levels = table.index.get_level_values("epoch").unique().to_numpy()
    retained = np.linspace(1, levels[-1], num=(len(levels) - 1) // stride, endpoint=True, dtype=int)
    if len(retained) < 10:
        raise ValueError("requested coarseness leaves fewer than ten levels")
    selected = (
        table.loc[table.index.get_level_values("epoch").isin(retained)].dropna().reset_index()
    )
    mapping = {level: index + 1 for index, level in enumerate(sorted(selected.epoch.unique()))}
    selected["epoch"] = selected.epoch.map(mapping)
    return selected.set_index(["id", "epoch"]).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="reuse Google's original pd1.tar.gz download")
    args = parser.parse_args()
    output = args.data_dir / "pd1-tabular"
    if output.exists():
        parser.error("choose a data root without pd1-tabular; existing inputs are not overwritten")
    sources = output / "sources"
    sources.mkdir(parents=True)
    archive = args.archive
    if archive is None:
        archive = sources / "pd1.tar.gz"
        partial = archive.with_suffix(".gz.partial")
        print(f"Downloading {ARCHIVE_URL}", flush=True)
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response, partial.open("xb") as f:
            shutil.copyfileobj(response, f)
        partial.rename(archive)
    groups = {f"{d}-{m}-{b}": [] for d, m, b, _ in PD1_IFBO_TASKS}
    names = ("pd1_matched_phase1_results.jsonl.gz", "pd1_unmatched_phase0_results.jsonl.gz")
    with tarfile.open(archive, "r:gz") as compressed:
        members = {
            Path(member.name).name: member for member in compressed.getmembers() if member.isfile()
        }
        for name in names:
            with compressed.extractfile(members[name]) as source, gzip.open(source, "rt") as lines:
                for line in lines:
                    record = json.loads(line)
                    task = f"{record['dataset']}-{record['model']}-{int(record['hps.batch_size'])}"
                    if task not in groups or (task == "imagenet-resnet-1024") != (name == names[1]):
                        continue
                    frame = record_frame(record)
                    if frame is not None and not frame.empty:
                        groups[task].append(frame)
    manifest = {
        "benchmark_views": ["pd1_ifbo"],
        "url": ARCHIVE_URL,
        "archive_sha256": file_digest(archive),
        "archive_bytes": archive.stat().st_size,
        "tables": {},
    }
    for task, frames in groups.items():
        if not frames:
            raise ValueError(f"missing original curves for {task}")
        table = enumerate_table(frames)
        strides = [c for d, m, b, c in PD1_IFBO_TASKS if f"{d}-{m}-{b}" == task]
        for stride in strides:
            suffix = "" if stride is None else f"-{stride}"
            processed = table if stride is None else coarsen(table, stride)
            path = output / f"{task}_tabular{suffix}.parquet"
            processed.to_parquet(path)
            manifest["tables"][path.name] = {"rows": len(processed), "sha256": file_digest(path)}
            print(f"Prepared {path.name}: {len(processed)} observations", flush=True)
    manifest["complete"] = True
    (output / "PREPARATION.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
