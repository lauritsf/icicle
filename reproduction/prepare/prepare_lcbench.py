"""Stream the original LCBench archive into ICICLE's benchmark input tables.

Only the raw validation balanced accuracy, epoch and hyperparameter columns are kept.
The production adapter applies epoch filtering and normalization. Streaming one dataset at
a time avoids loading the multi-gigabyte JSON document into memory as one Python object.
"""

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from reproduction.bench._hp_spaces import LCBENCH_SPACE

ARCHIVE_URL = "https://figshare.com/ndownloader/files/21188607"


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def iter_datasets(stream, chunk_size=16 * 1024 * 1024):
    """Decode one top-level dataset at a time, preserving Python JSON Infinity/NaN tokens."""
    decoder = json.JSONDecoder()
    buffer, position = "", 0

    def extend():
        nonlocal buffer, position
        block = stream.read(chunk_size)
        buffer = buffer[position:] + block
        position = 0
        return bool(block)

    def token():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                return buffer[position]
            if not extend():
                raise ValueError("unexpected end of LCBench JSON")

    def decode():
        nonlocal position
        token()
        while True:
            try:
                value, position = decoder.raw_decode(buffer, position)
                return value
            except json.JSONDecodeError:
                if not extend():
                    raise

    if token() != "{":
        raise ValueError("expected a top-level dataset mapping")
    position += 1
    while token() != "}":
        name = decode()
        if not isinstance(name, str) or token() != ":":
            raise ValueError("expected a dataset name followed by a colon")
        position += 1
        configs = decode()
        if not isinstance(configs, dict):
            raise ValueError("expected a configuration mapping")
        yield name, configs
        delimiter = token()
        if delimiter == "}":
            break
        if delimiter != ",":
            raise ValueError("expected a comma between datasets")
        position += 1
        if token() == "}":
            raise ValueError("trailing comma in dataset mapping")
    position += 1
    while True:
        if buffer[position:].strip():
            raise ValueError("unexpected content after the dataset mapping")
        position = len(buffer)
        if not extend():
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="reuse the original data_2k.zip download")
    args = parser.parse_args()
    output = args.data_dir / "lcbench-tabular"
    if output.exists():
        parser.error(
            "choose a data root without lcbench-tabular; existing inputs are not overwritten"
        )
    sources = output / "sources"
    sources.mkdir(parents=True)
    archive = args.archive
    if archive is None:
        archive = sources / "data_2k.zip"
        partial = archive.with_suffix(".zip.partial")
        print(f"Downloading {ARCHIVE_URL}", flush=True)
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response, partial.open("xb") as f:
            shutil.copyfileobj(response, f)
        partial.rename(archive)
    manifest = {
        "benchmark": "lcbench_ifbo",
        "url": ARCHIVE_URL,
        "archive_sha256": file_digest(archive),
        "archive_bytes": archive.stat().st_size,
        "metric": "raw validation balanced accuracy; all original epochs retained",
        "tables": {},
    }
    with zipfile.ZipFile(archive) as compressed, compressed.open("data_2k.json") as raw:
        for task, configs in iter_datasets(io.TextIOWrapper(raw, encoding="utf-8")):
            if Path(task).name != task or task in manifest["tables"]:
                raise ValueError(f"unexpected or duplicate task name: {task}")
            rows = []
            for identity, data in configs.items():
                log, config = data["log"], data["config"]
                rows.append(
                    pd.DataFrame(
                        {
                            "epoch": log["epoch"],
                            "val_balanced_accuracy": log["Train/val_balanced_accuracy"],
                        }
                    ).assign(id=identity, **{name: config[name] for name, *_ in LCBENCH_SPACE})
                )
            table = pd.concat(rows, ignore_index=True).set_index(["id", "epoch"]).sort_index()
            if len(configs) != 2000 or table.index.duplicated().any():
                raise ValueError(f"{task}: expected 2000 distinct configuration curves")
            path = output / f"{task}.parquet"
            table.to_parquet(path)
            manifest["tables"][task] = {"rows": len(table), "sha256": file_digest(path)}
            print(f"Prepared {task}: {len(table)} observations", flush=True)
            (output / "PREPARATION.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if len(manifest["tables"]) != 35:
        raise ValueError("expected all 35 LCBench tasks")
    manifest["complete"] = True
    (output / "PREPARATION.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
