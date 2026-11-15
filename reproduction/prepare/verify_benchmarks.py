"""Check prepared benchmark inputs against the recorded task lists and array hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reproduction.bench.scenario import benchmark_tasks


def input_record(task, fields):
    arrays = {}
    for field in fields:
        value = getattr(task, field)
        arrays[field] = None
        if value is not None:
            array = np.ascontiguousarray(value)
            arrays[field] = {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    # Hash the array descriptions as well as their contents so dtype and shape matter.
    encoded = json.dumps(arrays, sort_keys=True, separators=(",", ":")).encode()
    return {
        "configurations": task.hps.shape[0],
        "hyperparameters": task.hps.shape[1],
        "fidelities": task.curves.shape[-1],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def verify(data_dir, manifest_path, suites):
    manifest = json.loads(manifest_path.read_text())
    results = {}
    for suite in suites:
        spec = manifest[suite]
        checks = json.loads((manifest_path.parent / spec["inputs"]).read_text())
        fields = checks["fields"]
        if fields not in (
            ["hps", "curves", "fidelities"],
            ["hps", "curves", "fidelities", "cat_card", "costs"],
        ):
            raise ValueError(f"{suite}: unexpected input fields")
        expected = checks["tasks"]
        entries = benchmark_tasks(suite, data_dir)
        loaders = dict(entries)
        if (
            not expected
            or spec["tasks"] != len(expected)
            or len(entries) != len(loaders)
            or loaders.keys() != expected.keys()
        ):
            raise ValueError(f"{suite}: missing, duplicate or unexpected tasks")
        for name, load in loaders.items():
            task = load()
            if task.name != name:
                raise ValueError(f"{suite}/{name}: loaded task has a different name")
            if input_record(task, fields) != expected[name]:
                raise ValueError(f"{suite}/{name}: prepared inputs differ from the recorded inputs")
            print(f"MATCH {suite}: {name}", flush=True)
        results[suite] = len(loaders)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmarks/core_inputs.json",
    )
    parser.add_argument("--suites", nargs="+")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    suites = args.suites if args.suites is not None else list(manifest)
    if not suites or len(set(suites)) != len(suites) or not set(suites) <= manifest.keys():
        parser.error("choose distinct nonempty suites included in the manifest")
    print(json.dumps({"verified_tasks": verify(args.data_dir, args.manifest, suites)}))


if __name__ == "__main__":
    main()
