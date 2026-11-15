"""Check exported ONNX predictions and measure serving latency.

Compare predictions with reference logits, then sweep point,
configuration and hyperparameter counts. Timings include the ONNX Runtime
call and its input/output transfers. The portable graph on CPU supplies the
reference. Save progress after each case.

    python -m experiments.serving_bench --model-path MODEL.onnx \
        --out results/serving.json --provider cpu
"""

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
import tyro

from icicle.serving import make_session, orthonormal_frame

BASE = {"p": 1000, "c": 128, "d": 33}
AXES = {
    "p": [500, 1000, 2000, 4000, 8000, 16000, 32000],
    "c": [32, 128, 512, 2000, 4096],
    "d": [4, 10, 33, 64, 128],
}
PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "dml": "DmlExecutionProvider",
}


@dataclass
class Args:
    model_path: Path
    out: Path
    provider: str = "cpu"
    graph: Literal["portable", "fused"] = "portable"
    repeats: int = 30
    max_points: int | None = None
    threads: int | None = None


def make_inputs(rng, p, c, d, model):
    frame = orthonormal_frame(rng, d, model["emsize"])
    return {
        "config_hps": rng.uniform(size=(c, d)).astype(np.float32),
        "feature_identity": frame,
        "num_params": np.array(d, np.int64),
        "point_config_idx": rng.integers(0, c, size=p).astype(np.int64),
        "fidelities": rng.uniform(0.02, 1.0, size=p).astype(np.float32),
        "y": rng.uniform(size=p).astype(np.float32),
        "single_eval_pos": np.array(max(1, int(p * 0.75)), np.int64),
    }


def time_cell(session, feeds, repeats):
    for _ in range(3):
        session.run(["logits"], feeds)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.run(["logits"], feeds)
        times.append((time.perf_counter() - start) * 1000)
    median, p10, p90 = np.percentile(times, [50, 10, 90])
    return {"median_ms": float(median), "p10_ms": float(p10), "p90_ms": float(p90)}


def main():
    args = tyro.cli(Args)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive.")
    if args.max_points is not None and args.max_points < 2:
        raise SystemExit("--max-points must be at least two.")
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads must be positive.")

    manifest = json.loads(args.model_path.with_suffix(".json").read_text())
    model = manifest["model"]
    model_path = (args.model_path.with_name(f"{args.model_path.stem}_fused.onnx")
                  if args.graph == "fused" else args.model_path)
    provider = PROVIDERS.get(args.provider, args.provider)
    session = make_session(model_path, [provider], intra_op_num_threads=args.threads)

    d = min(model["config_dim"], model["emsize"])
    feeds = make_inputs(np.random.default_rng(0), min(256, args.max_points or 256), 16, d, model)
    cpu = make_session(args.model_path, ["CPUExecutionProvider"], intra_op_num_threads=args.threads)
    expected, = cpu.run(["logits"], feeds)
    del cpu

    logits, = session.run(["logits"], feeds)
    if logits.shape != expected.shape or not np.isfinite(logits).all() or not np.isfinite(expected).all():
        raise ValueError("Reference check produced mismatched shapes or nonfinite logits.")
    delta = float(np.max(np.abs(logits - expected)))
    if delta >= 1e-2:
        raise ValueError(f"Reference logits differ by {delta:.3g}")
    print(f"Reference maximum error: {delta:.3g}", flush=True)

    report = {
        "model_path": str(model_path.resolve()),
        "manifest": manifest,
        "provider": provider,
        "provider_options": session.get_provider_options(),
        "host": platform.node(),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "reference": "portable-onnx-cpu",
        "reference_max_error": delta,
        "repeats": args.repeats,
        "threads": args.threads,
        "max_points": args.max_points,
        "cells": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    axes = dict(AXES)
    if args.graph == "fused":
        axes["p"] = [*axes["p"], 65536]

    for axis, values in axes.items():
        for value in values:
            shape = {**BASE, axis: value}
            cell = {"axis": axis, **shape}
            if args.max_points is not None and shape["p"] > args.max_points:
                cell["skipped"] = "Exceeds --max-points"
            elif shape["d"] > model["emsize"]:
                cell["skipped"] = "Haar frame exceeds model width"
            else:
                try:
                    feeds = make_inputs(np.random.default_rng(0), **shape, model=model)
                    cell.update(time_cell(session, feeds, args.repeats))
                except Exception as error:
                    cell["error"] = str(error)

            report["cells"].append(cell)
            partial = args.out.with_suffix(".partial.json")
            partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            partial.replace(args.out)
            print(cell, flush=True)
            if "error" in cell:
                break
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
