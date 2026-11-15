"""Measure inference latency and peak device memory as input sizes vary.

Sweep the number of points, configurations and hyperparameters for ICICLE,
with an optional FT-PFN comparison. Each case runs in a separate process
so memory measurements start fresh. Latency is measured after compilation
and warmup, with inputs already on the device.

    python -m experiments.bench_scaling --out scaling.parquet
    python -m experiments.bench_scaling --out scaling.parquet --ftpfn PATH
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import dataclasses  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal  # noqa: E402

import numpy as np  # noqa: E402
import tyro  # noqa: E402

CENTRE = dict(P=2000, C=512, D=10)
SWEEPS = dict(
    P=[512, 1000, 2000, 4000, 8000, 16000],
    C=[32, 128, 512, 2000, 4096],
    D=[4, 10, 33, 64],
)
FTPFN_CONFIG_DIM = 10
FTPFN_MAX_CURVES = 1000


@dataclasses.dataclass(frozen=True)
class Args:
    out: Path = Path("scaling.parquet")
    config: Path = Path(__file__).resolve().parents[1] / "configs" / "icicle.json"
    ftpfn: str | None = None  # Optional FT-PFN checkpoint.
    repeats: int = 10
    backends: tuple[Literal["xla", "flash"], ...] = ("xla",)
    cell: str | None = None  # Subprocess entry point: one benchmark case encoded as JSON.


def benchmark_cases(args: Args):
    for axis, values in SWEEPS.items():
        for value in values:
            shape = {**CENTRE, axis: value}
            if axis == "C":
                shape["P"] = max(SWEEPS["C"])
            for backend in args.backends:
                yield {"model": "icicle", "backend": backend, "axis": axis, **shape}
            if args.ftpfn is not None:
                yield {"model": "ftpfn", "backend": "xla", "axis": axis, **shape}


def time_forward(forward, args, repeats):
    forward(*args).block_until_ready()  # Compile and warm up before timing.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        forward(*args).block_until_ready()
        times.append(time.perf_counter() - start)
    return times


def benchmark_icicle(spec, repeats, config_path):
    import jax
    from flax import nnx

    from icicle_jax.nn.ragged import next_pow2
    from icicle_jax.loading import ModelConfig, build_model

    p, c, d = spec["P"], spec["C"], spec["D"]
    model_cfg = dataclasses.replace(
        ModelConfig(**json.loads(config_path.read_text())["model"]),
        config_dim=d,
        icl_backend=spec["backend"],
    )
    if model_cfg.architecture != "icicle" or not model_cfg.ragged:
        raise ValueError("Scaling requires an ICICLE configuration with ragged encoding.")
    model = build_model(model_cfg, nnx.Rngs(0))

    rng = np.random.default_rng(0)
    config_hps = rng.uniform(size=(1, c, d)).astype(np.float32)
    point_config_idx = (np.arange(p) % c).astype(np.int32)[None]
    fidelities = rng.uniform(0.05, 1.0, size=(1, p)).astype(np.float32)
    y = rng.uniform(size=(1, p)).astype(np.float32)
    sep = p // 2
    n_pad = next_pow2(c)
    feature_key = jax.random.key(0)

    @nnx.jit
    def forward(model, hps, config_idx, fidelities, y):
        return model.forward_ragged(
            hps, d, config_idx, fidelities, y, sep, n_pad, feature_key=feature_key
        )

    inputs = jax.device_put((config_hps, point_config_idx, fidelities, y))
    jax.block_until_ready(inputs)
    times = time_forward(forward, (model, *inputs), repeats)
    return times, dataclasses.asdict(model_cfg)


def benchmark_ftpfn(spec, repeats, ftpfn_path):
    import jax
    from flax import nnx

    from icicle_jax.ftpfn import load_ftpfn

    p, c, d = spec["P"], spec["C"], spec["D"]
    model = load_ftpfn(ftpfn_path)
    rng = np.random.default_rng(0)
    # FT-PFN tokens repeat the full HP vector at each point:
    # [ curve_id, fidelity, hp_1, ..., hp_D ].
    hps = np.zeros((c, FTPFN_CONFIG_DIM), np.float32)
    hps[:, :d] = rng.uniform(size=(c, d)).astype(np.float32)
    idx = np.arange(p) % c
    x = np.zeros((1, p, 2 + FTPFN_CONFIG_DIM), np.float32)
    x[0, :, 0] = idx.astype(np.float32)
    x[0, :, 1] = rng.uniform(0.05, 1.0, size=p).astype(np.float32)
    x[0, :, 2:] = hps[idx]
    y = rng.uniform(size=(1, p)).astype(np.float32)
    sep = p // 2

    @nnx.jit
    def forward(model, x, y):
        return model(x, y, sep)

    x, y = jax.device_put((x, y))
    jax.block_until_ready((x, y))
    times = time_forward(forward, (model, x, y), repeats)
    return times, {"architecture": "ftpfn", "dtype": "float32", "config_dim": model.config_dim}


def run_cell(spec: dict, repeats: int, ftpfn_path: str | None, config_path: Path) -> dict:
    import jax

    times, model_config = (
        benchmark_ftpfn(spec, repeats, ftpfn_path)
        if spec["model"] == "ftpfn"
        else benchmark_icicle(spec, repeats, config_path)
    )
    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        **spec,
        "status": "ok",
        "t_median_s": float(np.median(times)),
        "t_iqr_lo_s": float(np.quantile(times, 0.25)),
        "t_iqr_hi_s": float(np.quantile(times, 0.75)),
        "peak_bytes": stats.get("peak_bytes_in_use"),
        "model_config": json.dumps(model_config, sort_keys=True),
        "dtype": model_config["dtype"],
        "device": jax.local_devices()[0].device_kind,
        "jax": jax.__version__,
        "platform": jax.default_backend(),
    }


def main() -> None:
    args = tyro.cli(Args)
    if args.repeats < 1 or not args.backends:
        raise SystemExit("Use positive repeats and at least one backend.")
    if args.cell is not None:
        row = run_cell(json.loads(args.cell), args.repeats, args.ftpfn, args.config)
        print(json.dumps(row), flush=True)
        return

    import pandas as pd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in benchmark_cases(args):
        row = {
            **spec,
            "repeats": args.repeats,
            "config": str(args.config.resolve()) if spec["model"] == "icicle" else None,
            "checkpoint": str(Path(args.ftpfn).resolve()) if spec["model"] == "ftpfn" else None,
            "status": "failed",
            "reason": None,
            "t_median_s": np.nan,
            "peak_bytes": None,
        }
        if spec["model"] == "ftpfn" and (
            spec["D"] > FTPFN_CONFIG_DIM or spec["C"] > FTPFN_MAX_CURVES
        ):
            row.update(status="cannot_run", reason="FT-PFN supports 10 HPs and 1000 curves")
        else:
            cmd = [
                sys.executable, "-m", "experiments.bench_scaling",
                "--cell", json.dumps(spec), "--repeats", str(args.repeats),
                "--config", str(args.config),
            ]
            if args.ftpfn is not None:
                cmd += ["--ftpfn", args.ftpfn]
            result = subprocess.run(cmd, capture_output=True, text=True)
            try:
                if result.returncode:
                    raise ValueError(f"Worker exited with code {result.returncode}")
                row.update(json.loads(result.stdout.strip().splitlines()[-1]))
            except (ValueError, IndexError) as error:
                lines = result.stderr.strip().splitlines()
                row["reason"] = lines[-1] if lines else str(error)

        rows.append(row)
        pd.DataFrame(rows).to_parquet(args.out)
        print(
            f"{row['model']}/{row['backend']} {row['axis']}: "
            f"P={row['P']} C={row['C']} D={row['D']} -> {row['status']} "
            f"{row['reason'] or ''}",
            flush=True,
        )
    print(f"Wrote {args.out} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
