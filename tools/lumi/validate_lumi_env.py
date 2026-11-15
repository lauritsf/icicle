"""Validate ICICLE's writable layer over the pinned lean LUMI image."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import flax
import jax
import jaxlib
import onnxruntime
import optax
import ray

ALLOWED_OGB_CONFLICTS = {
    "ogb 1.3.6 requires joblib, which is not installed.",
    "ogb 1.3.6 requires scikit-learn, which is not installed.",
    "ogb 1.3.6 requires torch, which is not installed.",
    "ogb 1.3.6 requires tqdm, which is not installed.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-note", action="store_true")
    args = parser.parse_args()
    venv = Path(os.environ["VENV_DIR"]).resolve()
    if Path(sys.prefix).resolve() != venv:
        raise RuntimeError(f"writable venv is not active: sys.prefix={sys.prefix}, expected={venv}")
    pins = tuple(
        version(package) for package in ("jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt")
    )
    expected_jax = (venv / "lumi-jax-line").read_text().strip()
    if expected_jax not in {"0.11.0", "0.11.2"}:
        raise RuntimeError(f"unsupported recorded JAX line: {expected_jax!r}")
    if pins != (expected_jax,) * 4:
        raise RuntimeError(f"JAX pin mismatch: {pins}")
    expected_flax = {"0.11.0": "0.12.8", "0.11.2": "0.12.10"}[expected_jax]
    if flax.__version__ != expected_flax:
        raise RuntimeError(
            f"Flax/JAX compatibility pin mismatch: flax={flax.__version__}, "
            f"expected={expected_flax} for jax={expected_jax}"
        )
    if (jax.__version__, jaxlib.__version__) != pins[:2]:
        raise RuntimeError("imported JAX versions differ from installed package metadata")
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX discovered no devices")
    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("lean environment unexpectedly exposes torch")

    result = subprocess.run(
        [os.sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    lines = {line for line in result.stdout.splitlines() if line}
    unexpected = lines - ALLOWED_OGB_CONFLICTS - {"No broken requirements found."}
    if unexpected:
        raise RuntimeError(f"unexpected dependency conflicts: {sorted(unexpected)}")
    if result.returncode and lines != ALLOWED_OGB_CONFLICTS:
        raise RuntimeError(f"unexpected pip check output: {sorted(lines)}")

    if args.write_note:
        (venv / "pip-check.txt").write_text(result.stdout)
        (venv / "pip-check-note.txt").write_text(
            "Accepted image-only orphan: ogb metadata is inherited from /opt/venv but ICICLE "
            "does not import or depend on ogb; installing torch would violate the lean "
            "contract.\n"
        )
    print(
        "validated",
        f"jax={pins[0]}",
        f"flax={flax.__version__}",
        f"optax={optax.__version__}",
        f"ray={ray.__version__}",
        f"onnxruntime={onnxruntime.__version__}",
        f"devices={len(devices)}x{devices[0].platform}",
    )


if __name__ == "__main__":
    main()
