"""Export checkpoint parameters as safetensors with a JSON model manifest.

The export contains float32 model tensors with flat module-path names. It can
be read without Orbax or JAX. The optional ONNX graph supports inference through
ONNX Runtime.

Usage: python -m icicle_jax.export RUN_DIR [--step N] [--out MODEL.safetensors] [--onnx]
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
from flax import nnx

from icicle.model_format import make_manifest, validate
from icicle_jax.loading import load_eval_model


def model_tensors(model) -> dict[str, np.ndarray]:
    """Flatten a Surrogate's learned parameters to ``{dot.joined.path: f32 array}``."""
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    return {
        ".".join(str(part) for part in path): np.asarray(v[...], dtype=np.float32)
        for path, v in flat
    }


def export_model(model, model_cfg: dict, step: int, out: str | Path) -> Path:
    """Write named safetensors weights and matching JSON metadata."""
    from safetensors.numpy import save_file

    if model_cfg.get("architecture", "icicle") != "icicle":
        raise ValueError("the portable ICICLE format does not represent FT-PFN research models")
    out = Path(out)
    if out.suffix != ".safetensors":
        raise ValueError("weights output must end in .safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    tensors = model_tensors(model)
    manifest = make_manifest(model_cfg, step)
    validate(manifest, tensors)  # Check tensor names and shapes against the specification.
    save_file(tensors, out, metadata={"format": manifest["format"]})
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
    return out


def export_run(run_dir: str | Path, *, step: int | None = None, out: str | Path | None = None):
    """Export weights and a path-independent training config for reproducible evaluation."""
    run_dir = Path(run_dir)
    model, _criterion, cfg, restored = load_eval_model(run_dir, step=step)
    source_dir = run_dir.parent if run_dir.is_file() else run_dir
    out = Path(out) if out else source_dir / "export" / f"step_{restored}" / "icicle.safetensors"
    if out.resolve() == run_dir.resolve():
        raise ValueError("export to a separate file to preserve the source")
    export_model(model, cfg["model"], restored, out)
    if "run" in cfg:
        cfg["run"]["checkpoint_dir"] = None
        out.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out


if __name__ == "__main__":
    import tyro

    @dataclasses.dataclass(frozen=True)
    class Args:
        run_dir: tyro.conf.Positional[str]
        step: int | None = None  # default: latest checkpoint
        out: str | None = None  # default: <run_dir>/export/step_<N>/icicle.safetensors
        onnx: bool = False  # also build the matching ONNX graph

    args = tyro.cli(Args)
    weights_path = export_run(args.run_dir, step=args.step, out=args.out)
    print(weights_path)
    if args.onnx:
        from icicle.onnx_export import export_onnx

        print(export_onnx(weights_path))
