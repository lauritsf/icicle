"""Load trained ICICLE parameters for inference or export."""

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from icicle.model_format import validate
from icicle_jax.bar_distribution import BarDistribution
from icicle_jax.nn.model import Surrogate


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    architecture: str = "icicle"  # "icicle" or "ftpfn"
    config_dim: int = 10
    emsize: int = 256
    num_heads: int = 8
    hidden_dim: int = 512
    n_feat_layers: int = 1
    n_icl_layers: int = 6
    num_bars: int = 1000
    icl_backend: str = "auto"  # flash when configured and available, otherwise XLA
    ragged: bool = True  # pack referenced configurations before encoding
    qk_norm: bool = True  # normalize queries and keys before attention
    ftpfn_max_curves: int = 1000
    dtype: str = "bfloat16"  # compute dtype ("float32" for parity/debug); loss stays f32
    # Recompute layer activations during backpropagation to save memory.
    remat: bool = True
    # None uses the same setting as remat.
    config_remat: bool | None = None
    feat_chunk: int = 0  # configuration rows per chunk; 0 processes all rows at once
    # Randomly assign hyperparameters to FT-PFN input slots during training.
    ftpfn_random_slots: bool = False
    ftpfn_initialization: str = "flax"  # "flax" or "pytorch"


def strip_retired_model_keys(model_cfg: dict) -> dict:
    """Read saved configuration fields and reject unsupported architectures."""
    d = dict(model_cfg)
    for name, supported in (("value_map", "raw_mul_tag"), ("config_aggregator", "dense_reference"),
                            ("feature_identity_kind", "haar")):
        if d.pop(name, supported) != supported:
            raise ValueError(f"unsupported configuration encoder: {name}")
    fidelity_code = d.pop("fidelity_code", "split")
    final_norm = d.pop("final_norm", True)
    if fidelity_code != "split" or final_norm is not True:
        raise ValueError(
            f"unsupported configuration (fidelity_code={fidelity_code!r}, "
            f"final_norm={final_norm!r})"
        )
    kind = d.pop("curve_identity", "none")
    scale = d.pop("curve_identity_scale", 1.0)
    vocab = d.pop("curve_vocab", None)
    if d.get("architecture", "icicle") == "ftpfn":
        if kind not in ("none", "learned") or scale != 1.0:
            raise ValueError("unsupported FT-PFN curve embedding settings")
        if vocab is not None:
            if "ftpfn_max_curves" in d and d["ftpfn_max_curves"] != vocab:
                raise ValueError("conflicting FT-PFN curve capacities")
            d["ftpfn_max_curves"] = vocab
    elif kind != "none":
        raise ValueError("ICICLE checkpoints with curve identities are not supported")
    return d


def build_model(model_cfg, rngs):
    """Build ICICLE or FT-PFN from its model configuration."""
    if model_cfg.dtype not in ("float32", "bfloat16"):
        raise ValueError(f"dtype {model_cfg.dtype!r} must be 'float32' or 'bfloat16'")
    if model_cfg.architecture == "ftpfn":
        from icicle_jax.ftpfn.train import TrainableFTPFN

        if not (
            model_cfg.config_dim >= 1
            and model_cfg.n_feat_layers == 0
            and not model_cfg.ragged
            and not model_cfg.qk_norm
            and model_cfg.icl_backend == "xla"
            and model_cfg.feat_chunk == 0
            and model_cfg.config_remat is None
        ):
            raise ValueError(
                "FT-PFN requires XLA, post-norm attention and no configuration-encoder layers"
            )
        return TrainableFTPFN(
            emsize=model_cfg.emsize,
            num_heads=model_cfg.num_heads,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.n_icl_layers,
            num_bars=model_cfg.num_bars,
            config_dim=model_cfg.config_dim,
            max_curves=model_cfg.ftpfn_max_curves,
            dtype=None if model_cfg.dtype == "float32" else jnp.bfloat16,
            remat=model_cfg.remat,
            random_slots=model_cfg.ftpfn_random_slots,
            initialization=model_cfg.ftpfn_initialization,
            rngs=rngs,
        )
    if model_cfg.architecture != "icicle":
        raise ValueError(f"unknown architecture: {model_cfg.architecture!r}")
    if model_cfg.ftpfn_random_slots or model_cfg.ftpfn_initialization != "flax":
        raise ValueError("FT-PFN initialization and slot options require architecture=ftpfn")
    return Surrogate(
        dtype=None if model_cfg.dtype == "float32" else jnp.bfloat16,
        emsize=model_cfg.emsize,
        num_heads=model_cfg.num_heads,
        hidden_dim=model_cfg.hidden_dim,
        n_feat_layers=model_cfg.n_feat_layers,
        n_icl_layers=model_cfg.n_icl_layers,
        num_bars=model_cfg.num_bars,
        icl_backend=model_cfg.icl_backend,
        qk_norm=model_cfg.qk_norm,
        remat=model_cfg.remat,
        config_remat=model_cfg.config_remat,
        feat_chunk=model_cfg.feat_chunk,
        param_dtype=jnp.float32,
        rngs=rngs,
    )


def load_eval_model(run_dir, *, step=None, config_dim=None, backend="xla", dtype=None):
    """Load a checkpoint directory or exported .safetensors model.

    Return (model, criterion, cfg, restored_step). Restore model parameters only.
    Portable exports also check model metadata; an explicit step must match the export.

    Defaults use float32 and XLA attention. Changing dtype or backend can change
    numerical results, so do not merge evaluations produced with different settings.
    config_dim may widen ICICLE's input layout without changing parameter shapes.
    """
    run_dir = Path(run_dir)
    weights_path = run_dir if run_dir.suffix == ".safetensors" else None
    config_path = weights_path.with_suffix(".config.json") if weights_path else run_dir / "config.json"
    if weights_path:
        manifest = json.loads(weights_path.with_suffix(".json").read_text())
        cfg = (json.loads(config_path.read_text()) if config_path.exists()
               else {"model": manifest["model"]})
    else:
        cfg = json.loads(config_path.read_text())
    cfg["model"] = strip_retired_model_keys(cfg["model"])
    model_cfg = dataclasses.replace(
        ModelConfig(**cfg["model"]), dtype=dtype or "float32", icl_backend=backend
    )
    if config_dim is not None:
        model_cfg = dataclasses.replace(model_cfg, config_dim=config_dim)
    # Random slot assignment is a training augmentation. Evaluation uses the coordinates as
    # supplied unless a diagnostic explicitly constructs another presentation.
    model_cfg = dataclasses.replace(model_cfg, ftpfn_random_slots=False)
    model = build_model(model_cfg, nnx.Rngs(0))
    if weights_path is not None:
        from safetensors.numpy import load_file

        if ModelConfig(**strip_retired_model_keys(manifest["model"])) != ModelConfig(**cfg["model"]):
            raise ValueError("portable manifest and training config describe different models")
        restored = manifest["step"]
        if step is not None and step != restored:
            raise ValueError(f"export contains step {restored}, not requested step {step}")
        tensors = load_file(weights_path)
        validate(manifest, tensors)
        state = nnx.state(model, nnx.Param)
        for path, variable in nnx.to_flat_state(state):
            name = ".".join(str(part) for part in path)
            if not np.isfinite(tensors[name]).all():
                raise ValueError(f"non-finite portable parameter: {name}")
            variable[...] = jnp.asarray(tensors[name])
        nnx.update(model, state)
        criterion = BarDistribution(borders=jnp.linspace(0.0, 1.0, model_cfg.num_bars + 1))
        return model, criterion, cfg, restored
    # config.json records the checkpoint dir as an absolute path; resolve its basename under
    # run_dir so a moved/copied run directory still evaluates.
    ckpt_dir = run_dir / Path(cfg["run"]["checkpoint_dir"]).name
    from icicle_jax import checkpoint

    restored = checkpoint.restore_model(str(ckpt_dir), model, step=step)
    if restored is None:
        raise FileNotFoundError(f"no checkpoint under {ckpt_dir}")
    criterion = BarDistribution(borders=jnp.linspace(0.0, 1.0, model_cfg.num_bars + 1))
    return model, criterion, cfg, restored
