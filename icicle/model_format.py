"""Names and shapes of the float32 tensors in a model export."""

import numpy as np

FORMAT = "icicle-spec-v1"


def _encoder_layer(prefix: str, embed: int, head_dim: int, hidden: int, qk_norm: bool) -> dict:
    t = {f"{prefix}.norm1.scale": (embed,), f"{prefix}.norm2.scale": (embed,)}
    t.update(_attention(f"{prefix}.attn", embed, head_dim, qk_norm))
    t.update(_linear(f"{prefix}.w_in", embed, hidden))
    t.update(_linear(f"{prefix}.w_out", hidden, embed))
    return t


def _attention(prefix: str, embed: int, head_dim: int, qk_norm: bool) -> dict:
    t = {}
    if qk_norm:
        t[f"{prefix}.q_norm.scale"] = (head_dim,)
        t[f"{prefix}.k_norm.scale"] = (head_dim,)
    for name in ("q", "k", "v", "out"):
        t.update(_linear(f"{prefix}.{name}", embed, embed))
    return t


def _linear(prefix: str, n_in: int, n_out: int) -> dict:
    return {f"{prefix}.kernel": (n_in, n_out), f"{prefix}.bias": (n_out,)}


def expected_tensors(model_cfg: dict) -> dict[str, tuple]:
    """Every tensor an export of this architecture must contain, ``{name: shape}``."""
    if model_cfg.get("curve_identity", "none") != "none":
        raise ValueError("ICICLE exports with curve identities are not supported")
    emsize, hidden = model_cfg["emsize"], model_cfg["hidden_dim"]
    head_dim = emsize // model_cfg["num_heads"]
    qk_norm = model_cfg["qk_norm"]
    t = {}
    for i in range(model_cfg["n_feat_layers"]):
        t.update(
            _encoder_layer(f"config_encoder.feat_layers.{i}", emsize, head_dim, hidden, qk_norm)
        )
    t["config_encoder.pool.seed"] = (emsize,)
    t.update(_attention("config_encoder.pool.attn", emsize, head_dim, qk_norm))
    t["config_encoder.norm.scale"] = (emsize,)
    for i in range(model_cfg["n_icl_layers"]):
        t.update(_encoder_layer(f"icl_layers.{i}", emsize, head_dim, hidden, qk_norm))
    t["final_norm.scale"] = (emsize,)
    t.update(_linear("decoder_linear1", emsize, hidden))
    t.update(_linear("decoder_linear2", hidden, model_cfg["num_bars"]))
    return t


def make_manifest(model_cfg: dict, step: int) -> dict:
    return {"format": FORMAT, "model": dict(model_cfg), "step": int(step)}


def validate(manifest: dict, tensors: dict[str, np.ndarray]) -> None:
    """Raise ValueError unless ``tensors`` is exactly the set the manifest's model demands."""
    if manifest.get("format") != FORMAT:
        raise ValueError(f"format {manifest.get('format')!r} != {FORMAT!r}")
    expected = expected_tensors(manifest["model"])
    missing = expected.keys() - tensors.keys()
    extra = tensors.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"tensor set mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    for name, shape in expected.items():
        a = tensors[name]
        if tuple(a.shape) != tuple(shape):
            raise ValueError(f"{name}: shape {tuple(a.shape)} != expected {tuple(shape)}")
        if a.dtype != np.float32:
            raise ValueError(f"{name}: dtype {a.dtype} != float32")
