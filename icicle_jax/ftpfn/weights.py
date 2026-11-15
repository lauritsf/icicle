"""Load the published FT-PFN checkpoint (its torch state_dict as numpy arrays)."""

import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import DTypeLike

from icicle_jax.ftpfn.model import FTPFN


def _take(sd: dict, key: str, dtype: DTypeLike) -> jax.Array:
    return jnp.asarray(sd.pop(key), dtype)


def _load_linear(linear: nnx.Linear, sd: dict, prefix: str, dtype: DTypeLike) -> None:
    # torch nn.Linear stores (out, in); flax wants (in, out).
    linear.kernel[...] = _take(sd, f"{prefix}.weight", dtype).T
    linear.bias[...] = _take(sd, f"{prefix}.bias", dtype)


def _load_attention(mha, sd: dict, prefix: str, embed: int, num_heads: int, dtype) -> None:
    # torch packs q/k/v into one (3 * embed, embed) weight; flax keeps per-head
    # (in, num_heads, head_dim) projections.
    head_dim = embed // num_heads
    in_w = _take(sd, f"{prefix}.in_proj_weight", dtype)
    in_b = _take(sd, f"{prefix}.in_proj_bias", dtype)
    for i, proj in enumerate((mha.query, mha.key, mha.value)):
        block = in_w[i * embed : (i + 1) * embed]
        proj.kernel[...] = block.reshape(num_heads, head_dim, embed).transpose(2, 0, 1)
        proj.bias[...] = in_b[i * embed : (i + 1) * embed].reshape(num_heads, head_dim)
    out_w = _take(sd, f"{prefix}.out_proj.weight", dtype)
    mha.out.kernel[...] = out_w.reshape(embed, num_heads, head_dim).transpose(1, 2, 0)
    mha.out.bias[...] = _take(sd, f"{prefix}.out_proj.bias", dtype)


def build_ftpfn(
    state_dict: Mapping[str, np.ndarray],
    *,
    num_heads: int = 4,
    param_dtype: DTypeLike = jnp.float32,
    attention_fn: Callable[..., jax.Array] | None = None,
) -> FTPFN:
    """Build the model and load a state_dict into it.

    Every dimension except the head count is read off the weights (the packed
    attention weight looks the same for any head split; 4 for the published
    checkpoint).
    """
    embed = state_dict["encoder.epoch_enc.weight"].shape[0]
    model = FTPFN(
        config_dim=state_dict["encoder.configuration_enc.base_encoder.weight"].shape[1],
        emsize=embed,
        num_heads=num_heads,
        hidden_dim=state_dict["transformer_encoder.layers.0.linear1.weight"].shape[0],
        num_layers=1
        + max(
            int(k.split(".")[2]) for k in state_dict if k.startswith("transformer_encoder.layers.")
        ),
        num_bars=state_dict["decoder_dict.standard.2.weight"].shape[0],
        max_curves=state_dict["encoder.idcurve_enc.weight"].shape[0] - 1,
        param_dtype=param_dtype,
        attention_fn=attention_fn,
        rngs=nnx.Rngs(0),
    )

    sd = dict(state_dict)
    model.epoch_enc.kernel[...] = _take(sd, "encoder.epoch_enc.weight", param_dtype).T
    model.idcurve_enc.embedding[...] = _take(sd, "encoder.idcurve_enc.weight", param_dtype)
    _load_linear(model.config_enc, sd, "encoder.configuration_enc.base_encoder", param_dtype)
    _load_linear(model.y_enc, sd, "y_encoder.1", param_dtype)
    for i, layer in enumerate(model.layers):
        prefix = f"transformer_encoder.layers.{i}"
        _load_attention(layer.attn, sd, f"{prefix}.self_attn", embed, num_heads, param_dtype)
        _load_linear(layer.linear1, sd, f"{prefix}.linear1", param_dtype)
        _load_linear(layer.linear2, sd, f"{prefix}.linear2", param_dtype)
        for norm, name in ((layer.norm1, "norm1"), (layer.norm2, "norm2")):
            norm.scale[...] = _take(sd, f"{prefix}.{name}.weight", param_dtype)
            norm.bias[...] = _take(sd, f"{prefix}.{name}.bias", param_dtype)
    _load_linear(model.decoder_linear1, sd, "decoder_dict.standard.0", param_dtype)
    _load_linear(model.decoder_linear2, sd, "decoder_dict.standard.2", param_dtype)

    leftover = {k for k in sd if not k.startswith("criterion.")}
    if leftover:
        raise ValueError(f"state_dict has unconverted weights: {sorted(leftover)}")
    return model


def load_ftpfn(path: str | Path, **kwargs) -> FTPFN:
    """Load the converted safetensors artifact (see ``convert``)."""
    from safetensors.numpy import load_file

    return build_ftpfn(load_file(str(path)), **kwargs)


def convert(npz_path: str, out_path: str) -> None:
    """One-time: the torch state_dict as an npz -> the safetensors artifact.

        python -m icicle_jax.ftpfn.weights STATE_DICT.npz OUT.safetensors

    The npz is one offline line from the downloaded checkpoint (the only step
    torch is ever needed for)::

        np.savez(out, **{k: v.numpy() for k, v in model.state_dict().items()})

    Validate the uniform 1000-bucket criterion and load all model weights before
    writing them. Rebuild the criterion when loading the exported model.
    """
    sd = dict(np.load(npz_path))
    np.testing.assert_allclose(sd["criterion.borders"], np.linspace(0.0, 1.0, 1001), atol=1e-6)
    assert float(sd.get("criterion.smoothing", 0.0)) == 0.0
    build_ftpfn(sd)  # raises on any missing or leftover weight

    from safetensors.numpy import save_file

    save_file({k: v for k, v in sd.items() if not k.startswith("criterion.")}, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
