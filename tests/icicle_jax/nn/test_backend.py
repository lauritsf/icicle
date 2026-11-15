"""Check XLA execution and attention-provider selection.

Auto selects flash when importable and otherwise XLA. An explicit flash request
raises if no provider is available.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from icicle_jax.nn.attention import resolve_backend
from icicle_jax.nn.attention_flash import flash_available
from icicle_jax.nn.model import Surrogate


def test_xla_always_resolves():
    assert resolve_backend("xla") == "xla"


def test_auto_degrades_to_available():
    assert resolve_backend("auto") == ("flash" if flash_available() else "xla")


def test_explicit_flash_fails_loudly_without_kernel():
    if flash_available():
        assert resolve_backend("flash") == "flash"
    else:
        with pytest.raises(RuntimeError):
            resolve_backend("flash")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        resolve_backend("bogus")


def test_xla_surrogate_builds_and_runs_anywhere():
    model = Surrogate(
        emsize=16,
        num_heads=4,
        hidden_dim=32,
        n_feat_layers=1,
        n_icl_layers=2,
        num_bars=8,
        icl_backend="xla",
        rngs=nnx.Rngs(0),
    )
    assert all(layer.attn.backend == "xla" for layer in model.icl_layers)
    b, c, h, p = 2, 4, 3, 6
    config_hps = jnp.zeros((b, c, h))
    point_config_idx = jnp.zeros((b, p), jnp.int32)
    fidelities = jnp.linspace(0.1, 1.0, p)[None].repeat(b, 0)
    y = jnp.zeros((b, p))
    logits = model(
        config_hps,
        jnp.array(2),
        point_config_idx,
        fidelities,
        y,
        single_eval_pos=3,
        feature_key=jax.random.key(1),
    )
    assert logits.shape == (b, p, 8)
    assert jnp.all(jnp.isfinite(logits))
