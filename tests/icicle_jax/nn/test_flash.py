"""Compare flash forward values and gradients with XLA attention.

Check agreement within bf16 tolerance and finite, bounded gradient norms.
Tests skip when no flash provider is available. Inputs are float32; the fused
kernel casts attention operands to bf16.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from icicle_jax.nn.attention import dot_product_attention, incontext_key_mask
from icicle_jax.nn.attention_flash import flash_available, flash_incontext_attention
from icicle_jax.nn.model import Surrogate
from icicle_jax.prior.bnn import build_output_sorted
from icicle_jax.prior.task import sample_task

pytestmark = pytest.mark.skipif(not flash_available(), reason="CK flash kernel unavailable")

B, S, HEADS, D, SEP = 2, 32, 2, 64, 12  # D=64 is a CK-supported head dim


def _qkv(key):
    return jax.random.normal(key, (3, B, S, HEADS, D), jnp.float32)


def test_flash_forward_matches_default():
    q, k, v = _qkv(jax.random.key(0))
    mask = incontext_key_mask(SEP, S)
    o_def = dot_product_attention(q, k, v, mask)
    o_flash = flash_incontext_attention(q, k, v, mask)
    np.testing.assert_allclose(np.asarray(o_flash), np.asarray(o_def), atol=3e-2, rtol=3e-2)


def test_flash_grad_matches_and_is_bounded():
    q, k, v = _qkv(jax.random.key(1))
    w = jax.random.normal(jax.random.key(2), (B, S, HEADS, D), jnp.float32)
    mask = incontext_key_mask(SEP, S)

    def loss(fn):
        return lambda a, b, c: jnp.sum(fn(a, b, c, mask) * w)

    g_def = jax.grad(loss(dot_product_attention), argnums=(0, 1, 2))(q, k, v)
    g_flash = jax.grad(loss(flash_incontext_attention), argnums=(0, 1, 2))(q, k, v)
    for gd, gf in zip(g_def, g_flash, strict=True):
        assert jnp.all(jnp.isfinite(gf))
        nd, nf = float(jnp.linalg.norm(gd)), float(jnp.linalg.norm(gf))
        assert 0.5 < nf / nd < 2.0
        np.testing.assert_allclose(np.asarray(gf), np.asarray(gd), atol=5e-2, rtol=5e-2)


def _small_batch():
    cfg_dim = 4
    table = build_output_sorted(
        jax.random.key(0),
        config_dim=cfg_dim,
        layers=(2, 4),
        hidden=(8, 16),
        num_networks=64,
        configs_per_network=16,
    )
    sample = partial(
        sample_task,
        seq_len=S,
        config_dim=cfg_dim,
        num_params=jnp.array(3),
        n_steps=jnp.array(6),
        single_eval_pos=jnp.array(SEP),
        table=table,
        layers=(2, 4),
        hidden=(8, 16),
        n_steps_max=8,
    )
    batch = jax.vmap(sample)(jax.random.split(jax.random.key(1), B))
    return cfg_dim, batch


def _surrogate(cfg_dim, backend):
    # emsize/num_heads = 64 = a CK-supported head dim; same seed -> identical weights across
    # backends, so the only difference is the ICL attention kernel.
    return Surrogate(
        emsize=64,
        num_heads=1,
        hidden_dim=64,
        n_feat_layers=1,
        n_icl_layers=2,
        num_bars=8,
        icl_backend=backend,
        rngs=nnx.Rngs(0),
    )


def test_surrogate_flash_backend_routes_and_trains():
    cfg_dim, batch = _small_batch()
    m_flash, m_def = _surrogate(cfg_dim, "flash"), _surrogate(cfg_dim, "xla")
    assert all(layer.attn.backend == "flash" for layer in m_flash.icl_layers)
    assert all(layer.attn.backend == "xla" for layer in m_def.icl_layers)

    fk = jax.random.key(7)
    args = (batch.config_hps, jnp.array(3), batch.point_config_idx, batch.fidelities, batch.y, SEP)
    out_flash = m_flash(*args, feature_key=fk)
    out_def = m_def(*args, feature_key=fk)
    assert jnp.all(jnp.isfinite(out_flash))
    np.testing.assert_allclose(np.asarray(out_flash), np.asarray(out_def), atol=1e-1, rtol=1e-1)

    def loss(m):
        return jnp.sum(m(*args, feature_key=fk) ** 2)

    g_flash = nnx.grad(loss)(m_flash)
    norms = [float(jnp.linalg.norm(g)) for g in jax.tree.leaves(g_flash)]
    assert all(np.isfinite(n) for n in norms)
    assert max(norms) < 1e4  # a blown-up bf16 backward produces ~1e7 norms; guard that regime


def test_config_encoder_flash_matches_xla():
    """Stage-1 feature attention under flash == xla (fwd + grad), at a cfg40-like shape.

    The validity mask is a shared prefix (num_params active cells), which is what qualifies
    the feature self-attention for the fused prefix kernel; the PMA pool stays xla inside.
    """
    from icicle_jax.nn.config_encoder import ConfigEncoder

    b, c, h, e, num_params = 2, 7, 40, 64, 23
    rng = jax.random.PRNGKey(3)
    cells = jax.random.normal(rng, (b, c, h, e), jnp.float32)
    mask = jnp.broadcast_to(jnp.arange(h) < num_params, (b, c, h))

    encoders = {}
    for backend in ("xla", "flash"):
        enc = ConfigEncoder(e, 2, 128, 1, qk_norm=True, backend=backend, rngs=nnx.Rngs(0))
        encoders[backend] = enc
    # same rngs seed -> identical params; assert to be safe
    px, pf = (nnx.state(encoders[k], nnx.Param) for k in ("xla", "flash"))
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b), px, pf)

    def loss(enc):
        return jnp.sum(enc(cells, mask) ** 2)

    o_x, o_f = encoders["xla"](cells, mask), encoders["flash"](cells, mask)
    np.testing.assert_allclose(np.asarray(o_f), np.asarray(o_x), atol=5e-2, rtol=5e-2)
    g_x = nnx.grad(loss)(encoders["xla"])
    g_f = nnx.grad(loss)(encoders["flash"])
    for a, b_ in zip(jax.tree.leaves(g_x), jax.tree.leaves(g_f), strict=True):
        np.testing.assert_allclose(np.asarray(b_), np.asarray(a), atol=8e-2, rtol=8e-2)
