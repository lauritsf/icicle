"""Tests for the shared masked attention primitive."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from icicle_jax.nn.attention import Attention, incontext_key_mask

jax.config.update("jax_enable_x64", True)


def _attn(**kw):
    return Attention(16, num_heads=4, param_dtype=jnp.float64, rngs=nnx.Rngs(0), **kw)


def test_no_query_to_query_leakage():
    rng = np.random.default_rng(0)
    seq, sep = 12, 7
    attn = _attn()
    km = incontext_key_mask(sep, seq)
    src = jnp.asarray(rng.normal(size=(2, seq, 16)))
    out = attn(src, key_mask=km)

    j = sep + 2  # a query position
    src2 = src.at[:, j].set(jnp.asarray(rng.normal(size=(2, 16))))
    out2 = attn(src2, key_mask=km)

    keep = jnp.arange(seq) != j
    np.testing.assert_array_equal(np.asarray(out2[:, keep]), np.asarray(out[:, keep]))
    assert not np.allclose(np.asarray(out2[:, j]), np.asarray(out[:, j]))


def test_context_permutation_invariance():
    rng = np.random.default_rng(1)
    seq, sep = 12, 7
    attn = _attn()
    km = incontext_key_mask(sep, seq)
    src = jnp.asarray(rng.normal(size=(2, seq, 16)))
    out = attn(src, key_mask=km)

    perm = rng.permutation(sep)
    src_perm = src.at[:, :sep].set(src[:, :sep][:, perm])
    out_perm = attn(src_perm, key_mask=km)

    np.testing.assert_allclose(
        np.asarray(out_perm[:, sep:]), np.asarray(out[:, sep:]), rtol=1e-5, atol=1e-6
    )


def test_cross_attention_shape():
    rng = np.random.default_rng(3)
    attn = _attn()
    q = jnp.asarray(rng.normal(size=(2, 1, 16)))  # one query per set
    kv = jnp.asarray(rng.normal(size=(2, 5, 16)))  # 5 cells
    km = jnp.ones((2, 1, 1, 5), dtype=bool)
    assert attn(q, kv, key_mask=km).shape == (2, 1, 16)


def test_qk_norm_bounds_logits():
    """RMSNorm qk_norm makes the logits independent of input magnitude and bounded.

    RMSNorm(c*x) == RMSNorm(x), so the logit magnitude is invariant to input scale (no learned
    temperature that could drift) -- an unbounded input can't reach the ~1e7 regime that breaks
    a bf16 flash backward.
    """
    rng = np.random.default_rng(0)
    attn = _attn(qk_norm=True)
    src = jnp.asarray(rng.normal(size=(2, 12, 16)))

    def logit_max(scaled):
        q, k = attn._split(attn.q(scaled)), attn._split(attn.k(scaled))
        q, k = attn.q_norm(q), attn.k_norm(k)
        lg = jnp.max(jnp.abs(jnp.einsum("bqhd,bkhd->bhqk", q, k)))
        return float(lg) / np.sqrt(attn.head_dim)

    ls = [logit_max(c * src) for c in (1.0, 1e3, 1e6)]
    assert max(ls) < 20  # bounded ~sqrt(head_dim), never the extreme regime
    np.testing.assert_allclose(ls[0], ls[2], rtol=1e-4)  # scale-invariant: logit(1) == logit(1e6)
    plain = _attn()

    def plain_logit_max(scaled):
        q, k = plain._split(plain.q(scaled)), plain._split(plain.k(scaled))
        lg = jnp.max(jnp.abs(jnp.einsum("bqhd,bkhd->bhqk", q, k)))
        return float(lg) / np.sqrt(plain.head_dim)

    assert plain_logit_max(1e3 * src) > 1e4


def test_flash_requires_explicit_provider(monkeypatch):
    import pytest
    from icicle_jax.nn.attention import resolve_backend

    monkeypatch.delenv("ICICLE_FLASH_PROVIDER", raising=False)
    assert resolve_backend("auto") == "xla"
    assert resolve_backend("xla") == "xla"
    with pytest.raises(RuntimeError, match="ICICLE_FLASH_PROVIDER"):
        resolve_backend("flash")


def test_configured_flash_provider(monkeypatch):
    import sys
    from types import SimpleNamespace
    from icicle_jax.nn.attention import resolve_backend
    from icicle_jax.nn import attention_flash

    provider = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "test_flash_provider", provider)
    monkeypatch.setenv("ICICLE_FLASH_PROVIDER", "test_flash_provider")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert attention_flash._flash_provider() is provider
    assert resolve_backend("auto") == "flash"
    assert resolve_backend("flash") == "flash"
    monkeypatch.setattr(provider, "is_available", lambda: False)
    assert resolve_backend("auto") == "xla"


def test_flash_full_partial_and_empty_prefixes():
    import pytest
    from icicle_jax.nn.attention import dot_product_attention
    from icicle_jax.nn.attention_flash import flash_available, flash_incontext_attention

    if not flash_available():
        pytest.skip("requires the configured GPU flash provider")
    for length in (1, 5):
        q, k, v = jax.random.normal(jax.random.key(0), (3, 2, length, 2, 16), dtype=jnp.float32)
        q, k, v = (x.astype(jnp.bfloat16).astype(jnp.float32) for x in (q, k, v))
        for active in sorted({0, 1, length - 1, length}):
            mask = incontext_key_mask(active, length)

            def forward(q, k, v, flash):
                # Excluded values must not contaminate outputs or gradients.
                valid = jnp.arange(length)[None, :, None, None] < active
                k = jnp.where(valid, k, jnp.nan)
                v = jnp.where(valid, v, jnp.nan)
                if flash:
                    return flash_incontext_attention(q, k, v, mask)
                k, v = jnp.where(valid, k, 0), jnp.where(valid, v, 0)
                return jnp.where(active > 0, dot_product_attention(q, k, v, mask), 0)

            got = jax.jit(lambda q, k, v: forward(q, k, v, True))(q, k, v)
            want = forward(q, k, v, False)
            np.testing.assert_allclose(got, want, atol=1e-2, rtol=1e-2)
            for flash in (False, True):
                grad = jax.jit(jax.grad(
                    lambda q, k, v: forward(q, k, v, flash).astype(jnp.float32).sum(),
                    argnums=(0, 1, 2),
                ))(q, k, v)
                assert all(np.isfinite(x).all() for x in grad)
                if flash:
                    for actual, expected in zip(grad, reference, strict=True):
                        np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)
                else:
                    reference = grad
