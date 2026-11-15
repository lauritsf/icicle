"""Flash attention for a context prefix shared across the batch.

Keys and values are packed into fixed-size buffers. Sequence offsets let the
kernel handle different context lengths without recompiling.

ICICLE_FLASH_PROVIDER selects a module with is_available() and
flash_attention_varlen_traced(). Flash requires an explicitly configured provider.
Float32 and float64 inputs are converted to bf16; use XLA for precision checks.
"""

import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np

def _flash_provider():
    """Import the configured flash-kernel provider module (``ICICLE_FLASH_PROVIDER``)."""
    name = os.environ.get("ICICLE_FLASH_PROVIDER")
    if not name:
        raise RuntimeError("set ICICLE_FLASH_PROVIDER to use flash attention")
    return importlib.import_module(name)


def flash_available() -> bool:
    """Return whether the GPU provider and its shared library are available.

    The provider must be on the Python import path. Missing providers and
    CPU-only execution cause auto selection to use XLA.
    """
    if not os.environ.get("ICICLE_FLASH_PROVIDER"):
        return False
    try:
        # The kernel's FFI is registered for GPU execution. Use XLA in CPU processes.
        if jax.default_backend() == "cpu":
            return False
        return bool(_flash_provider().is_available())
    except Exception:  # noqa: BLE001 - provider / .so absent off a flash-capable env
        return False


def _prefix_len(key_mask: jax.Array) -> jax.Array:
    """Recover ``sep`` from a shared prefix key mask ``(..., Sk)`` (True = attendable key)."""
    key_valid = jnp.any(key_mask.reshape(-1, key_mask.shape[-1]), axis=0)  # (Sk,)
    return jnp.count_nonzero(key_valid)


def flash_incontext_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, key_mask: jax.Array
) -> jax.Array:
    """Apply fused attention to q, k and v of shape (B, S, H, D).

    key_mask specifies a prefix shared across the batch. Its length may be a
    traced scalar, allowing one compiled function to handle different splits.
    Returns (B, S, H, D).
    """
    b, s, h, d = q.shape
    scale = 1.0 / np.sqrt(d)
    orig = q.dtype
    if orig not in (jnp.bfloat16, jnp.float16):  # CK kernel needs bf16/fp16
        q, k, v = (a.astype(jnp.bfloat16) for a in (q, k, v))
    active = jnp.int32(_prefix_len(key_mask))
    sep = jnp.maximum(active, 1)  # Empty contexts use one zero key per sequence.

    # Each query sequence contains all S tokens.
    q_flat = q.reshape(b * s, h, d)
    q_seq_offsets = jnp.asarray(np.arange(b + 1) * s, jnp.int32)

    # Pack the context prefixes together. Send excluded tokens to the last slot.
    e = jnp.arange(s)[None, :]  # (1, S)
    row = jnp.arange(b)[:, None]  # (B, 1)
    dump = b * s
    k_idx = jnp.where(e < sep, row * sep + e, dump).reshape(-1)  # (B*S,)
    kbuf = jnp.zeros((b * s + 1, h, d), q.dtype).at[k_idx].set(k.reshape(b * s, h, d))[:dump]
    vbuf = jnp.zeros((b * s + 1, h, d), q.dtype).at[k_idx].set(v.reshape(b * s, h, d))[:dump]
    k_seq_offsets = (jnp.arange(b + 1) * sep).astype(jnp.int32)

    # Zero padding with where so NaNs in excluded tokens cannot enter the gradients.
    valid = (jnp.arange(b * s) < k_seq_offsets[-1])[:, None, None]
    kbuf = jnp.where(valid & (active > 0), kbuf, 0)
    vbuf = jnp.where(valid & (active > 0), vbuf, 0)

    out = _flash_provider().flash_attention_varlen_traced(
        q_flat,
        kbuf,
        vbuf,
        q_seq_offsets,
        k_seq_offsets,
        max_seqlen_q=s,
        max_seqlen_k=s,
        causal=False,
        scale=scale,
    )
    return jnp.where(active > 0, out.reshape(b, s, h, d).astype(orig), 0)
