"""Masked multi-head attention shared by configuration and point encoders.

The caller supplies a key-only mask with shape (B, 1, 1, Sk). Queries and keys
can use per-head RMSNorm before scaled dot-product attention. Cross-attention uses
the same projections, normalization and masking interface.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import DTypeLike

from icicle_jax.nn.attention_flash import flash_available, flash_incontext_attention


def dot_product_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, key_mask: jax.Array, *, bias=None
) -> jax.Array:
    """Compute masked attention for head-split q, k and v arrays of shape (B, S, H, D).

    key_mask broadcasts to (B, 1, 1, Sk); True permits attention. Scores are
    scaled by 1/sqrt(head_dim). JAX computes softmax in float32 even when the
    surrounding arrays use float64. Reduction order can change with batch
    shape, so parity checks should allow float32 rounding differences.
    """
    # Call JAX's 4-D attention core directly. The wrapper
    # reshapes Q to 5D for GQA even at G=1, which on the ROCm MI250X backend triggers a
    # ~33 GB compile-time allocation. The 4D core peaks ~1 GB for the same program.
    from jax._src.nn.functions import _dot_product_attention_core

    return _dot_product_attention_core(
        q,
        k,
        v,
        bias=bias,
        mask=key_mask,
        is_causal=False,
        scale=1.0 / np.sqrt(q.shape[-1]),
        q_seqlen=None,
        kv_seqlen=None,
        local_window_size=None,
        return_residual=False,
    )


def resolve_backend(backend: str) -> str:
    """Resolve xla, flash or auto to an available attention implementation.

    xla uses the masked JAX implementation. flash requires a compatible kernel
    and raises if it is unavailable. auto selects flash when available and xla
    otherwise.
    """
    if backend == "xla":
        return "xla"
    if backend == "auto":
        return "flash" if flash_available() else "xla"
    if backend == "flash":
        if not flash_available():
            raise RuntimeError(
                "flash attention requires an available provider configured through "
                "ICICLE_FLASH_PROVIDER; use 'xla' or 'auto' for XLA fallback"
            )
        return "flash"
    raise ValueError(f"backend {backend!r} must be 'auto', 'xla', or 'flash'")


def incontext_key_mask(single_eval_pos: int | jax.Array, seq: int) -> jax.Array:
    """Mask of shape (1, 1, 1, seq) allowing attention to context keys.

    single_eval_pos may vary between calls without recompiling.
    """
    return (jnp.arange(seq) < single_eval_pos)[None, None, None, :]


class Attention(nnx.Module):
    """Multi-head attention with optional per-head RMSNorm.

    Pass x_kv for cross-attention; otherwise x_q supplies queries, keys and values.
    The flash backend requires a context prefix shared across the batch.
    Callers use XLA for PMA pooling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qk_norm: bool = False,
        backend: str = "xla",
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qk_norm = qk_norm
        self.backend = resolve_backend(backend)
        kw = {"dtype": dtype, "param_dtype": param_dtype, "rngs": rngs}
        self.q_norm = nnx.RMSNorm(self.head_dim, **kw) if qk_norm else None
        self.k_norm = nnx.RMSNorm(self.head_dim, **kw) if qk_norm else None
        self.q = nnx.Linear(embed_dim, embed_dim, **kw)
        self.k = nnx.Linear(embed_dim, embed_dim, **kw)
        self.v = nnx.Linear(embed_dim, embed_dim, **kw)
        self.out = nnx.Linear(embed_dim, embed_dim, **kw)

    def _split(self, x: jax.Array) -> jax.Array:
        return x.reshape(*x.shape[:2], self.num_heads, self.head_dim)

    def __call__(
        self,
        x_q: jax.Array,
        x_kv: jax.Array | None = None,
        *,
        key_mask: jax.Array,
    ) -> jax.Array:
        x_kv = x_q if x_kv is None else x_kv
        q = self._split(self.q(x_q))
        k = self._split(self.k(x_kv))
        v = self._split(self.v(x_kv))
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)  # RMSNorm over head_dim
        attend = flash_incontext_attention if self.backend == "flash" else dot_product_attention
        o = attend(q, k, v, key_mask)
        y = self.out(o.reshape(*o.shape[:2], self.num_heads * self.head_dim))
        # Return zero when every key is masked, discarding any NaNs from the kernel.
        return jnp.where(key_mask.any(-1), y, 0.0)
