"""Fourier features for observed values and fidelity coordinates.

The highest frequency completes one cycle over [0, 1]; lower frequencies are
spaced geometrically with base 10000.
"""

import jax
import jax.numpy as jnp


def sincos(value: jax.Array, dim: int) -> jax.Array:
    """Additive Fourier embedding of a continuous value in [0, 1].

    ``value`` is any shape ``(...)``, ``dim`` the output width (must be even); returns
    ``(..., dim)``. Angles are computed at least in float32. Supply float32 values to preserve
    fine differences: promotion cannot recover precision already lost to bf16.
    """
    half = dim // 2
    compute_dtype = jnp.result_type(value.dtype, jnp.float32)
    i = jnp.arange(half, dtype=compute_dtype)
    inv_freq = 10000.0 ** (-2.0 * i / dim)
    angles = (value.astype(compute_dtype) * 2.0 * jnp.pi)[..., None] * inv_freq
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
