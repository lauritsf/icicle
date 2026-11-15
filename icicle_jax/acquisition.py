"""JAX acquisition reductions matching icicle.hpo.decider.

Reduce bucket probabilities on device so the host receives one score per candidate.
Thresholds broadcast over the leading chain axis. Exceedance uses the fraction
of each bucket above the threshold, including a partial intersected bucket.
"""

import jax
import jax.numpy as jnp


def exceedance_prob(probs: jax.Array, borders: jax.Array, threshold: jax.Array) -> jax.Array:
    """P(y > threshold) for (B, Q, bars) probabilities and (B,) thresholds."""
    threshold = jnp.clip(threshold, borders[0], borders[-1])
    lower, upper = borders[:-1], borders[1:]
    width = jnp.maximum(upper - lower, 1e-12)
    fraction_above = jnp.clip((upper - threshold[..., None]) / width, 0.0, 1.0)
    return (probs * fraction_above[:, None, :]).sum(-1)
