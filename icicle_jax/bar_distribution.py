"""Piecewise-constant predictive density over fixed value buckets.

The model produces one logit per bucket. Softmax gives bucket masses; dividing
by bucket widths gives densities. Training minimizes the negative log density
of each target.
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class BarDistribution:
    """Categorical over value buckets, read as a density.

    Bucket borders are fixed at construction. Training uses uniform borders over [0, 1].
    """

    borders: jax.Array  # sorted bucket edges, shape (num_bars + 1,)

    @property
    def num_bars(self) -> int:
        return self.borders.shape[-1] - 1

    @property
    def bucket_widths(self) -> jax.Array:
        return self.borders[1:] - self.borders[:-1]

    @property
    def bucket_midpoints(self) -> jax.Array:
        return (self.borders[:-1] + self.borders[1:]) / 2.0

    def map_to_bucket_idx(self, y: jax.Array) -> jax.Array:
        """Bucket index per target.

        Interior borders go to the lower bucket; targets outside the range clamp to the
        first/last bucket, so tail targets are absorbed by the edge buckets.
        """
        idx = jnp.searchsorted(self.borders, y, side="left") - 1
        return jnp.clip(idx, 0, self.num_bars - 1)

    def log_density(self, logits: jax.Array) -> jax.Array:
        """Log density per bucket: log-softmax mass minus log width."""
        return jax.nn.log_softmax(logits, axis=-1) - jnp.log(self.bucket_widths)

    def probs(self, logits: jax.Array) -> jax.Array:
        return jax.nn.softmax(logits, axis=-1)

    def mean(self, logits: jax.Array) -> jax.Array:
        """Posterior-predictive mean: bucket masses times bucket midpoints."""
        return self.probs(logits) @ self.bucket_midpoints

    def nll(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        """Negative log density of the target bucket.

        NaN targets contribute zero. Replace them before bucketing to keep
        the loss and its gradients finite.
        """
        nan_mask = jnp.isnan(y)
        y_safe = jnp.where(nan_mask, self.borders[0], y)
        target = self.map_to_bucket_idx(y_safe)
        picked = jnp.take_along_axis(self.log_density(logits), target[..., None], axis=-1)
        return jnp.where(nan_mask, 0.0, -picked[..., 0])
